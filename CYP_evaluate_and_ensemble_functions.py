import os
import re

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

import CYP_inhibition_functions


print(r"""
________________________________________________________________________________
 ______              _     _                                                        
|  ___ \            | |   (_)                                                       
| | _ | | ____  ____| | _  _ ____   ____                                            
| || || |/ _  |/ ___) || \| |  _ \ / _  )                                           
| || || ( ( | ( (___| | | | | | | ( (/ /                                            
|_||_||_|\_||_|\____)_| |_|_|_| |_|\____)                                                                            
 _                             _                                                    
| |                           (_)                                                   
| |      ____ ____  ____ ____  _ ____   ____                                        
| |     / _  ) _  |/ ___)  _ \| |  _ \ / _  |                                       
| |____( (/ ( ( | | |   | | | | | | | ( ( | |                                       
|_______)____)_||_|_|   |_| |_|_|_| |_|\_|| |                                       
                                      (_____|      _                                                     
   /\                                 _           | |                               
  /  \  _   _  ____ ____   ____ ____ | |_  ____ _ | |                               
 / /\ \| | | |/ _  |    \ / _  )  _ \|  _)/ _  ) || |                               
| |__| | |_| ( ( | | | | ( (/ /| | | | |_( (/ ( (_| |                               
|______|\____|\_|| |_|_|_|\____)_| |_|\___)____)____|                               
 _____       (_____|_     _                                                                                    
(____ \            | |   (_)                                                        
 _   \ \ ___   ____| |  _ _ ____   ____                                             
| |   | / _ \ / ___) | / ) |  _ \ / _  |                                            
| |__/ / |_| ( (___| |< (| | | | ( ( | |                                            
|_____/ \___/ \____)_| \_)_|_| |_|\_|| |                                            
                                 (_____|                                            



  __       ______ _     _ ______     _____       _     _ _     _      _             
 /  |     / _____) |   | (_____ \   (_____)     | |   (_) |   (_)_   (_)            
/_/ |    | /     | |___| |_____) )     _   ____ | | _  _| | _  _| |_  _  ___  ____  
  | |    | |      \_____/|  ____/     | | |  _ \| || \| | || \| |  _)| |/ _ \|  _ \ 
  | |_   | \_____   ___  | |         _| |_| | | | | | | | |_) ) | |__| | |_| | | | |
  |_(_)   \______) (___) |_|        (_____)_| |_|_| |_|_|____/|_|\___)_|\___/|_| |_|

____________________________________________________________________________________
____________________________________________________________________________________

                      Author: Benjamin Kachkowski Weiser
                        Date: September 6, 2023

This script embodies cutting-edge algorithms to augment traditional molecular docking, FITTED, processes
with machine learning for the prediction of Cytochrome P450 (CYP) enzyme inhibition.

Sections:

    1. Cleaned and combined test and train Pei sets with CYP_clean_files.ipnyb. Sets combined and then clustered to create new train and test sets
    2. Dock each ligand 5 times to its respective isoform using FITTED. Docked data can be found here: (to be inserted)
    3. Create analogue sets using FITTED. Create max train-test similarity using CYP_TC_DataSets.py
    4. Run RF with Feature Importances using max train-test similarity of 0.8 using ML_over_Tanimoto.py which calls CYP_inhibition_functions.py and Do_ML2.py
    5. Using these selected features run all ML models on all datasets using ML_over_Tanimoto.py which calls CYP_inhibition_functions.py and Do_ML2.py
    6. Use CYP_evaluate_and_ensemble.py which calls CYP_evaluate_and_ensemble_functions.py to make ensembles and evaluate and graph model performance

Please ensure you have all the required libraries installed.
For any issues or questions, please contact: benjamin.weiser@mail.mcgill.ca
Github: https://github.com/MoitessierLab/ML-augmented-docking-CYP-inhibition

____________________________________________________________________________________
____________________________________________________________________________________

""")

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)


def load_model(model_path):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model


def get_model_details(model_name):
    # extract isoform and TC from model name
    isoform, tc = re.search('XGB_(.*)-(\d+)', model_name).groups()
    # feature_selection_type is the word in XGB model after the number (like 80, 30, or 50)
    feature_selection_type = re.search('XGB_.*-\d+(.*)_model.json', model_name).group(1)
    return isoform, int(tc), feature_selection_type


def select_features(X_testtt, y, isoform, feature_selection_type):
    if feature_selection_type == 'all':
        X_test = X_testtt
    elif feature_selection_type == 'docked':
        X_test, y = CYP_inhibition_functions.select_features_tanimoto(X_testtt, y, isoform, features=3,
                                                                      drop_duplicated=False)
    elif feature_selection_type == 'ligand':
        X_test, y = CYP_inhibition_functions.select_features_tanimoto(X_testtt, y, isoform, features=2,
                                                                      drop_duplicated=False)
    return X_test, y


from sklearn.utils import resample
from sklearn.metrics import roc_auc_score


def bootstrap_auc(y_true, y_pred, n_bootstraps=1000):
    n = len(y_true)
    auc_scores = []
    for _ in range(n_bootstraps):
        indices = resample(
            np.arange(n), replace=True, n_samples=n
        )
        if len(np.unique(y_true[indices])) < 2:
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        auc_scores.append(score)
    auc_scores = np.array(auc_scores)
    return auc_scores


def bootstrap_auc_fixed_cluster(y_true, y_pred, cluster_size=10, n_bootstraps=1000):
    n = len(y_true)
    auc_scores = []

    if n % cluster_size != 0:
        print("Warning: data size is not divisible by cluster_size. Some data will be ignored.")

    n_clusters = n // cluster_size

    for _ in range(n_bootstraps):
        y_true_boot = []
        y_pred_boot = []

        for _ in range(n_clusters):
            # Choose a random cluster id
            cluster_id = np.random.randint(0, n_clusters)
            # Calculate the start and end index of this cluster
            start, end = cluster_id * cluster_size, (cluster_id + 1) * cluster_size
            # Append the samples in this cluster to the bootstrap sample
            y_true_boot.extend(y_true[start:end])
            y_pred_boot.extend(y_pred[start:end])

        score = roc_auc_score(y_true_boot, y_pred_boot)
        auc_scores.append(score)

    return np.array(auc_scores)


# Function to plot ROC curve
def plot_roc_curve(y_test, y_pred, isoform, feature_selection_type, plot=False):
    fpr, tpr, thresh = roc_curve(y_test, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)

    # Add bootstrap call here
    bootstrapped_scores = bootstrap_auc_fixed_cluster(y_test, y_pred[:, 1], cluster_size=10, n_bootstraps=1000)
    bootstrapped_scores_mean = np.mean(bootstrapped_scores)
    bootstrapped_scores_std = np.std(bootstrapped_scores)

    print(
        f'{feature_selection_type} for {isoform} Bootstrap clustered AUC: {bootstrapped_scores_mean:.4f} +/- {bootstrapped_scores_std:.3f}')
    bootstrap_info = f'Bootstrap clustered AUC: {bootstrapped_scores_mean:.3f} +/- {bootstrapped_scores_std:.3f}'

    if plot == True:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc})\n{bootstrap_info}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic for {isoform} using {feature_selection_type} features')
        plt.legend(loc="lower right")
        plt.show()

    return roc_auc, bootstrapped_scores_mean, bootstrapped_scores_std


# Function to predict output
def get_predictions(X_test, model):
    return model.predict_proba(X_test)


def compute_accuracy(y_test, y_pred):
    threshold = 0.5
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_class = (y_pred[:, 1] > threshold).astype(int)
    else:
        # then y pred class was already given
        y_pred_class = y_pred
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
    active_pos_rate = tp / (tp + fn)
    inactive_neg_rate = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return active_pos_rate, inactive_neg_rate, accuracy


def compute_thresholds(y_pred, y_test):
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []
    accuracies = {'actives': [], 'inactives': []}
    for threshold in thresholds:
        y_pred_class = (y_pred[:, 1] > threshold).astype(int)
        active_pos_rate, inactive_neg_rate, accuracy = compute_accuracy(y_test, y_pred_class)
        accuracies['actives'].append(active_pos_rate)
        accuracies['inactives'].append(inactive_neg_rate)
        f1_scores.append(f1_score(y_test, y_pred_class))

    # get threshold for 95% active accuracy
    for i in range(len(accuracies['actives'])):
        if accuracies['actives'][i] <= 0.95:
            threshold_95 = thresholds[i]
            break

    return thresholds, threshold_95, f1_scores


def display_optimal_stats(y_pred, y_test, thresholds, threshold_95, f1_scores, isoform, feature_selection_type,
                          plot=False):
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    y_pred_class = (y_pred[:, 1] > optimal_threshold).astype(int)
    active_pos_rate, inactive_neg_rate, accuracy = compute_accuracy(y_test, y_pred_class)
    print(f"Optimal threshold is: {optimal_threshold:.3f}, accuracy is: {accuracy:.3f}")
    print(f"Active rate at optimal threshold: {active_pos_rate:.3f}")
    print(f"Inactive rate at optimal threshold: {inactive_neg_rate:.3f}")

    y_pred_class = (y_pred[:, 1] > threshold_95).astype(int)
    active_acc_95, inactive_acc_95, accuracy_95 = compute_accuracy(y_test, y_pred_class)
    print(f"95% active threshold is: {threshold_95:.3f}, accuracy is: {accuracy_95:.3f}")
    print(f"Active rate at 95% active rate: {active_acc_95:.3f}")
    print(f"Inactive rate at 95% active rate: {inactive_acc_95:.3f}")

    if plot:
        # visualize the F1 score over all thresholds
        plt.figure(figsize=(10, 8))
        plt.plot(thresholds, f1_scores, label='F1 score')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title(
            f'Optimal Threshold Determination for \n {isoform} using {feature_selection_type} features using val set')
        plt.legend()
        plt.grid(True)
        plt.show()

        col_labels = ['Threshold', 'Accuracy', 'Active Rate', 'Inactive Rate']
        cell_text = [
            [f"{round(optimal_threshold, 2)}", f"{round(accuracy, 2)}", f"{round(active_pos_rate, 2)}",
             f"{round(inactive_neg_rate, 2)}"],
            [f"{round(threshold_95, 2)}", f"{round(accuracy_95, 2)}", f"{round(active_acc_95, 2)}",
             f"{round(inactive_acc_95, 2)}"]
        ]

        fig, ax = plt.subplots(1, 1)
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center')
        plt.show()
    return optimal_threshold, accuracy, active_pos_rate, inactive_neg_rate, threshold_95, accuracy_95, active_acc_95, inactive_acc_95


from sklearn.model_selection import train_test_split


def load_prep_data(pm, isoform, feature_selection_type, scaler, test_size=0.8, random_state=42):
    file_path = os.path.join(pm['dir'], '80TC_TESTSETS', '80TC_complete_test_set_' + isoform + '.csv')
    test_set_MT = pd.read_csv(file_path)

    X = test_set_MT.drop(['Activity', 'id'], axis=1)
    X = X.convert_dtypes()
    y = test_set_MT['Activity']

    X, y = select_features(X, y, isoform, feature_selection_type)
    ppp = scaler.get_feature_names_out()
    X = scaler.transform(X)

    # splitting the test set into a new test set and validation set
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X, y, X_test, y_test, X_val, y_val

def load_prep_data_analog(pm, isoform, tc, feature_selection_type, scaler, test_size=0.8, random_state=42):
    file_path = os.path.join(pm['dir'], 'Analog_Clusters', isoform + str(tc) + '.csv')
    test_set_MT = pd.read_csv(file_path)

    X = test_set_MT.drop(['Activity', 'id'], axis=1)
    X = X.convert_dtypes()
    y = test_set_MT['Activity']

    X, y = select_features(X, y, isoform, feature_selection_type)
    ppp = scaler.get_feature_names_out()
    X = scaler.transform(X)

    # splitting the test set into a new test set and validation set
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X, y, X_test, y_test, X_val, y_val



def load_test_set(pm, isoform, test_size=0.8, random_state=42):
    file_path = os.path.join(pm['dir'], '80TC_TESTSETS', '80TC_complete_test_set_' + isoform + '.csv')
    test_set_MT = pd.read_csv(file_path)

    X = test_set_MT.drop(['Activity', 'id'], axis=1)
    X = X.convert_dtypes()
    y = test_set_MT['Activity']

    # splitting the test set into a new test set and validation set
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X, y, X_test, y_test, X_val, y_val


def make_cyp_plots(dataframes, title, start_TC, end_TC):
    # Define the feature types
    feature_types = ['all', 'ligand', 'docked']

    # Create a range of Tanimoto coefficients
    tanimotos = list(range(start_TC, end_TC + 1, 10))

    # Define line types for feature types
    linestyles = ['-', '--', '-.', ':']

    # Define colors for isoforms
    isoform_colors = {'2D6': '#648fff', '1A2': '#674ce6', '2C9': '#dc267f', '2C19': '#fe6100', '3A4': '#ffb000'}

    # Create a figure
    fig, ax = plt.subplots(len(dataframes), figsize=(12, 18))

    # Iterate through the isoforms (dataframes)
    for isoform_index, (isoform, df) in enumerate(dataframes.items()):
        ax[isoform_index].set_title(f'{isoform}')
        ax[isoform_index].set_xlabel('Tanimoto coefficient')
        ax[isoform_index].set_ylabel('AUC')
        ax[isoform_index].set_ylim(0.50, 0.87)

        # Iterate through the feature types
        for feature_index, feature in enumerate(feature_types):
            auc_values = []
            for tc in tanimotos:
                try:
                    auc_values.append(df.loc[tc, f"roc_auc_{feature}"])
                except KeyError:
                    continue
            ax[isoform_index].plot(tanimotos[:len(auc_values)], auc_values, label=f'{isoform} - {feature}',
                                   linestyle=linestyles[feature_index], color=isoform_colors[isoform])

        # Add a smaller-sized legend
        ax[isoform_index].legend(fontsize='small', loc='lower right', ncol=2)

    plt.tight_layout()
    plt.show()


def make_cyp_plot(data, title, start_TC, end_TC, show=True):
    train_sizes = [[100, 200, 300, 400], [150, 250, 350, 450]]
    isoforms = list(data.keys())
    feature_types_names = ['All', 'Ligand', 'Docked']
    linestyles = ['-', '--', '-.', ':']
    isoform_colors = {'2D6': '#648fff', '1A2': '#674ce6', '2C9': '#dc267f', '2C19': '#fe6100', '3A4': '#ffb000'}

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 10))
    # make the font bigger
    fs = 20
    plt.rcParams.update({'font.size': fs})
    # make ticks labels bigger
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    plt.xlabel('Max train-test Tanimoto similarity', fontsize=fs)
    plt.ylabel('AUC', fontsize=fs)

    for isoform in isoforms:
        df = data[isoform]
        for feature_index, feature in enumerate(feature_types_names):
            auc_values = []
            dev_values = []  # to collect standard deviation values corresponding to auc_values
            tanimoto_values = []  # to collect tanimoto values corresponding to auc_values
            for tanimoto in range(start_TC, end_TC, 10):
                try:
                    auc_value = df.loc[tanimoto, f'roc_auc_{feature.lower()}']
                    auc_values.append(auc_value)
                    dev_value = df.loc[tanimoto, f'bootstrapped_scores_std_{feature.lower()}']
                    dev_values.append(dev_value)
                    tanimoto_values.append(tanimoto)  # save the tanimoto value
                except KeyError:
                    continue
            plt.errorbar(np.divide(tanimoto_values, 100), auc_values, yerr=dev_values,
                         label=f'{isoform} - {feature}', linestyle=linestyles[feature_index],
                         color=isoform_colors[isoform], capsize=4)

    legend_elements = [Line2D([0], [0], color='black', linestyle=linestyles[i], label=feature) for i, feature in
                       enumerate(feature_types_names)]
    legend_elements += [Line2D([0], [0], marker='.', color='black', linestyle='None', label='Train Data')]
    legend_elements += [Patch(facecolor=isoform_colors[isoform], label=isoform) for isoform in isoforms]

    plt.legend(handles=legend_elements, fontsize='medium', loc='lower right', ncol=1)
    plt.ylim(0.73, 0.95)

    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.locator_params(axis='y', nbins=5)

    if title != None:
        plt.title(title, fontsize=fs)

    if show:
        plt.show()


def make_cyp_plot_select(data, title, start_TC, end_TC, show=True):
    train_sizes = [[100, 200, 300, 400], [150, 250, 350, 450]]
    isoforms = list(data.keys())
    feature_types_names = ['All', 'Ligand', 'Docked']
    linestyles = ['-', '--', '-.', ':']
    isoform_colors = {'2D6': '#648fff', '1A2': '#674ce6', '2C9': '#dc267f', '2C19': '#fe6100', '3A4': '#ffb000'}

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 10))
    # make the font bigger
    fs = 20
    plt.rcParams.update({'font.size': fs})
    # make ticks labels bigger
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    plt.xlabel('Max train-test Tanimoto similarity', fontsize=fs)
    plt.ylabel('AUC', fontsize=fs)

    processed_isoforms = []
    for isoform in isoforms:
        # only plot 1A2 and 2C19
        if isoform != '1A2' and isoform != '3A4' and isoform != '2C19':
            continue

        processed_isoforms.append(isoform)
        df = data[isoform]
        for feature_index, feature in enumerate(feature_types_names):
            auc_values = []
            dev_values = []  # to collect standard deviation values corresponding to auc_values
            tanimoto_values = []  # to collect tanimoto values corresponding to auc_values
            for tanimoto in range(start_TC, end_TC, 10):
                try:
                    auc_value = df.loc[tanimoto, f'roc_auc_{feature.lower()}']
                    auc_values.append(auc_value)
                    dev_value = df.loc[tanimoto, f'bootstrapped_scores_std_{feature.lower()}']
                    dev_values.append(dev_value)
                    tanimoto_values.append(tanimoto)  # save the tanimoto value
                except KeyError:
                    continue
            plt.errorbar(np.divide(tanimoto_values, 100), auc_values, yerr=dev_values,
                         label=f'{isoform} - {feature}', linestyle=linestyles[feature_index],
                         color=isoform_colors[isoform], capsize=4)

    legend_elements = [Line2D([0], [0], color='black', linestyle=linestyles[i], label=feature) for i, feature in
                       enumerate(feature_types_names)]
    legend_elements += [Line2D([0], [0], marker='.', color='black', linestyle='None', label='Train Data')]
    legend_elements += [Patch(facecolor=isoform_colors[isoform], label=isoform) for isoform in processed_isoforms]

    plt.legend(handles=legend_elements, fontsize='medium', loc='lower right', ncol=1)

    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.locator_params(axis='y', nbins=5)

    plt.ylim(0.73, 0.95)
    if title != None:
        plt.title(title, fontsize=fs)

    if show:
        plt.show()

def make_cyp_plot_analog_select(data, title, start_TC, end_TC, show=True):
    import matplotlib.ticker as mticker
    train_sizes = {
        '2C19': [2254, 8447, 17861, 22455, 24856, 27134, 29775, 32425],
        '2C9': [2310, 10726, 19522, 23506, 25354, 27191, 29152, 30855],
        '2D6': [3165, 14863, 23296, 26869, 28687, 30322, 32035, 33570],
        '1A2': [2330, 8956, 17298, 21112, 22953, 24931, 27274, 29513],
        '3A4': [2068, 9023, 17547, 21596, 23907, 25807, 27773, 29530]
    }
    isoforms = list(data.keys())
    feature_types_names = ['All', 'Ligand', 'Docked']
    linestyles = ['-', '--', '-.', ':']
    isoform_colors = {'2D6': '#648fff', '1A2': '#674ce6', '2C9': '#dc267f', '2C19': '#fe6100', '3A4': '#ffb000'}

    sns.set(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 10))

    # make the font bigger
    fs = 20
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.set_xlabel('Max train-test Tanimoto similarity', fontsize=fs)
    ax1.set_ylabel('AUC', fontsize=fs)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Training Datasize', fontsize=fs)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelsize=fs)
    ax2.grid(False)

    processed_isoforms = []

    for isoform in isoforms:
        # only plot 1A2 and 2C19
        if isoform != '1A2' and isoform != '3A4' and isoform != '2C9':
            continue

        processed_isoforms.append(isoform)

        df = data[isoform]
        for feature_index, feature in enumerate(feature_types_names):
            auc_values, dev_values, tanimoto_values, train_sizes_values = [], [], [], []
            for tanimoto_index, tanimoto in enumerate(range(start_TC, end_TC, 10)):
                try:
                    auc_values.append(df.loc[tanimoto, f'roc_auc_{feature.lower()}'])
                    dev_values.append(df.loc[tanimoto, f'bootstrapped_scores_std_{feature.lower()}'])
                    tanimoto_values.append(tanimoto)  # save the tanimoto value
                    try:
                        train_sizes_values.append(train_sizes[isoform][tanimoto_index])  # get the training data size
                    except IndexError:
                        print(
                            f"Warning: No train size data available for {isoform} at index {tanimoto_index}. Skipping...")
                        continue
                except KeyError:
                    continue
            ax1.errorbar(np.divide(tanimoto_values, 100), auc_values, yerr=dev_values,
                         label=f'{isoform} - {feature}', linestyle=linestyles[feature_index],
                         color=isoform_colors[isoform], capsize=4)
            ax2.plot(np.divide(tanimoto_values, 100), train_sizes_values, 'o', color=isoform_colors[isoform], markersize=10)

    legend_elements = [Line2D([0], [0], color='black', linestyle=linestyles[i], label=feature) for i, feature in enumerate(feature_types_names)]
    legend_elements += [Line2D([0], [0], marker='.', color='black', linestyle='None', label='Train Data')]
    legend_elements += [Patch(facecolor=isoform_colors[isoform], label=isoform) for isoform in processed_isoforms]

    ax1.legend(handles=legend_elements, fontsize='medium', loc='lower right', ncol=1)
    ax1.set_ylim(0.60, 0.95)
    formatter = mticker.ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # to enable compact scientific notation
    ax2.yaxis.set_major_formatter(formatter)


    if title:
        ax1.set_title(title, fontsize=fs)

    if show:
        plt.show()

def make_cyp_plot_analog(data, title, start_TC, end_TC, show=True):
    import matplotlib.ticker as mticker
    train_sizes = {
        '2C19': [2254, 8447, 17861, 22455, 24856, 27134, 29775, 32425],
        '2C9': [2310, 10726, 19522, 23506, 25354, 27191, 29152, 30855],
        '2D6': [3165, 14863, 23296, 26869, 28687, 30322, 32035, 33570],
        '1A2': [2330, 8956, 17298, 21112, 22953, 24931, 27274, 29513],
        '3A4': [2068, 9023, 17547, 21596, 23907, 25807, 27773, 29530]
    }
    isoforms = list(data.keys())
    feature_types_names = ['All', 'Ligand', 'Docked']
    linestyles = ['-', '--', '-.', ':']
    isoform_colors = {'2D6': '#648fff', '1A2': '#674ce6', '2C9': '#dc267f', '2C19': '#fe6100', '3A4': '#ffb000'}

    sns.set(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 10))

    # make the font bigger
    fs = 15
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.set_xlabel('Max train-test Tanimoto similarity', fontsize=fs)
    ax1.set_ylabel('AUC', fontsize=fs)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Training Datasize', fontsize=fs)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelsize=fs)
    ax2.grid(False)

    processed_isoforms = []

    for isoform in isoforms:
        # only plot 1A2 and 2C19

        processed_isoforms.append(isoform)

        df = data[isoform]
        for feature_index, feature in enumerate(feature_types_names):
            auc_values, dev_values, tanimoto_values, train_sizes_values = [], [], [], []
            for tanimoto_index, tanimoto in enumerate(range(start_TC, end_TC, 10)):
                try:
                    auc_values.append(df.loc[tanimoto, f'roc_auc_{feature.lower()}'])
                    dev_values.append(df.loc[tanimoto, f'bootstrapped_scores_std_{feature.lower()}'])
                    tanimoto_values.append(tanimoto)  # save the tanimoto value
                    try:
                        train_sizes_values.append(train_sizes[isoform][tanimoto_index])  # get the training data size
                    except IndexError:
                        print(
                            f"Warning: No train size data available for {isoform} at index {tanimoto_index}. Skipping...")
                        continue
                except KeyError:
                    continue
            ax1.errorbar(np.divide(tanimoto_values, 100), auc_values, yerr=dev_values,
                         label=f'{isoform} - {feature}', linestyle=linestyles[feature_index],
                         color=isoform_colors[isoform], capsize=4)
            ax2.plot(np.divide(tanimoto_values, 100), train_sizes_values, 'o', color=isoform_colors[isoform], markersize=10)

    legend_elements = [Line2D([0], [0], color='black', linestyle=linestyles[i], label=feature) for i, feature in enumerate(feature_types_names)]
    legend_elements += [Line2D([0], [0], marker='.', color='black', linestyle='None', label='Train Data')]
    legend_elements += [Patch(facecolor=isoform_colors[isoform], label=isoform) for isoform in processed_isoforms]

    ax1.legend(handles=legend_elements, fontsize='small', loc='lower right', ncol=1)
    ax1.set_ylim(0.60, 0.95)
    formatter = mticker.ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # to enable compact scientific notation
    ax2.yaxis.set_major_formatter(formatter)

    if title:
        ax1.set_title(title, fontsize=fs)

    if show:
        plt.show()


import matplotlib.pyplot as plt
import seaborn as sns


def plot_ensemble(isoform_stats):
    import matplotlib.patches as mpatches
    isoform_colors = {'2D6': '#648fff', '1A2': '#674ce6', '2C9': '#dc267f', '2C19': '#fe6100', '3A4': '#ffb000'}

    # Split your isoform_stats into separate isoforms and measures
    separated_stats = {}
    for k, v in isoform_stats.items():
        isoform = k.replace('all', '').replace('docked', '').replace('ligand', '').replace('combined', '')
        if isoform not in separated_stats:
            separated_stats[isoform] = [None, None, None, None]
        if 'all' in k:
            separated_stats[isoform][2] = v
        elif 'docked' in k:
            separated_stats[isoform][0] = v
        elif 'ligand' in k:
            separated_stats[isoform][1] = v
        elif 'combined' in k:
            separated_stats[isoform][3] = v

    # Set the style for a professional looking plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 10))

    bar_width = 0.25  # the width of the bars
    r = range(len(separated_stats))

    # Plot each bar for each measure ('all', 'docked', and 'ligand')
    for i, k in enumerate(separated_stats):
        all_, docked_, ligand_, combined_ = separated_stats[k]
        plt.bar(i - 1.5 * bar_width, all_[0], width=bar_width, color=isoform_colors[k], yerr=all_[1], capsize=7,
                label='All features ensemble' if i == 0 else "", hatch='-\\\\')
        plt.bar(i - 0.5 * bar_width, docked_[0], width=bar_width, color=isoform_colors[k], yerr=docked_[1], capsize=7,
                label='Docked features ensemble' if i == 0 else "", hatch='//')
        plt.bar(i + 0.5 * bar_width, ligand_[0], width=bar_width, color=isoform_colors[k], yerr=ligand_[1], capsize=7,
                label='Ligand features ensemble' if i == 0 else "", hatch='++')
        plt.bar(i + 1.5 * bar_width, combined_[0], width=bar_width, color=isoform_colors[k], yerr=combined_[1],
                capsize=7, label='Ensemble of all models' if i == 0 else "", hatch='')
    # font size
    fs = 14
    plt.ylabel('Bootstrapped Scores Mean AUC', fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlabel('Isoforms', fontsize=fs)
    # plt.title('CYP Inhibition Ensemble Models for Isoforms and Feature Types for TC =< 0.4', fontsize=fs)
    plt.xticks([r for r in range(len(separated_stats))], list(separated_stats.keys()), rotation='vertical', fontsize=fs)
    labels = ['Docked features ensemble', 'Ligand features ensemble', 'All features ensemble', 'Ensemble of all models']
    patches = [mpatches.Patch(facecolor='grey', hatch='-\\\\', label=labels[0]),
               mpatches.Patch(facecolor='grey', hatch='//', label=labels[1]),
               mpatches.Patch(facecolor='grey', hatch='++', label=labels[2]),
               mpatches.Patch(facecolor='grey', hatch='', label=labels[3])]
    plt.legend(handles=patches, handlelength=3, handletextpad=1, handleheight=2, fontsize=fs)


    plt.ylim([0.70, 0.95])
    plt.show()


def weighted_ensemble(isoform_predictions, val_scores_df, name):
    # Initialize the ensemble predictions to zeros
    y_ensemble = np.zeros_like(isoform_predictions[name][next(iter(isoform_predictions[name]))][
                                   0])  # Assuming each prediction array has the same shape
    # subract 0.5 from all val scores to make a random prediction give 0 weight
    val_scores_df = val_scores_df - 0.75
    # if val_scores_df a element is less than zero, set it to zero
    val_scores_df[val_scores_df < 0] = 0

    # Sum of AUCs to normalize the weights
    total_auc = sum(val_scores_df.loc[name].values)

    TC_40 = True
    for tc in isoform_predictions[name]:
        if (TC_40 == True) and ('80' in tc or '70' in tc or '60' in tc or '50' in tc):
            print('TC 40 only')
            continue
        # Get the corresponding AUC for this 'tc'
        auc_weight = val_scores_df.at[name, tc]

        # Compute the weighted prediction
        y_ensemble += (auc_weight / total_auc) * np.squeeze(isoform_predictions[name][tc])
    return y_ensemble


def weighted_ensemble_isoform(isoform_predictions, val_scores_df, isoform):
    # Initialize the ensemble predictions to zeros
    y_ensemble = np.zeros_like(isoform_predictions[next(key for key in isoform_predictions.keys() if isoform in key)][
                                   next(iter(isoform_predictions[
                                                 next(key for key in isoform_predictions.keys() if isoform in key)]))][
                                   0])
    # Sum of AUCs to normalize the weights
    total_auc = 0
    # subract 0.5 from all val scores to make a random prediction give 0 weight
    val_scores_df = val_scores_df - 0.75
    # if val_scores_df a element is less than zero, set it to zero
    val_scores_df[val_scores_df < 0] = 0
    for index, _ in val_scores_df.iterrows():
        if isoform in index:
            total_auc += val_scores_df.loc[index].values.sum()

    if total_auc == 0:  # add this check for zero total_auc
        raise ValueError(f"No models found for the isoform {isoform} with non-zero AUC.")

    for name in isoform_predictions:
        if isoform in name:

            TC_40 = True
            for tc in isoform_predictions[name]:
                if (TC_40 == True) and ('80' in tc or '70' in tc or '60' in tc or '50' in tc):
                    print('TC 40 only')
                    continue
                # Get the corresponding AUC for this 'tc'
                auc_weight = val_scores_df.at[name, tc]

                # Compute the weighted prediction
                y_ensemble += (auc_weight / total_auc) * np.squeeze(isoform_predictions[name][tc])
    return y_ensemble
