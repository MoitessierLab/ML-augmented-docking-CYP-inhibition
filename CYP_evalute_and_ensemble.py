import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

from CYP_evaluate_and_ensemble_functions import load_model, load_prep_data, get_predictions, compute_accuracy, \
    compute_thresholds, display_optimal_stats, plot_roc_curve, weighted_ensemble, weighted_ensemble_isoform, \
    make_cyp_plot, make_cyp_plot_select, plot_ensemble, get_model_details, load_test_set, load_prep_data_analog, \
    make_cyp_plot_analog, make_cyp_plot_analog_select

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

# Main thing you actually change!

'''
This code avaluates XGB models and creates ensembles as well as investigates threshold tunning. 
Not test sets loaded from 80TC_TESTSETS folder and for analog test sets from the Analog_Clusters folder. 

SETTINGS: 
    select_folder: 'same' or 'alldata' or 'analog' (Correspond to XGB models with different training data)
                XGB_MaxTCModelsamedModels is for MaxTC cluster with same datasize
                XGB_MaxTCModels with different data sizes
                XGB_analogModels is for analog models
    run_save_df: 1 or 0 (1 to run and save df, 0 to load df)
    load_df: 1 or 0 (1 to load df, 0 to run and save df)
    run_ensemble: 1 or 0 (1 to run ensemble, 0 to load ensemble. Note: run seperealte from run_save_df)
    load_ensemble: 1 or 0 (1 to load ensemble, 0 to run ensemble)
'''

select_folder = 'alldata'
run_save_df = 1
load_df = 1
run_ensemble = 0
load_ensemble = 0
##################### END OF USER INPUTS #####################

pm = {'dir': '/home/weiser/PYTHON/EnsembleLearning/CSVFiles/TanimotoCSV_Final'}
if select_folder == 'same':
    models_dir = "/home/weiser/PYTHON/EnsembleLearning/CSVFiles/TanimotoCSV_Final/XGB_MaxTCsamedModels"
    # title = 'CYP Inhibition AUC over Similarity of Train and Test Sets Using Consitent Data Size'
    title = None

if select_folder == 'alldata':
    models_dir = "/home/weiser/PYTHON/EnsembleLearning/CSVFiles/TanimotoCSV_Final/XGB_MaxTCModels"
    # title = 'CYP Inhibition AUC over Similarity of Train and Test Sets'
    title = None

if select_folder == 'analog':
    print('Using analog models')
    models_dir = "/home/weiser/PYTHON/EnsembleLearning/CSVFiles/TanimotoCSV_Final/XGB_analogModels"
    # title = 'CYP Inhibition AUC over Similarity of Train and Test Sets'
    title = None

model_paths = glob.glob(os.path.join(models_dir, 'XGB_*.json'))


def main():
    isoforms = ['2C19', '2C9', '1A2', '2D6', '3A4']
    dfs = {isoform: pd.DataFrame(
        columns=['roc_auc_all', 'bootstrapped_scores_mean_all', 'bootstrapped_scores_std_all', 'roc_auc_ligand',
                 'bootstrapped_scores_mean_ligand', 'bootstrapped_scores_std_ligand', 'roc_auc_docked',
                 'bootstrapped_scores_mean_docked', 'bootstrapped_scores_std_docked']) for isoform in isoforms}
    isoform_predictions = {}
    fitted_auc_iso = {'3A4': 0.7414807387827167, '2C9': 0.6868582600195503, '2C19': 0.6156633557328691,
                      '2D6': 0.5512369688341652, '1A2': 0.6769454157333774}

    folder_name = os.path.basename(os.path.normpath(models_dir))
    output_dir = 'CYP_Threshold_test_data_' + folder_name
    if run_save_df == 1:
        if not model_paths:
            raise FileNotFoundError('No XGB_*.json files found in directory: ', models_dir)

        val_scores = {}
        for i, model_path in enumerate(model_paths):
            model_name = os.path.basename(model_path)

            # if model does not contain '80' or '50' then skip
            '''if '80' not in model_name: #and '70' not in model_name:
                continue
            if 'all' not in model_name and 'ligand' not in model_name:
                continue
            if '2D6' in model_name:
                continue'''

            print(str(i) + '/' + str(len(model_path)) + '\n' + "--------NEW MODEL--------")
            print("Model: {}".format(model_name))

            isoform, tc, feature_selection_type = get_model_details(model_name)
            # load scaler transform with name isoform-tcfeature_selection_type_scaler.pkl
            scaler_path = os.path.join(models_dir, isoform + '-' + str(tc) + feature_selection_type + '_scaler.pkl')
            scaler = joblib.load(scaler_path)
            name = isoform + feature_selection_type

            model = load_model(model_path)
            print("Number of features for model : {}".format(model.feature_importances_.shape[0]))
            print(model.get_booster().feature_names)
            if select_folder == 'analog':
                X_all, y_all, X_test, y_test, X_val, y_val = load_prep_data_analog(pm, isoform, tc,
                                                                                   feature_selection_type, scaler,
                                                                                   test_size=0.8, random_state=42)
            else:
                X_all, y_all, X_test, y_test, X_val, y_val = load_prep_data(pm, isoform, feature_selection_type, scaler,
                                                                            test_size=0.8, random_state=42)

            y_pred_all = get_predictions(X_all, model)
            y_pred_test = get_predictions(X_test, model)
            y_pred_val = get_predictions(X_val, model)

            # get validation accuracy for weigthed ensemble
            fpr, tpr, thresh = roc_curve(y_val, y_pred_val[:, 1])
            roc_auc_val = auc(fpr, tpr)

            if name not in isoform_predictions:
                isoform_predictions[name] = {}
                val_scores[name] = {}

            if tc not in val_scores[name]:
                isoform_predictions[name][tc] = []

            val_scores[name][tc] = roc_auc_val
            isoform_predictions[name][tc].append(y_pred_test)

            # Compute the accuracy
            active_pos_rate, inactive_neg_rate, accuracy = compute_accuracy(y_all, y_pred_all)
            print(f"Accuracy is: {accuracy:.3f}")
            print(f"Active rate: {active_pos_rate:.3f}")
            print(f"Inactive rate: {inactive_neg_rate:.3f}")

            thresholds, threshold_95, f1_scores = compute_thresholds(y_pred_val, y_val)
            optimal_threshold, accuracy, active_pos_rate, inactive_neg_rate, threshold_95, accuracy_95, active_acc_95, \
                inactive_acc_95, senstivity, specificity, senstivity95, specificity95 = display_optimal_stats(y_pred_test, y_test, thresholds, threshold_95, f1_scores,
                                                        isoform, f'{tc}-' + feature_selection_type, plot=False)

            roc_auc, bootstrapped_scores_mean, bootstrapped_scores_std = plot_roc_curve(y_all, y_pred_all, isoform,
                                                                                        f'{tc}-' + feature_selection_type)

            roc_auc, bootstrapped_scores_mean, bootstrapped_scores_std = round(roc_auc, 4), round(
                bootstrapped_scores_mean, 4), round(bootstrapped_scores_std, 4)
            fitted_auc = fitted_auc_iso[isoform]
            fitted_auc = round(fitted_auc, 4)
            auc_improvement = round((roc_auc - fitted_auc) * 100, 2)

            # Save the stats to the dataframe
            df = dfs[isoform]
            if tc not in df.index:
                df.loc[tc, :] = [None] * len(df.columns)
            df.loc[tc, f'fitted_auc_{feature_selection_type}'] = fitted_auc
            df.loc[tc, f'auc_improvement_{feature_selection_type}'] = auc_improvement
            df.loc[tc, f'roc_auc_{feature_selection_type}'] = roc_auc
            df.loc[tc, f'bootstrapped_scores_mean_{feature_selection_type}'] = bootstrapped_scores_mean
            df.loc[tc, f'bootstrapped_scores_std_{feature_selection_type}'] = bootstrapped_scores_std
            # save optimal_threshold, accuracy, active_pos_rate, inactive_neg_rate, threshold_95, accuracy_95, active_acc_95, inactive_acc_95
            df.loc[tc, f'optimal_threshold_{feature_selection_type}'] = optimal_threshold
            df.loc[tc, f'accuracy_{feature_selection_type}'] = accuracy
            df.loc[tc, f'active_pos_rate_{feature_selection_type}'] = active_pos_rate
            df.loc[tc, f'inactive_neg_rate_{feature_selection_type}'] = inactive_neg_rate
            df.loc[tc, f'threshold_95_{feature_selection_type}'] = threshold_95
            df.loc[tc, f'accuracy_95_{feature_selection_type}'] = accuracy_95
            df.loc[tc, f'active_acc_95_{feature_selection_type}'] = active_acc_95
            df.loc[tc, f'inactive_acc_95_{feature_selection_type}'] = inactive_acc_95

        # save df and implement an option to load from CYP_Threshold_test_data folder
        # get last bit of path of models_dir
        os.makedirs(output_dir, exist_ok=True)
        # save val_scores to a csv
        val_scores_df = pd.DataFrame.from_dict(val_scores, orient='index')
        val_scores_df.to_csv(os.path.join(output_dir, 'val_scores.csv'))

        # Save predictions with additional tc layer
        for isoform, df in dfs.items():
            df.to_csv(os.path.join(output_dir, f'{isoform}_df.csv'))
            for feature_type in ['ligand', 'docked', 'all']:
                name = isoform + feature_type
                if name in isoform_predictions:
                    for tc in isoform_predictions[name]:
                        save_path = os.path.join(output_dir, f'{name}_{tc}_predictions.npy')
                        np.save(save_path, isoform_predictions[name][tc])

    # Load predictions
    if load_df == 1:
        for isoform in isoforms:
            csv_path = os.path.join(output_dir, f'{isoform}_df.csv')
            df = pd.read_csv(csv_path, index_col=0)
            dfs[isoform] = df
            for feature_type in ['ligand', 'docked', 'all']:
                name = isoform + feature_type
                if name not in isoform_predictions:
                    isoform_predictions[name] = {}
                if feature_type == 'all':
                    data_95 = {
                        'roc_auc': round(df.loc[40, f'roc_auc_{feature_type}'], 2),
                        'threshold_95': round(df.loc[40, f'threshold_95_{feature_type}'], 2),
                        'accuracy_95': round(df.loc[40, f'accuracy_95_{feature_type}'], 2),
                        'active_acc_95': round(df.loc[40, f'active_acc_95_{feature_type}'], 2),
                        'inactive_acc_95': round(df.loc[40, f'inactive_acc_95_{feature_type}'], 2),
                    }

                    print(isoform, ': ', data_95)

                prediction_files = glob.glob(os.path.join(output_dir, f'{name}_*_predictions.npy'))
                for f in prediction_files:
                    tc = os.path.basename(f).split('_')[1]  # Extracting tc from the filename
                    isoform_predictions[name][tc] = np.load(f)

            # for each isoform do df.loc[30, f'roc_auc_all'] - df.loc[30, f'roc_auc_ligand']
            improvement = df.loc[30, 'roc_auc_all'] - df.loc[30, 'roc_auc_ligand']
            print(isoform, 'All feature - Ligand Features AUC improvement: ', round(improvement, 4))
        val_scores = pd.read_csv(os.path.join(output_dir, 'val_scores.csv'), index_col=0)

    if select_folder == 'analog':
        make_cyp_plot_analog_select(dfs, title, start_TC=10, end_TC=90)
        make_cyp_plot_analog(dfs, title, start_TC=10, end_TC=90)
    elif select_folder == 'same':
        make_cyp_plot(dfs, title, start_TC=20, end_TC=90)
        make_cyp_plot_select(dfs, title, start_TC=20, end_TC=90)
    else:
        make_cyp_plot(dfs, title, start_TC=30, end_TC=90)
        make_cyp_plot_select(dfs, title, start_TC=30, end_TC=90)

    if run_ensemble == 1:
        print("--------ENSEMBLE MODEL--------")
        isoform_stats = {}  # This new variable stores the bootstrapped scores for each isoform.
        for name in isoform_predictions:
            # get isoform from name by deleting all or docked or ligand from name
            isoform = name.replace('all', '').replace('docked', '').replace('ligand', '')
            _, _, _, y_test, _, _ = load_test_set(pm, isoform)

            y_ensemble = weighted_ensemble(isoform_predictions, val_scores, name)
            # print('shapes', y_ensemble.shape, y_test.shape)
            # if first dimension of y_ensemble and y_test are not the same
            if y_ensemble.shape[0] != y_test.shape[0]:
                print('y_ensemble and y_test are not the same shape')
            roc_auc, bootstrapped_scores_mean, bootstrapped_scores_std = plot_roc_curve(y_test, y_ensemble, name,
                                                                                        'Ensemble', plot=False)
            isoform_stats[name] = [round(bootstrapped_scores_mean, 4), round(bootstrapped_scores_std, 4)]

        print(isoform_stats)
        # Now we use the modified weighted_ensemble() function for the ensemble with more granular combinations
        for isoform in isoforms:
            y_ensemble_combined = weighted_ensemble_isoform(isoform_predictions, val_scores, isoform)
            _, _, _, y_test, _, _ = load_test_set(pm, isoform)
            roc_auc, bootstrapped_scores_mean, bootstrapped_scores_std = plot_roc_curve(y_test, y_ensemble_combined,
                                                                                        isoform, 'Combined Ensemble',
                                                                                        plot=False)
            isoform_stats[isoform + 'combined'] = [round(bootstrapped_scores_mean, 4),
                                                   round(bootstrapped_scores_std, 4)]
        with open(os.path.join(output_dir, 'isoform_ensemble_stats.json'), 'w') as fp:
            json.dump(isoform_stats, fp)
        # also save as a csv
        isoform_stats_df = pd.DataFrame.from_dict(isoform_stats, orient='index')
        isoform_stats_df.to_csv(os.path.join(output_dir, 'isoform_ensemble_stats.csv'))
        plot_ensemble(isoform_stats)

    if load_ensemble == 1:
        with open(os.path.join(output_dir, 'isoform_ensemble_stats.json'), 'r') as fp:
            isoform_stats = json.load(fp)

        plot_ensemble(isoform_stats)
        # from isoform stats compare increase from fitted_auc
        fitted_auc_iso = {'3A4': 0.7414807387827167, '2C9': 0.6868582600195503, '2C19': 0.6156633557328691,
                          '2D6': 0.5512369688341652, '1A2': 0.6769454157333774}
        auc_improvement_ensemble = {}
        for isoform in isoforms:
            fitted_auc = fitted_auc_iso[isoform]
            fitted_auc = round(fitted_auc, 4)
            for feature_type in ['all', 'ligand', 'docked', 'combined']:
                name = isoform + feature_type
                auc_improvement = round((isoform_stats[name][0] - fitted_auc) * 100, 2)
                if name not in auc_improvement_ensemble:
                    auc_improvement_ensemble[name] = {}
                auc_improvement_ensemble[name] = auc_improvement
        print(auc_improvement_ensemble)
        # save auc_improvement_ensemble to csv
        auc_improvement_ensemble_df = pd.DataFrame.from_dict(auc_improvement_ensemble, orient='index')
        auc_improvement_ensemble_df.to_csv(os.path.join(output_dir, 'auc_improvement_ensemble.csv'))


if __name__ == '__main__':
    main()
