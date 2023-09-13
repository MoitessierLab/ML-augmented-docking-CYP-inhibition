import copy
import os

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import tensorflow as tf
from tensorflow import keras

keras.backend.clear_session()
import random

global id_in_test_train
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from tqdm import tqdm

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


# Set seed
seed = 1
os.environ['PYTHONHASHEDSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

'''Here's a high level overview of what this script does:
load_parameters(): This function returns a dictionary of parameters required for the operation of the script.
Next, a directory is created if it doesn't already exist.

Loop over different Cytochrome P450 isoforms. For each isoform:

a. Read in SMILES (Simplified Molecular Input Line Entry System) strings for inactive and active states of the isoform.
b. Depending on the parameters specified, it reads and filters molecules that have been docked / scored. It ensures all the SMILE strings used have been docked before.
c. If needed, it can also limit the dataset to a specific smaller number of molecules, or adjust the datasets based on the clusters.
d. The script then constructs a test set of molecules based on specified conditions (like Tanimoto coefficient).
e. The script creates a training set, ensuring none of the samples are in the test set, and the most similar one to the test set has a Tanimoto similarity below a certain threshold. Different training sets are created for various Tanimoto thresholds.

It saves detailed information about the training sets and Tanimoto coefficients to a CSV file.
Finally, it plots the size of the train set against different Tanimoto coefficients and saves this plot as a PNG file.
Overall this script is a mix of data preparation, filtering based on the application of certain conditions and measures (Tanimoto coefficient calculation) and basic exploratory data analysis (creation of plots). 
'''
def load_parameters():
    pm = {'Project_name': 'feb_17_Tan',
          'dir': '/home/weiser/PYTHON/EnsembleLearning/CSVFiles/AllCSVs',
          'data_dir': '/Results_CSV/',
          'fig_dir': 'Figures',
          'model_dir': 'Models',
          'CYPs': ['2D6' , '2C19','3A4','1A2', '2C9' ],
          'num_test_set_clusters': 100, # number of clusters to make test set from
          'test_set_cluster_size': 10, # number of molecules to take from each cluster for test set
          'use_Been_Docked': 1, # 1 to check that are only using molecules that have been docked. Use this one
          'use_Been_Docked_all': 0,
          'use_some_data': 0, # Use only 500 molecules for testing
          'clusters_100': 0 # to use clusters of 100 to get molecules filtered from REDUCE in FITTED if not using FITTED csv data
          }
    return pm


tanimoto = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
# make a dataframe with columns for each tanimoto value
# make a dataframe with columns for each tanimoto value


pm = load_parameters()
DATA_DIR = pm['data_dir']
dir_path = pm['dir'] + DATA_DIR
# make directory for tanimoto smiles csv in pm['dir']
if not os.path.exists(pm['dir'] + '/Clusters_Max_TC'):
    os.makedirs(pm['dir'] + '/Clusters_Max_TC')
for isoform in pm['CYPs']:
    print('Isoform: ', isoform)
    tanimoto_df = pd.DataFrame(columns=tanimoto)
    docked = []

    # loop over files in directory and subdirectories

    i_to_smiles = pd.read_csv(
        '/home/weiser/PYTHON/EnsembleLearning/CSVFiles/TanimotoCSV_Final' + '/Smiles/' + isoform + '-inactive.smi',
        names=['id', 'smiles'], header=None, sep='\s+')
    # i_to_smiles_ids = list(i_to_smiles.iloc[:, 0])
    a_to_smiles = pd.read_csv(
        '/home/weiser/PYTHON/EnsembleLearning/CSVFiles/TanimotoCSV_Final' + '/Smiles/' + isoform + '-active.smi',
        names=['id', 'smiles'], header=None, sep='\s+')
    # a_to_smiles_ids = list(a_to_smiles.iloc[:, 0])
    print('Loaded smiles')

    # Been_docked to compare with docked files and only use docked molecule in clusters
    if pm['use_Been_Docked'] == 1:
        activeData = pd.read_csv(
            '/home/weiser/PYTHON/EnsembleLearning/CSVFiles/TanimotoCSV_Final/Results_CSV/' + isoform + '/' + isoform + '-docked-actives-scored' + '.csv')
        activeData['Activity'] = 1.0

        inactiveData = pd.read_csv(
            '/home/weiser/PYTHON/EnsembleLearning/CSVFiles/TanimotoCSV_Final/Results_CSV/' + isoform + '/' + isoform + '-docked-inactives-scored' + '.csv')
        inactiveData['Activity'] = 0.0
        docked_combined = pd.concat([activeData, inactiveData], ignore_index=True)
        docked_combined['id'] = docked_combined['Molecule Name'].str.extract('(\d+)', expand=False).astype(int)

        allsmiles = pd.concat([i_to_smiles, a_to_smiles])
        #delete any duplicates and print size
        allsmiles = allsmiles.drop_duplicates(subset=['smiles'])
        #print
        print('Size of all smiles after dropping duplicates: ', len(allsmiles))

        # print size of all smiles
        print('Size of all smiles: ', len(allsmiles))
        # make it so that it only uses the 'id; that have been docked
        allsmiles['been_docked'] = allsmiles['id'].isin(docked_combined['id'])
        allsmiles = allsmiles[allsmiles['been_docked'] == True]
        # drop been_docked column
        allsmiles = allsmiles.drop(columns=['been_docked'])
        # print size of all smiles
        print('Size of all smiles that have been docked: ', len(allsmiles))

    if pm['use_Been_Docked_all'] == 1:

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                # check if file is a CSV
                if file.endswith('.csv'):
                    # construct file path
                    file_path = os.path.join(root, file)
                    # if path contains '\Final\
                    if 'Final' in file_path:
                        if '80' in file_path:
                            print('Loading: ', file_path)
                            # read in CSV data as dataframe and append to list
                            df = pd.read_csv(file_path)
                            docked = pd.concat([docked, df], ignore_index=True)

        # concatenate dataframes into a single dataframe
        docked_combined = pd.concat([activeData, inactiveData], ignore_index=True)

        docked_combined['id'] = docked_combined['Molecule Name'].str.extract('(\d+)', expand=False).astype(int)
        i_to_smiles['been_docked'] = i_to_smiles['id'].isin(docked_combined['id'])
        a_to_smiles['been_docked'] = a_to_smiles['id'].isin(docked_combined['id'])
        print('Done with been_docked')

        num_i_docked = i_to_smiles['been_docked'].sum()
        num_a_docked = a_to_smiles['been_docked'].sum()
        percent_a_docked = num_a_docked / len(a_to_smiles)
        percent_i_docked = num_i_docked / len(i_to_smiles)

        print('A frac:', percent_a_docked, 'I frac:', percent_i_docked)

        ##############################################################################################################
        allsmiles = pd.concat([i_to_smiles, a_to_smiles])
        # drop if 'been_docked' is False
        allsmiles = allsmiles[allsmiles['been_docked'] == True]
        # drop been_docked column
        allsmiles = allsmiles.drop(columns=['been_docked'])
        # drop any in docked_combined that are not in allsmiles
        # pick an id name from Docked_combined and call it test set origin.
        # get all uniques docked ids from docked_combined
        print('Size of docked_combined before:', len(docked_combined))
        docked_combined = docked_combined[docked_combined['id'].isin(allsmiles['id'])]
        print('Size of docked_combined after:', len(docked_combined))

    if pm['use_some_data'] == 1:
        allsmiles = allsmiles.head(500)

    if pm['clusters_100'] == 1:
        # to use clusters of 100 to get molecules filtered from REDUCE in FITTED if not using FITTED csv data
        # read cluster_list_1a2-actives_100.txt
        a_cluster_list = pd.read_csv(
            '/home/weiser/PYTHON/EnsembleLearning/CSVFiles/TanimotoCSV_Final/Clusters_100/cluster_list_' + isoform.lower() + '-actives_100.txt',
            header=None, names=['id'])
        # read cluster_list_1a2-inactives_100.txt
        i_cluster_list = pd.read_csv(
            '/home/weiser/PYTHON/EnsembleLearning/CSVFiles/TanimotoCSV_Final/Clusters_100/cluster_list_' + isoform.lower() + '-inactives_100.txt',
            header=None, names=['id'])
        # concatenate cluster_list_1a2-actives_100.txt and cluster_list_1a2-inactives_100.txt
        cluster_list = pd.concat([a_cluster_list, i_cluster_list])
        print('Size of allsmiles before 100:', len(allsmiles))
        # filter out allsmiles that are not in cluster_list
        allsmiles = allsmiles[allsmiles['id'].isin(cluster_list['id'])]
        print('Size of allsmiles after 100:', len(allsmiles))

    # Make Test Set
    ##############################################################################################################
    print('Making Test Set')
    test_set = pd.DataFrame(columns=['smiles', 'tanimoto_coefficient', 'id'])
    for i in tqdm(range(pm['num_test_set_clusters']), total=pm['num_test_set_clusters'], desc="Making Test Set"):
        origin_seed = random.randint(0, len(allsmiles)) - 1
        test_set_origin = allsmiles['smiles'].iloc[
            origin_seed]  # pick a random id from docked_ids to be the test set origin
        # Get the tanimoto similarity between the test set origin and all uniques docked smiles
        # keep these 500 and call them the test ids
        origin_mol = AllChem.MolFromSmiles(test_set_origin)
        origin_bit = AllChem.GetMorganFingerprintAsBitVect(origin_mol, radius=2, nBits=2048)
        origin_tanimoto_coefficient = pd.DataFrame(columns=['smiles', 'tanimoto_coefficient'])
        for test_compound in allsmiles['smiles']:
            test_mol = Chem.MolFromSmiles(test_compound)
            test_bit = AllChem.GetMorganFingerprintAsBitVect(test_mol, radius=2, nBits=2048)
            tanimoto_coefficient = DataStructs.TanimotoSimilarity(origin_bit, test_bit)
            # add tano to origin_tanimoto_coefficient and test_compound
            add = pd.DataFrame({'smiles': [test_compound], 'tanimoto_coefficient': [tanimoto_coefficient]})
            origin_tanimoto_coefficient = pd.concat([origin_tanimoto_coefficient, add])

        # add id column of allsmiles to origin_tanimoto_coefficient and merge on smiles
        test_set_cluster = origin_tanimoto_coefficient.merge(allsmiles, on='smiles')
        # sort origin_tanimoto_coefficient by tanimoto_coefficient and take top 500
        test_set_cluster = test_set_cluster.sort_values(by=['tanimoto_coefficient'], ascending=False)
        test_set_cluster = test_set_cluster.head(pm['test_set_cluster_size'])
        test_set = pd.concat([test_set, test_set_cluster], ignore_index=True)

    # get size for test set dataframe
    print('Size of test set:', len(test_set))
    test_set.to_csv(pm['dir'] + '/Clusters_Max_TC/' + 'test_set_' + isoform + '.csv', index=False)

    # Make Train Set
    ##############################################################################################################

    print('Making Train Set')
    # drop test set from Allsmiles
    train_smiles = copy.deepcopy(allsmiles)
    # drop test set from train_smiles
    train_smiles = train_smiles[~train_smiles['smiles'].isin(test_set['smiles'])]

    most_similar_to_in_train = pd.DataFrame(columns=['smiles', 'tanimoto_coefficient'])
    for train_compound in tqdm(train_smiles['smiles'], total=len(train_smiles), desc="Making Train Set"):
        train_mol = Chem.MolFromSmiles(train_compound)
        train_bit = AllChem.GetMorganFingerprintAsBitVect(train_mol, radius=2, nBits=2048)
        train_simularities = pd.DataFrame(columns=['tanimoto_coefficient'])
        for test_compound in test_set['smiles']:
            # Convert the train compound to a mol object
            test_mol = Chem.MolFromSmiles(test_compound)
            test_bit = AllChem.GetMorganFingerprintAsBitVect(test_mol, radius=2, nBits=2048)
            tanimoto_coefficient = DataStructs.TanimotoSimilarity(test_bit, train_bit)
            new_row = pd.DataFrame({'tanimoto_coefficient': [tanimoto_coefficient]})
            train_simularities = pd.concat([train_simularities, new_row], ignore_index=True)

        max_tanimoto = train_simularities['tanimoto_coefficient'].max()
        new_row = pd.DataFrame({'smiles': [train_compound], 'tanimoto_coefficient': [max_tanimoto]})
        most_similar_to_in_train = pd.concat([most_similar_to_in_train, new_row], ignore_index=True)

    for MAX_TANIMOTO in tanimoto:
        # keep only rows of most_similar_to_in_train where tanimoto_coefficient is less than 0.5
        most_similar_to_in_train_tan = most_similar_to_in_train[
            most_similar_to_in_train['tanimoto_coefficient'] < MAX_TANIMOTO]
        train_set = most_similar_to_in_train_tan.merge(allsmiles, on='smiles')
        # get size for train set dataframe
        print(isoform, 'TAN:', MAX_TANIMOTO, 'Size train:', len(train_set))
        # put size of train_set into row 'train size and tanimoto column into tanimoto_df for plotting
        tanimoto_df.loc['train_set_size', MAX_TANIMOTO] = len(train_set)
        train_set.to_csv(pm['dir'] + '/Clusters_Max_TC/' + 'train_set_' + isoform + '_' + str(MAX_TANIMOTO) + '.csv',
                         index=False)
        tanimoto_df.to_csv(pm['dir'] + '/Clusters_Max_TC/' + 'tanimoto_df_' + isoform + '.csv', index=False)

    # Plot using matplot lib train_set_size over tanimoto
    plt.plot(tanimoto_df.columns, tanimoto_df.loc['train_set_size'])
    plt.xlabel('Tanimoto Coefficient')
    plt.ylabel('Train Set Size')
    plt.title('Train Set Size vs Tanimoto Coefficient')
    # add y-axis value to each point
    for i in tanimoto:
        plt.annotate(tanimoto_df.loc['train_set_size'][i], (i, tanimoto_df.loc['train_set_size'][i]))
    plt.savefig(pm['dir'] + '/Clusters_Max_TC/' + 'train_set_size_vs_tanimoto_' + isoform + '.png')
    plt.close()


