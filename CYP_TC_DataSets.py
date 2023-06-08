import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import matplotlib
import sklearn
import tensorflow as tf
from tensorflow import keras
import CYP_inhibition_functions
import make_models
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import roc_auc_score
keras.backend.clear_session()
import random

global id_in_test_train
# Import the required libraries
import statistics as stat
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from tqdm import tqdm
# Set seed
seed = 1
os.environ['PYTHONHASHEDSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


#Need to filter out the ones that werent docked. get list from nick.

def load_parameters():
    # Parameters from params.json and exp.json loaded here to override parameters set below.
    pm = {'Project_name': 'feb_17_Tan',
          'dir': '/home/benweiser/virtual/EnsembleLearning/CSVFiles/AllCSVs',
        'data_dir': '/Results_CSV/',
        'fig_dir': 'Figures',
          'model_dir': 'Models',
          'CYPs': ['3A4'], #'2D6', '2C9'],  # ['1A2', '2C9'], # ,        'CYPs': ['2D6', '3A4', '2C9'],  # ['1A2', '2C9'], # ,
          'num_test_set_clusters': 100,
        'test_set_cluster_size': 10,
          'use_Been_Docked': 1,
        'use_Been_Docked_all': 0,
        'use_some_data': 0, 'clusters_100': 0
        } #1 to calc tanimoto analysis
    return pm

tanimoto = [0.1, 0.15, 0.2, 0.25 , 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
#make a dataframe with columns for each tanimoto value
#make a dataframe with columns for each tanimoto value




pm = load_parameters()
DATA_DIR = pm['data_dir']
dir_path = pm['dir'] + DATA_DIR
#make directory for tanimoto smiles csv in pm['dir']
if not os.path.exists(pm['dir'] + '/Clusters_Max_TC'):
    os.makedirs(pm['dir'] + '/Clusters_Max_TC')
for isoform in pm['CYPs']:
    print('Isoform: ', isoform)
    tanimoto_df = pd.DataFrame(columns=tanimoto)
    docked = []

    # loop over files in directory and subdirectories



    i_to_smiles = pd.read_csv('/home/benweiser/virtual/EnsembleLearning/CSVFiles/TanimotoCSV_Final' + '/Smiles/' + isoform + '-inactive.smi',names = ['id', 'smiles'], header=None, sep='\s+')
    #i_to_smiles_ids = list(i_to_smiles.iloc[:, 0])
    a_to_smiles = pd.read_csv('/home/benweiser/virtual/EnsembleLearning/CSVFiles/TanimotoCSV_Final' + '/Smiles/' + isoform + '-active.smi',names = ['id', 'smiles'], header=None, sep='\s+')
    #a_to_smiles_ids = list(a_to_smiles.iloc[:, 0])
    print('Loaded smiles')


    #Been_docked to compare with docked files and only use docked molecule in clusters
    if pm['use_Been_Docked'] == 1:
        activeData = pd.read_csv(
            '/home/benweiser/virtual/EnsembleLearning/CSVFiles/TanimotoCSV_Final/Results_CSV/' + isoform + '/' + isoform + '-docked-actives-scored' + '.csv')
        activeData['Activity'] = 1.0

        inactiveData = pd.read_csv(
            '/home/benweiser/virtual/EnsembleLearning/CSVFiles/TanimotoCSV_Final/Results_CSV/' + isoform + '/' + isoform + '-docked-inactives-scored' + '.csv')
        inactiveData['Activity'] = 0.0
        docked_combined = pd.concat([activeData, inactiveData], ignore_index=True)
        docked_combined['id'] = docked_combined['Molecule Name'].str.extract('(\d+)', expand=False).astype(int)



        allsmiles = pd.concat([i_to_smiles, a_to_smiles])
        # print size of all smiles
        print('Size of all smiles: ', len(allsmiles))
        #make it so that it only uses the 'id; that have been docked
        allsmiles['been_docked'] = allsmiles['id'].isin(docked_combined['id'])
        allsmiles = allsmiles[allsmiles['been_docked'] == True]
        # drop been_docked column
        allsmiles = allsmiles.drop(columns=['been_docked'])
        #print size of all smiles
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
                            docked.append(df)


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
        #drop been_docked column
        allsmiles = allsmiles.drop(columns=['been_docked'])
        #drop any in docked_combined that are not in allsmiles

        # pick an id name from Docked_combined and call it test set origin.
        # get all uniques docked ids from docked_combined
        print('Size of docked_combined before:', len(docked_combined))
        docked_combined = docked_combined[docked_combined['id'].isin(allsmiles['id'])]
        print('Size of docked_combined after:', len(docked_combined))


    if pm['use_some_data'] == 1:
        allsmiles = allsmiles.head(500)


    if pm['clusters_100'] == 1: # to use clusters of 100 to get molecules filtered from REDUCE in FITTED if not using FITTED csv data


        #read cluster_list_1a2-actives_100.txt
        a_cluster_list = pd.read_csv('/home/benweiser/virtual/EnsembleLearning/CSVFiles/TanimotoCSV_Final/Clusters_100/cluster_list_' + isoform.lower() + '-actives_100.txt', header=None, names = ['id'])
        #read cluster_list_1a2-inactives_100.txt
        i_cluster_list = pd.read_csv('/home/benweiser/virtual/EnsembleLearning/CSVFiles/TanimotoCSV_Final/Clusters_100/cluster_list_' + isoform.lower() + '-inactives_100.txt', header=None, names = ['id'])
        #concatenate cluster_list_1a2-actives_100.txt and cluster_list_1a2-inactives_100.txt
        cluster_list = pd.concat([a_cluster_list, i_cluster_list])
        print('Size of allsmiles before 100:', len(allsmiles))
        #filter out allsmiles that are not in cluster_list
        allsmiles = allsmiles[allsmiles['id'].isin(cluster_list['id'])]
        print('Size of allsmiles after 100:', len(allsmiles))


    #Make Test Set
    ##############################################################################################################
    print('Making Test Set')
    test_set = pd.DataFrame(columns=['smiles','tanimoto_coefficient' , 'id'])
    for i in tqdm(range(pm['num_test_set_clusters']), total=pm['num_test_set_clusters'], desc="Making Test Set"):
        origin_seed = random.randint(0, len(allsmiles))-1
        test_set_origin = allsmiles['smiles'].iloc[origin_seed]  # pick a random id from docked_ids to be the test set origin
        ###test_set_orgin_smiles = allsmiles[allsmiles['id'] == test_set_origin]['smiles']  # get the smiles of the test set origin

        # Get the tanimoto simularity between the test set origin and all uniques docked smiles

        # keep these 500 and call them the test ids
        origin_mol = AllChem.MolFromSmiles(test_set_origin)
        origin_bit = AllChem.GetMorganFingerprintAsBitVect(origin_mol, radius=2, nBits=2048)
        origin_tanimoto_coefficient = pd.DataFrame(columns=['smiles', 'tanimoto_coefficient'])
        for test_compound in allsmiles['smiles']:
            test_mol = Chem.MolFromSmiles(test_compound)
            test_bit = AllChem.GetMorganFingerprintAsBitVect(test_mol, radius=2, nBits=2048)
            tanimoto_coefficient = DataStructs.TanimotoSimilarity(origin_bit,test_bit)
            #add tano to origin_tanimoto_coefficient and test_compound
            origin_tanimoto_coefficient = origin_tanimoto_coefficient.append({'smiles': test_compound, 'tanimoto_coefficient': tanimoto_coefficient}, ignore_index=True)

        #add id column of allsmiles to origin_tanimoto_coefficient and merge on smiles
        test_set_cluster = origin_tanimoto_coefficient.merge(allsmiles, on='smiles')
        #sort origin_tanimoto_coefficient by tanimoto_coefficient and take top 500
        test_set_cluster = test_set_cluster.sort_values(by=['tanimoto_coefficient'], ascending=False)
        test_set_cluster = test_set_cluster.head(pm['test_set_cluster_size'])
        test_set = test_set.append(test_set_cluster)

    # get size for test set dataframe
    print('Size of test set:', len(test_set))
    test_set.to_csv(pm['dir']+'/Clusters_Max_TC/' + 'test_set_' + isoform  + '.csv', index=False)

    #Make Train Set
    ##############################################################################################################

    print('Making Train Set')
    #drop test set from Allsmiles
    train_smiles = copy.deepcopy(allsmiles)
    #drop test set from train_smiles
    train_smiles = train_smiles[~train_smiles['smiles'].isin(test_set['smiles'])]

    most_similar_to_in_train = pd.DataFrame(columns=['smiles','tanimoto_coefficient'])
    for train_compound in tqdm(train_smiles['smiles'], total=len(train_smiles), desc="Making Train Set"):
        train_mol = Chem.MolFromSmiles(train_compound)
        train_bit = AllChem.GetMorganFingerprintAsBitVect(train_mol, radius=2, nBits=2048)
        train_simularities = pd.DataFrame(columns=['tanimoto_coefficient'])
        for test_compound in test_set['smiles']:
            # Convert the train compound to a mol object
            test_mol = Chem.MolFromSmiles(test_compound)
            test_bit = AllChem.GetMorganFingerprintAsBitVect(test_mol, radius=2, nBits=2048)
            tanimoto_coefficient = DataStructs.TanimotoSimilarity(test_bit, train_bit)
            train_simularities = train_simularities.append({'tanimoto_coefficient': tanimoto_coefficient}, ignore_index=True)
        most_similar_to_in_train = most_similar_to_in_train.append({'smiles': train_compound, 'tanimoto_coefficient': train_simularities['tanimoto_coefficient'].max()}, ignore_index=True)

    for MAX_TANIMOTO in tanimoto:
        #keep only rows of most_similar_to_in_train where tanimoto_coefficient is less than 0.5
        most_similar_to_in_train_tan = most_similar_to_in_train[most_similar_to_in_train['tanimoto_coefficient'] < MAX_TANIMOTO]
        train_set = most_similar_to_in_train_tan.merge(allsmiles, on='smiles')
        #get size for train set dataframe
        print(isoform, 'TAN:', MAX_TANIMOTO, 'Size train:', len(train_set))
        #put size of train_set into row 'train size and tanimoto column into tanimoto_df for plotting
        tanimoto_df.loc['train_set_size', MAX_TANIMOTO] = len(train_set)
        train_set.to_csv(pm['dir']+'/Clusters_Max_TC/'+'train_set_' + isoform + '_' + str(MAX_TANIMOTO) + '.csv', index=False)
        tanimoto_df.to_csv(pm['dir'] + '/Clusters_Max_TC/' + 'tanimoto_df_' + isoform + '.csv', index=False)

    #Plot using matplot lib train_set_size over tanimoto
    plt.plot(tanimoto_df.columns, tanimoto_df.loc['train_set_size'])
    plt.xlabel('Tanimoto Coefficient')
    plt.ylabel('Train Set Size')
    plt.title('Train Set Size vs Tanimoto Coefficient')
    #add y axis value to each point
    for i in tanimoto:
        plt.annotate(tanimoto_df.loc['train_set_size'][i], (i,tanimoto_df.loc['train_set_size'][i]))
    plt.savefig(pm['dir'] + '/Clusters_Max_TC/' + 'train_set_size_vs_tanimoto_' + isoform + '.png')
    plt.close()




    #now do this over all tanimoto. see the size of tanimoto 10 or 20
















'''
id_names = [int(x.split('_')[0]) for x in docked_combined['Molecule Name']]
#Find fraction of id names in a_id_names are in a_to_smiles_ids
a_to_smiles_ids = np.array(a_to_smiles_ids)
i_to_smiles_ids = np.array(i_to_smiles_ids)
id_names = np.array(docked_combined)

a_ids = np.isin(a_to_smiles_ids, id_names)
i_ids = np.isin(i_to_smiles_ids, id_names)

#find fraction of True in a_ids over size of a_id_names
a_frac = np.sum(a_ids) / len(a_to_smiles_ids)
i_frac = np.sum(i_ids) / len(i_to_smiles_ids)
print('A frac:', a_frac, 'I frac:', i_frac)'''

