# ML-augmented-docking-CYP-inhibition
Code repository for Machine Learning-Augmented Docking. 1. CYP inhibition prediction. Authors: Benjamin Weiser, Jérôme Genzling , Mihai Burai Patrascu,  Ophélie Rostain and Nicolas Moitessier

# Code Workflow Procedure
1. Cleaned and combined test and train Pei sets with CYP\_clean\_files.ipnyb. Sets combined and then clustered to create new train and test sets
2. Dock each ligand 5 times to its respective isoform using FITTED. Docked data can be found here: (to be inserted)
3. Create analogue sets using FITTED. Create max train-test similarity using CYP\_TC\_DataSets.py
4. Run RF with Feature Importances using max train-test similarity of 0.8 using ML\_over\_Tanimoto.py which calls CYP\_inhibition\_functions.py and Do\_ML2.py 
5. Using these selected features run all ML models on all datasets using ML\_over\_Tanimoto.py which calls CYP\_inhibition\_functions.py and Do\_ML2.py 
6. Use CYP_evaluate_and_ensemble.py which calls CYP_evaluate_and_ensemble_functions.py to make ensembles and evaluate and graph model performance

All code and folders focused in this project were in a folder called EnsembleLearning which you will have to change to your directory
Can run code, CYP_evaluate_and_ensemble, to load and evaluate models from ../CSVFiles/TanimotoCSV_Final

# Train Models
Trained model can be found in CSVFiles/TanimotoCSV_Final/Models
MaxTC is model made using different max train-test Tanimoto shown in figure 5, 7, 8
MaxTCsamed is model made using different max train-test Tanimoto with same data size shown in figure 6
analog is model trained using the anolog datasets shown in figure 9

Results_CSV contains all docked data used To implement into the workflow for replication unzip into /EnsembleLearning/CSVFiles/TanimotoCSV_Final
EnsembleLearning contains code and files used for replication
Figure_Hyperparameters contains figure of all hyperparameter searchs
Model_Statistics contains all results of the evaluated models and Feature Importances
Models contains all model made for cyp inhibition for this work. To implement into the workflow for replication unzip into /EnsembleLearning/CSVFiles/TanimotoCSV_Final
