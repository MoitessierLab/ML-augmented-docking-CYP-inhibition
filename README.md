# ML-augmented-docking-CYP-inhibition
Code repository for Machine Learning-Augmented Docking. 1. CYP inhibition prediction. Authors: Benjamin Weiser, Jérôme Genzling , Mihai Burai Patrascu,  Ophélie Rostain and Nicolas Moitessier

\section*{Code Workflow Procedure (To be Finalized!!!}
\begin{enumerate}
    \item Cleaned and combined test and train Pei sets with CYP\_clean\_files.ipnyb. Sets combined and then clustered to create new train and test sets
    \item Dock each ligand 5 times to its respective isoform using FITTED. Docked data can be found here: (to be inserted)
    \item Create analogue sets using FITTED. Create max train-test similarity using CYP\_TC\_DataSets.py
    \item Run RF with Feature Importances using max train-test similarity of 0.8 using ML\_over\_Tanimoto.py which calls CYP\_inhibition\_functions.py and Do\_ML2.py 
    \item Using these selected features run all ML models on all datasets using ML\_over\_Tanimoto.py which calls CYP\_inhibition\_functions.py and Do\_ML2.py 
\end{enumerate}
