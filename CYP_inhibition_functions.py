import pandas
import pandas as pd


def drop_features(wholeSet, name):
    pm = {'drop_all': ['Molecule Name', 'Hydrogen Bond-like Metal interaction', 'Metal Coordination',
                       'Number of Water Molecules', 'RMSD', 'RMSD_(MBG)', 'Strain Energy', 'Michael_Acceptors',
                       'carboxylic_acid', 'carboxylate', 'ester_aromatic', 'ester_aliphatic', 'ester_conjugated',
                       'ester', 'lactone', 'aldehyde_aromatic', 'aldehyde_aliphatic', 'aldehyde_conjugated', 'aldehyde',
                       'acyl_chloride', 'acyl_bromide', 'amide_ter', 'amide', 'lactame', 'anhydride', 'ketone',
                       'sulfonamide', 'thiol', 'thiolate', 'carbamate', 'primary_amine', 'secondary_amine',
                       'tertiary_amine', 'quat_ammonium', 'amine', 'aniline', 'nitrile', 'imine', 'oxime', 'nitroso',
                       'nitro', 'azide', 'isocyanate', 'hydroxamic_acid', 'hydroxamate', 'alkyl_bromide',
                       'alkyl_chloride', 'alkyl_iodide', 'aryl_chloride', 'aryl_bromide', 'aryl_iodide',
                       'boronic_acid_aromatic', 'boronic_ester_aromatic', 'boronic_acid_aliphatic',
                       'boronic_ester_aliphatic', 'boronic_acid_vinylic', 'boronic_ester_vinylic', 'boronic_acid',
                       'sulfonyl_chloride', 'sulfonyl_bromide', 'vinyl_chloride', 'vinyl_bromide', 'vinyl_iodide',
                       'alkene', 'ketone_O', 'formyl', 'chloride', 'alcohol', 'bromide', 'fluoride', 'iodide',
                       'silicon', 'primary_amine_neutral', 'secondary_amine_neutral', 'tertiary_amine_neutral',
                       'alpha_fluoroketone', 'alpha_chloroketone', 'epoxide', 'aziridine', 'beta-lactam',
                       'primary_alcohol', 'disulfide', 'azo', 'isothiocyanate', 'hydrazine', 'ketene', 'peroxide',
                       'silyl_fluoride', 'silyl_bromide', 'silyl_chloride', 'silyl_iodide', 'thiocarbonyl', 'thioester',
                       'betaKeto', 'undefined', 'undefined.1', 'undefined.2', 'undefined.3', 'undefined.4',
                       'undefined.5', 'undefined.6', 'undefined.7', 'undefined.8', 'undefined.9', 'undefined.10',
                       'Blood-Brain_Barrier', 'logS', 'Badapple_pscore'],
          'drop_1a2_0': ['Sulfurs', 'Heavy_Atoms', 'Stereogenic_Centers', 'Halogens', 'ARG108-Side-Chain-HB',
                         'ILE117-Backbone-HB', 'ILE117-Side-Chain-Elec', 'ILE117-Side-Chain-HB', 'THR118-Backbone-HB',
                         'ASP119-Backbone-HB', 'ASP119-Side-Chain-HB', 'GLY120-Backbone-HB', 'GLY120-Side-Chain-vdW',
                         'GLY120-Side-Chain-Elec', 'GLY120-Side-Chain-HB', 'GLN121-Backbone-HB',
                         'GLN121-Side-Chain-vdW', 'GLN121-Side-Chain-Elec', 'GLN121-Side-Chain-HB',
                         'SER122-Backbone-HB', 'SER122-Side-Chain-HB', 'LEU123-Backbone-HB', 'LEU123-Side-Chain-Elec',
                         'LEU123-Side-Chain-HB', 'THR124-Backbone-HB', 'PHE125-Backbone-HB', 'PHE125-Side-Chain-Elec',
                         'PHE125-Side-Chain-HB', 'SER126-Backbone-vdW', 'SER126-Backbone-Elec', 'SER126-Backbone-HB',
                         'SER126-Side-Chain-vdW', 'SER126-Side-Chain-Elec', 'SER126-Side-Chain-HB',
                         'LEU219-Backbone-HB', 'LEU219-Side-Chain-Elec', 'LEU219-Side-Chain-HB', 'VAL220-Backbone-HB',
                         'VAL220-Side-Chain-Elec', 'VAL220-Side-Chain-HB', 'THR223-Side-Chain-HB',
                         'HISP224-Backbone-HB', 'HISP224-Side-Chain-vdW', 'HISP224-Side-Chain-Elec',
                         'HISP224-Side-Chain-HB', 'PHE226-Backbone-HB', 'PHE226-Side-Chain-Elec',
                         'PHE226-Side-Chain-HB', 'VAL227-Backbone-HB', 'VAL227-Side-Chain-Elec', 'VAL227-Side-Chain-HB',
                         'ALA230-Backbone-HB', 'ALA230-Side-Chain-Elec', 'ALA230-Side-Chain-HB', 'PHE253-Backbone-vdW',
                         'PHE253-Backbone-HB', 'PHE253-Side-Chain-Elec', 'PHE253-Side-Chain-HB', 'PHE256-Backbone-HB',
                         'PHE256-Side-Chain-Elec', 'PHE256-Side-Chain-HB', 'ASN257-Backbone-HB', 'PHE260-Backbone-HB',
                         'PHE260-Side-Chain-HB', 'PHE260-Side-Chain-Elec', 'LEU261-Backbone-HB',
                         'LEU261-Side-Chain-Elec', 'LEU261-Side-Chain-HB', 'LEU264-Backbone-vdW', 'LEU264-Backbone-HB',
                         'LEU264-Side-Chain-vdW', 'LEU264-Side-Chain-Elec', 'LEU264-Side-Chain-HB',
                         'ASN309-Backbone-HB', 'ASN309-Side-Chain-Elec', 'ASN309-Side-Chain-HB', 'LEU310-Backbone-HB',
                         'LEU310-Side-Chain-vdW', 'LEU310-Side-Chain-Elec', 'LEU310-Side-Chain-HB',
                         'VAL311-Backbone-HB', 'VAL311-Side-Chain-vdW', 'VAL311-Side-Chain-Elec',
                         'VAL311-Side-Chain-HB', 'ASN312-Side-Chain-HB', 'ILE314-Backbone-HB', 'ILE314-Side-Chain-vdW',
                         'ILE314-Side-Chain-Elec', 'ILE314-Side-Chain-HB', 'PHE315-Backbone-HB',
                         'PHE315-Side-Chain-Elec', 'PHE315-Side-Chain-HB', 'GLY316-Side-Chain-vdW',
                         'GLY316-Side-Chain-Elec', 'GLY316-Side-Chain-HB', 'ALA317-Side-Chain-HB', 'GLY318-Backbone-HB',
                         'GLY318-Side-Chain-vdW', 'GLY318-Side-Chain-Elec', 'GLY318-Side-Chain-HB',
                         'PHE319-Backbone-HB', 'PHE319-Side-Chain-Elec', 'PHE319-Side-Chain-HB', 'ASP320-Backbone-HB',
                         'THR321-Side-Chain-HB', 'THR324-Backbone-vdW', 'THR324-Backbone-Elec', 'THR324-Backbone-HB',
                         'THR324-Side-Chain-HB', 'PHE381-Backbone-vdW', 'PHE381-Backbone-HB', 'PHE381-Side-Chain-Elec',
                         'PHE381-Side-Chain-HB', 'LEU382-Backbone-HB', 'LEU382-Side-Chain-Elec', 'LEU382-Side-Chain-HB',
                         'PRO383-Backbone-vdW', 'PRO383-Backbone-Elec', 'PRO383-Backbone-HB', 'PRO383-Side-Chain-vdW',
                         'PRO383-Side-Chain-Elec', 'PRO383-Side-Chain-HB', 'PHE384-Backbone-HB',
                         'PHE384-Side-Chain-vdW', 'PHE384-Side-Chain-Elec', 'PHE384-Side-Chain-HB',
                         'THR385-Backbone-HB', 'THR385-Side-Chain-vdW', 'THR385-Side-Chain-Elec',
                         'THR385-Side-Chain-HB', 'ILE386-Backbone-HB', 'ILE386-Side-Chain-Elec', 'ILE386-Side-Chain-HB',
                         'GLN411-Backbone-vdW', 'GLN411-Backbone-Elec', 'GLN411-Backbone-HB', 'GLN411-Side-Chain-vdW',
                         'GLN411-Side-Chain-Elec', 'GLN411-Side-Chain-HB', 'LEU450-Backbone-vdW',
                         'LEU450-Backbone-Elec', 'LEU450-Backbone-HB', 'LEU450-Side-Chain-vdW',
                         'LEU450-Side-Chain-Elec', 'LEU450-Side-Chain-HB', 'PHE451-Backbone-HB',
                         'PHE451-Side-Chain-vdW', 'PHE451-Side-Chain-Elec', 'PHE451-Side-Chain-HB',
                         'CYS458-Backbone-HB', 'CYS458-Side-Chain-Elec', 'CYS458-Side-Chain-HB', 'ILE459-Backbone-vdW',
                         'ILE459-Backbone-Elec', 'ILE459-Backbone-HB', 'ILE459-Side-Chain-vdW',
                         'ILE459-Side-Chain-Elec', 'ILE459-Side-Chain-HB', 'LEU497-Backbone-HB',
                         'LEU497-Side-Chain-Elec', 'LEU497-Side-Chain-HB', 'THR498-Backbone-HB', 'THR498-Side-Chain-HB',
                         'LYS500-Backbone-vdW', 'LYS500-Backbone-Elec', 'LYS500-Backbone-HB', 'LYS500-Side-Chain-HB',
                         'HEM900-Backbone-vdW', 'HEM900-Backbone-Elec', 'HEM900-Backbone-HB'],
          'drop_1a2_1': ['Water Electrostatics', 'Water van der Waals', 'HEM900-Side-Chain-HB', 'THR498-Backbone-vdW',
                         'ARG108-Backbone-HB', 'ARG108-Backbone-Elec', 'Negatively_Charged_Oxygens',
                         'Positively_Charged_Nitrogen', 'ARG108-Backbone-vdW', 'ILE117-Backbone-vdW',
                         'THR223-Backbone-vdW', 'LEU382-Side-Chain-vdW', 'LEU264-Backbone-Elec', 'VAL227-Backbone-Elec',
                         'PHE451-Backbone-Elec', 'LEU497-Backbone-Elec'],
          'drop_2c9_0': ['PHE100-Side-Chain-HB', 'ALA106-Side-Chain-HB', 'ILE112-Side-Chain-HB', 'VAL113-Side-Chain-HB',
                         'PHE114-Side-Chain-HB', 'LEU197-Side-Chain-HB', 'LEU201-Side-Chain-HB', 'ILE205-Side-Chain-HB',
                         'ILE207-Side-Chain-HB', 'LEU208-Side-Chain-HB', 'LEU233-Side-Chain-HB', 'VAL237-Side-Chain-HB',
                         'ALA291-Side-Chain-HB', 'VAL292-Side-Chain-HB', 'LEU294-Side-Chain-HB', 'PHE295-Side-Chain-HB',
                         'GLY296-Side-Chain-vdW', 'GLY296-Side-Chain-Elec', 'GLY296-Side-Chain-HB',
                         'ALA297-Side-Chain-HB', 'GLY298-Side-Chain-vdW', 'GLY298-Side-Chain-Elec',
                         'GLY298-Side-Chain-HB', 'LEU361-Side-Chain-HB', 'LEU362-Side-Chain-HB', 'LEU366-Side-Chain-HB',
                         'VAL436-Side-Chain-HB', 'PHE476-Side-Chain-HB', 'ALA477-Side-Chain-HB', 'HEM500-Backbone-vdW',
                         'HEM500-Backbone-Elec', 'HEM500-Backbone-HB'],
          'drop_2c9_1': ['LEU366-Backbone-Elec', 'ALA297-Backbone-vdW', 'LEU233-Side-Chain-Elec',
                         'LYS200-Side-Chain-Elec', 'THR290-Side-Chain-vdW', 'THR290-Side-Chain-HB',
                         'Water van der Waals', 'Water Electrostatics', 'Water Hydrogen Bonds', 'CYS435-Backbone-HB',
                         'ARG97-Backbone-HB', 'LYS200-Side-Chain-HB', 'ARG97-Backbone-Elec', 'ARG97-Backbone-vdW',
                         'Stereogenic_Centers'],
          'drop_2c19_0': ['PHE100-Side-Chain-HB', 'LEU102-Side-Chain-HB', 'ALA103-Side-Chain-HB',
                          'ALA106-Side-Chain-HB', 'GLY111-Side-Chain-vdW', 'GLY111-Side-Chain-Elec',
                          'GLY111-Side-Chain-HB', 'ILE112-Side-Chain-HB', 'VAL113-Side-Chain-HB',
                          'PHE114-Side-Chain-HB', 'LEU201-Side-Chain-HB', 'ILE205-Side-Chain-HB',
                          'VAL208-Side-Chain-HB', 'ILE213-Side-Chain-HB', 'LEU233-Side-Chain-HB',
                          'LEU237-Side-Chain-HB', 'ILE289-Side-Chain-HB', 'ALA292-Side-Chain-HB',
                          'LEU294-Side-Chain-HB', 'LEU295-Side-Chain-HB', 'GLY296-Side-Chain-vdW',
                          'GLY296-Side-Chain-Elec', 'GLY296-Side-Chain-HB', 'ALA297-Side-Chain-HB',
                          'GLY298-Side-Chain-vdW', 'GLY298-Side-Chain-Elec', 'GLY298-Side-Chain-HB',
                          'LEU361-Side-Chain-HB', 'ILE362-Side-Chain-HB', 'PRO363-Side-Chain-HB',
                          'LEU366-Side-Chain-HB', 'PRO367-Side-Chain-HB', 'LEU391-Side-Chain-HB',
                          'PRO427-Side-Chain-HB', 'PHE428-Side-Chain-HB', 'PHE476-Side-Chain-HB',
                          'ALA477-Side-Chain-HB', 'VAL479-Side-Chain-HB', 'HEM501-Backbone-vdW', 'HEM501-Backbone-Elec',
                          'HEM501-Backbone-HB'],
          'drop_2d6_0': ['LEU102-Side-Chain-HB', 'ALA103-Side-Chain-HB', 'ALA106-Side-Chain-HB',
                         'GLY111-Side-Chain-vdW', 'GLY111-Side-Chain-Elec', 'GLY111-Side-Chain-HB',
                         'ILE112-Side-Chain-HB', 'VAL113-Side-Chain-HB', 'PHE114-Side-Chain-HB', 'LEU201-Side-Chain-HB',
                         'ILE205-Side-Chain-HB', 'VAL208-Side-Chain-HB', 'ILE213-Side-Chain-HB', 'LEU233-Side-Chain-HB',
                         'LEU237-Side-Chain-HB', 'ILE289-Side-Chain-HB', 'ALA292-Side-Chain-HB', 'LEU294-Side-Chain-HB',
                         'LEU295-Side-Chain-HB', 'GLY296-Side-Chain-vdW', 'GLY296-Side-Chain-Elec',
                         'GLY296-Side-Chain-HB', 'ALA297-Side-Chain-HB', 'GLY298-Side-Chain-vdW',
                         'GLY298-Side-Chain-Elec', 'GLY298-Side-Chain-HB', 'LEU361-Side-Chain-HB',
                         'ILE362-Side-Chain-HB', 'PRO363-Side-Chain-HB', 'THR364-Side-Chain-HB', 'LEU366-Side-Chain-HB',
                         'PRO367-Side-Chain-HB', 'LEU391-Side-Chain-HB', 'PRO427-Side-Chain-HB', 'PHE428-Side-Chain-HB',
                         'PHE476-Side-Chain-HB', 'ALA477-Side-Chain-HB', 'VAL479-Side-Chain-HB,', 'HEM501-Backbone-vdW',
                         'HEM501-Backbone-Elec', 'HEM501-Backbone-HB'],
          'drop_3a4_0': ['ILE50-Side-Chain-HB', 'PHE57-Side-Chain-HB', 'PHE74-Side-Chain-HB', 'VAL81-Side-Chain-HB',
                         'PRO107-Side-Chain-HB', 'PHE108-Side-Chain-HB', 'GLY109-Side-Chain-vdW',
                         'GLY109-Side-Chain-Elec', 'GLY109-Side-Chain-HB', 'VAL111-Side-Chain-HB',
                         'ALA117-Side-Chain-HB', 'ILE118-Side-Chain-HB', 'ILE120-Side-Chain-HB', 'ALA121-Side-Chain-HB',
                         'LEU210-Side-Chain-HB', 'LEU211-Side-Chain-HB', 'PHE213-Side-Chain-HB', 'PHE215-Side-Chain-HB',
                         'PHE220-Side-Chain-HB', 'LEU221-Side-Chain-HB', 'ILE223-Side-Chain-HB', 'THR224-Side-Chain-HB',
                         'VAL225-Side-Chain-HB', 'PRO227-Side-Chain-HB', 'VAL240-Side-Chain-HB', 'PHE241-Side-Chain-HB',
                         'PRO242-Side-Chain-HB', 'VAL245-Side-Chain-HB', 'THR246-Side-Chain-HB', 'LEU249-Side-Chain-HB',
                         'ALA297-Side-Chain-HB', 'ILE300-Side-Chain-HB', 'ILE301-Side-Chain-HB', 'PHE302-Side-Chain-HB',
                         'ILE303-Side-Chain-HB', 'PHE304-Side-Chain-HB', 'ALA305-Side-Chain-HB',
                         'GLY306-Side-Chain-vdW', 'GLY306-Side-Chain-Elec', 'GLY306-Side-Chain-HB',
                         'VAL313-Side-Chain-HB', 'PRO368-Side-Chain-HB', 'ILE369-Side-Chain-HB', 'ALA370-Side-Chain-HB',
                         'LEU373-Side-Chain-HB', 'VAL393-Side-Chain-HB', 'VAL394-Side-Chain-HB', 'ILE396-Side-Chain-HB',
                         'PRO397-Side-Chain-HB', 'PRO434-Side-Chain-HB', 'PHE435-Side-Chain-HB',
                         'GLY436-Side-Chain-vdW', 'GLY436-Side-Chain-Elec', 'GLY436-Side-Chain-HB',
                         'ILE443-Side-Chain-HB', 'GLY444-Side-Chain-vdW', 'GLY444-Side-Chain-Elec',
                         'GLY444-Side-Chain-HB', 'ALA448-Side-Chain-HB', 'LEU479-Side-Chain-HB',
                         'GLY480-Side-Chain-vdW', 'GLY480-Side-Chain-Elec', 'GLY480-Side-Chain-HB',
                         'GLY481-Side-Chain-vdW', 'GLY481-Side-Chain-Elec', 'GLY481-Side-Chain-HB',
                         'LEU482-Side-Chain-HB', 'LEU483-Side-Chain-HB', 'HEM508-Backbone-vdW', 'HEM508-Backbone-Elec',
                         'HEM508-Backbone-HB']

          }
    for drop in pm['drop_all']:
        wholeSet = wholeSet.drop([drop], axis=1)

    if name == "1A2":
        for drop in pm['drop_1a2_0']:
            wholeSet = wholeSet.drop([drop], axis=1)
        for drop in pm['drop_1a2_1']:
            wholeSet = wholeSet.drop([drop], axis=1)
    if name == "2C9":
        print('Protein: 2C9')
        for drop in pm['drop_2c9_0']:
            wholeSet = wholeSet.drop([drop], axis=1)
        for drop in pm['drop_2c9_1']:
            wholeSet = wholeSet.drop([drop], axis=1)
    if name == "2C19":
        print('Protein: 2C19')
        for drop in pm['drop_2c19_0']:
            wholeSet = wholeSet.drop([drop], axis=1)

    if name == "2D6":
        print('Protein: 2D6')
        #for drop in pm['drop_2d6_0']:  #
        #    wholeSet = wholeSet.drop([drop], axis=1)

    if name == "3A4":
        print('Protein: 3A4')
        for drop in pm['drop_3a4_0']:
            wholeSet = wholeSet.drop([drop], axis=1)
    return wholeSet


'''def select_features(data, name):
    # dropped features selected using random forest
    select_for_all = ['Activity', 'RankScore', 'MatchScore']
    data0 = data[select_for_all]
    # select features selected using random forest
    select_features_RF = {
        '1A2': ['Fsp3', 'Molecular_Hardness', 'Aromatic_Proportion', 'Aromatic_Rings', 'Molecular_Polarizability', 'Stereogenic_Centers_Ratio', 'GLY318-Backbone-vdW', 'ALA230-Side-Chain-vdW', 'ILE314-Backbone-vdW', 'Polar Ligand Solvation Energy', 'ALA230-Backbone-vdW', 'Polar Protein Solvation Energy', 'aromatic', 'Molecular_Softness', 'Oxygens', 'ALA317-Side-Chain-Elec', 'Molecular_van_der_Waals_Surface_Area', 'Non-polar Solvation Energy', 'Ovality_Index', 'logP', 'McGowan_Molecular_Volume', 'Globularity_Factor', 'MatchScore', '3D-Wiener_Index', 'Net_Charge', 'van der Waals', 'PHE226-Side-Chain-vdW', 'PHE315-Backbone-vdW', 'Ionizable_Centers', 'tPSA', 'RankScore', 'Heteroatoms', 'Molecular_Weight', 'LEU497-Side-Chain-vdW', 'THR321-Backbone-vdW', 'GLY316-Backbone-vdW', 'Molecular_Density', 'Molecular_Electronegativity', 'Energy', 'ASP320-Backbone-vdW', 'ALA317-Backbone-vdW', 'ALA317-Side-Chain-vdW', 'ASP320-Side-Chain-Elec', 'Polar_SASA', 'non_Polar_SASA', 'PHE319-Backbone-vdW', 'Molecular_Polarity', 'GLY316-Backbone-Elec', 'THR324-Side-Chain-vdW'],
        '2C9': ['Molecular_Polarizability', 'Fsp3', 'Molecular_Hardness', 'logP', 'Molecular_Weight', 'aromatic', 'Aromatic_Rings', 'Polar Ligand Solvation Energy', 'McGowan_Molecular_Volume', 'Ionizable_Centers', 'Polar Protein Solvation Energy', 'Stereogenic_Centers_Ratio', 'Aromatic_Proportion', 'Ovality_Index', 'Globularity_Factor', 'Heavy_Atoms', 'Molecular_van_der_Waals_Surface_Area', 'Molecular_Density', 'PHE100-Backbone-vdW', 'Geometric_Radius', 'ALA297-Side-Chain-Elec', 'Net_Charge', 'VAL113-Side-Chain-Elec', 'LEU208-Side-Chain-Elec', 'ILE207-Side-Chain-vdW', 'LEU208-Backbone-vdW', 'Geometric_Diameter', 'Molecular_Electronegativity', 'Molecular_Polarity', 'tPSA', 'MatchScore', 'van der Waals', 'Polar_SASA', 'ILE205-Side-Chain-Elec', 'ILE207-Backbone-vdW', 'Molecular_Softness', 'Nitrogens', '3D-Wiener_Index', 'Dipole moment', 'Energy', 'ARG108-Side-Chain-Elec', 'LEU366-Side-Chain-Elec', 'PHE100-Side-Chain-vdW', 'VAL292-Side-Chain-Elec', 'SER209-Backbone-vdW', 'Oxygens', 'non_Polar_SASA', 'Non-polar Solvation Energy', 'ASP293-Side-Chain-Elec'],
        '2C19': ['Molecular_Polarizability', 'Fsp3', 'Molecular_Hardness', 'aromatic', 'Net_Charge', 'logP', 'Aromatic_Rings', 'Polar Protein Solvation Energy', 'Aromatic_Proportion', 'VAL113-Side-Chain-Elec', 'ALA297-Side-Chain-Elec', 'Polar Ligand Solvation Energy', 'Stereogenic_Centers_Ratio', 'Molecular_Weight', 'McGowan_Molecular_Volume', 'Ovality_Index', 'Geometric_Radius', 'Molecular_van_der_Waals_Surface_Area', 'Molecular_Electronegativity', 'Globularity_Factor', 'tPSA', 'Molecular_Density', 'LEU366-Side-Chain-Elec', 'ASP293-Side-Chain-Elec', 'MatchScore', 'non_Polar_SASA', '3D-Wiener_Index', 'Molecular_Polarity', 'Geometric_Diameter', 'ILE205-Side-Chain-Elec', 'VAL208-Side-Chain-Elec', 'Heavy_Atoms', 'Energy', 'Oxygens', 'HEM501-Side-Chain-Elec', 'ILE362-Side-Chain-Elec', 'Halogens', 'Polar_SASA', 'THR301-Backbone-Elec', 'van der Waals', 'Dipole moment', 'Molecular_Softness', 'Rotatable Bonds Entropy', 'Hydrogen_Bond_Donors', 'LEU237-Side-Chain-Elec', 'CYS435-Side-Chain-Elec', 'Ionizable_Centers', 'ALA292-Side-Chain-Elec', 'GLY298-Backbone-HB'],
        '2D6': ['Molecular_Hardness', 'Polar Protein Solvation Energy', 'Polar Ligand Solvation Energy', 'logP', 'Aromatic_Proportion', 'Molecular_Polarizability', 'Quantitative_Estimate_of_Druglikeness', 'ALA305-Side-Chain-Elec', 'Net_Charge', 'SER304-Backbone-Elec', 'Fsp3', 'tPSA', 'GLU216-Side-Chain-Elec', 'Oxygens', 'PHE120-Side-Chain-Elec', 'Hydrogen_Bond_Acceptors', 'VAL308-Side-Chain-Elec', 'ASP301-Side-Chain-HB', 'GLU216-Side-Chain-HB', 'Polar_SASA', 'Molecular_Electronegativity', 'ALA300-Side-Chain-Elec', 'Aromatic_Rings', 'ASP301-Side-Chain-Elec', 'McGowan_Molecular_Volume', 'Energy', 'aromatic', 'Heteroatoms', 'non_Polar_SASA', 'Molecular_Polarity', 'Molecular_Weight', 'Molecular_van_der_Waals_Surface_Area', 'Globularity_Factor', 'Dipole moment', 'Molecular_Density', 'Ovality_Index', '3D-Wiener_Index', 'LEU121-Side-Chain-Elec', 'Molecular_Softness', 'LEU248-Side-Chain-Elec', 'CYS443-Side-Chain-Elec', 'Rotatable Bonds Entropy', 'Ionizable_Centers', 'MatchScore', 'Stereogenic_Centers_Ratio', 'PHE247-Side-Chain-Elec', 'LEU213-Side-Chain-Elec', 'ALA209-Backbone-HB', 'Nitrogens'],
        '3A4': ['Aromatic_Proportion', 'McGowan_Molecular_Volume', 'Water Hydrogen Bonds', 'Hydrogen_Bond_Acceptors',
                'Heteroatoms', 'Number of Rotatable Bonds', 'Ionizable_Centers', 'Rings', 'Oxygens', 'Molecular_Weight',
                'Rotatable_Bonds', 'Hydrogen_Bond_Donors', 'Non-polar Solvation Energy', 'Heavy_Atoms', 'tPSA',
                'Water van der Waals', 'van der Waals', 'Electrostatics', 'Energy', 'Polar Protein Solvation Energy',
                'Water Electrostatics', 'Sulfurs', 'Hydrogen Bonds', 'Rotatable Bonds Entropy', 'Net_Charge',
                'Polar Ligand Solvation Energy', 'Nitrogens', 'logP']}
    data1 = data[select_features_RF[name]]

    # selecting active site residues and import interacting residues identified
    # select_features_residues = {'1A2' : [], '2C9' : [], '2C19' : [], '2D6' : [], '3A4' : []}
    select_features_residues = {
        '1A2': ['ILE117', 'THR118', 'SER122', 'THR124', 'PHE125', 'THR223', 'PHE226', 'VAL227', 'PHE256', 'ASN257',
                'PHE260', 'ASN312', 'ASP313', 'GLY316', 'ALA317', 'GLY318', 'PHE319', 'ASP320', 'THR321', 'LEU382',
                'ILE386', 'LEU497', 'THR498', 'HEM'],
        '2C9': ['ARG97', 'ARG108', 'VAL113', 'PHE114', 'LEU201', 'ASN204', 'ILE205', 'LEU208', 'LEU233', 'VAL237',
                'MET240', 'VAL292', 'ASP293', 'LEU294', 'PHE295', 'GLY296', 'ALA297', 'GLY298', 'THR301', 'LEU362',
                'SER365', 'LEU366', 'PHE476', 'HEM'],
        '2C19': ['PHE100', 'LEU102', 'ALA103', 'ASN107', 'VAL113', 'PHE114', 'ILE205', 'VAL208', 'ALA292', 'ASP293',
                 'LEU294', 'GLY296', 'ALA297', 'GLY298', 'GLU300', 'THR301', 'THR304', 'LEU361', 'ILE362', 'SER365',
                 'LEU366', 'PHE476', 'ALA477', 'HEM'],
        '2D6': ['PHE112', 'PHE120', 'ALA209', 'GLY212', 'LEU213', 'LYS214', 'GLU216', 'GLN244', 'LEU248', 'ILE297',
                'ALA300', 'ASP301', 'SER304', 'ALA305', 'THR309', 'VAL374', 'PHE483', 'HEM'],
        '3A4': ['PHE57', 'ASP76', 'ARG105', 'ARG106', 'PHE108', 'SER119', 'PHE213', 'PHE215', 'ILE301', 'PHE304',
                'ALA305', 'GLY306', 'THR309', 'ILE369', 'ALA370', 'ARG372', 'LEU373', 'GLU374', 'CYS442', 'HEM']}

    data2 = data[data.columns[data.columns.str.startswith(tuple(select_features_residues[name]))]]
    # data2 = data.filter(like=select_features_residues[name])
    # print('data2:', data2)

    # data = pandas.concat([data0, data2], axis=1) #This is for interaction/docking features only!
    data = pandas.concat([data0, data1, data2], axis=1)  # This is all selected features after dropping
    # data = pandas.concat([data0], axis=1)  #Just Rank and Match score

    return data'''


def do_PCA(X_train, X_test):
    # sees what number of components need to get 95% varience in data and set as i for number of component in data
    from sklearn.decomposition import PCA
    import pandas as pd
    pca = PCA()
    pca.fit(X_train)
    pca_variance = pca.explained_variance_ratio_
    print("ecplained variance ration : ", pca_variance)
    sum = 0
    i = 0
    while sum < 0.95 and i < 500:  # get index of 95% variance
        sum = sum + pca_variance[i]
        i = i + 1
    pca = PCA(n_components=i)
    print('PCA components :', i)
    new_X_train = pca.fit_transform(X_train)
    new_X_test = pca.fit_transform(X_test)
    new_X_train = pd.DataFrame(new_X_train)
    new_X_test = pd.DataFrame(new_X_test)

    return new_X_train, new_X_test


def testresults(y_true, y_predicted):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import roc_auc_score
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted, labels=[0, 1]).ravel()
    # print("accuracy: ")
    # print(accuracy_score(y_true, y_predicted))
    # print("sensitivity")
    auc = roc_auc_score(y_true, y_predicted)
    sensitivity = tp / (tp + fn)
    # print("specicifity")
    # print(tn / (tn + fp))
    specicifity = tn / (tn + fp)
    # print("MCC is: ")
    MCC = matthews_corrcoef(y_true, y_predicted)
    r = {'sensitivity': sensitivity, 'specicifity': specicifity, 'AUC': auc, 'MCC': MCC}
    return r


def select_features_tanimoto(data, name, features):
    # if features = 0 then rank/match score
    # if feature = 1 then use all ligand and docked feature
    # if feature = 2 then uses ligand features
    # if featuer = 3 then uses docked features
    # dropped features selected using random forest
    select_for_all = ['RankScore', 'MatchScore']
    data00 = data[select_for_all]  # without activity so that it can be used on X_train/X_test
    # select features selected using random forest
    select_features_RF = {
        '1A2': ['Fsp3', 'Molecular_Hardness', 'Aromatic_Proportion', 'Aromatic_Rings', 'Molecular_Polarizability',
                'Stereogenic_Centers_Ratio', 'GLY318-Backbone-vdW', 'ALA230-Side-Chain-vdW', 'ILE314-Backbone-vdW',
                'Polar Ligand Solvation Energy', 'ALA230-Backbone-vdW', 'Polar Protein Solvation Energy', 'aromatic',
                'Molecular_Softness', 'Oxygens', 'ALA317-Side-Chain-Elec', 'Molecular_van_der_Waals_Surface_Area',
                'Non-polar Solvation Energy', 'Ovality_Index', 'logP', 'McGowan_Molecular_Volume', 'Globularity_Factor',
                'MatchScore', '3D-Wiener_Index', 'Net_Charge', 'van der Waals', 'PHE226-Side-Chain-vdW',
                'PHE315-Backbone-vdW', 'Ionizable_Centers', 'tPSA', 'RankScore', 'Heteroatoms', 'Molecular_Weight',
                'LEU497-Side-Chain-vdW', 'THR321-Backbone-vdW', 'GLY316-Backbone-vdW', 'Molecular_Density',
                'Molecular_Electronegativity', 'Energy', 'ASP320-Backbone-vdW', 'ALA317-Backbone-vdW',
                'ALA317-Side-Chain-vdW', 'ASP320-Side-Chain-Elec', 'Polar_SASA', 'non_Polar_SASA',
                'PHE319-Backbone-vdW', 'Molecular_Polarity', 'GLY316-Backbone-Elec', 'THR324-Side-Chain-vdW'],
        '2C9': ['Molecular_Polarizability', 'Fsp3', 'Molecular_Hardness', 'logP', 'Molecular_Weight', 'aromatic',
                'Aromatic_Rings', 'Polar Ligand Solvation Energy', 'McGowan_Molecular_Volume', 'Ionizable_Centers',
                'Polar Protein Solvation Energy', 'Stereogenic_Centers_Ratio', 'Aromatic_Proportion', 'Ovality_Index',
                'Globularity_Factor', 'Heavy_Atoms', 'Molecular_van_der_Waals_Surface_Area', 'Molecular_Density',
                'PHE100-Backbone-vdW', 'Geometric_Radius', 'ALA297-Side-Chain-Elec', 'Net_Charge',
                'VAL113-Side-Chain-Elec', 'LEU208-Side-Chain-Elec', 'ILE207-Side-Chain-vdW', 'LEU208-Backbone-vdW',
                'Geometric_Diameter', 'Molecular_Electronegativity', 'Molecular_Polarity', 'tPSA', 'MatchScore',
                'van der Waals', 'Polar_SASA', 'ILE205-Side-Chain-Elec', 'ILE207-Backbone-vdW', 'Molecular_Softness',
                'Nitrogens', '3D-Wiener_Index', 'Dipole moment', 'Energy', 'ARG108-Side-Chain-Elec',
                'LEU366-Side-Chain-Elec', 'PHE100-Side-Chain-vdW', 'VAL292-Side-Chain-Elec', 'SER209-Backbone-vdW',
                'Oxygens', 'non_Polar_SASA', 'Non-polar Solvation Energy', 'ASP293-Side-Chain-Elec'],
        '2C19': ['Molecular_Polarizability', 'Fsp3', 'Molecular_Hardness', 'aromatic', 'Net_Charge', 'logP',
                 'Aromatic_Rings', 'Polar Protein Solvation Energy', 'Aromatic_Proportion', 'VAL113-Side-Chain-Elec',
                 'ALA297-Side-Chain-Elec', 'Polar Ligand Solvation Energy', 'Stereogenic_Centers_Ratio',
                 'Molecular_Weight', 'McGowan_Molecular_Volume', 'Ovality_Index', 'Geometric_Radius',
                 'Molecular_van_der_Waals_Surface_Area', 'Molecular_Electronegativity', 'Globularity_Factor', 'tPSA',
                 'Molecular_Density', 'LEU366-Side-Chain-Elec', 'ASP293-Side-Chain-Elec', 'MatchScore',
                 'non_Polar_SASA', '3D-Wiener_Index', 'Molecular_Polarity', 'Geometric_Diameter',
                 'ILE205-Side-Chain-Elec', 'VAL208-Side-Chain-Elec', 'Heavy_Atoms', 'Energy', 'Oxygens',
                 'HEM501-Side-Chain-Elec', 'ILE362-Side-Chain-Elec', 'Halogens', 'Polar_SASA', 'THR301-Backbone-Elec',
                 'van der Waals', 'Dipole moment', 'Molecular_Softness', 'Rotatable Bonds Entropy',
                 'Hydrogen_Bond_Donors', 'LEU237-Side-Chain-Elec', 'CYS435-Side-Chain-Elec', 'Ionizable_Centers',
                 'ALA292-Side-Chain-Elec', 'GLY298-Backbone-HB'],
        '2D6': ['Molecular_Hardness', 'Polar Protein Solvation Energy', 'Polar Ligand Solvation Energy', 'logP',
                'Aromatic_Proportion', 'Molecular_Polarizability', 'Quantitative_Estimate_of_Druglikeness',
                'ALA305-Side-Chain-Elec', 'Net_Charge', 'SER304-Backbone-Elec', 'Fsp3', 'tPSA',
                'GLU216-Side-Chain-Elec', 'Oxygens', 'PHE120-Side-Chain-Elec', 'Hydrogen_Bond_Acceptors',
                'VAL308-Side-Chain-Elec', 'ASP301-Side-Chain-HB', 'GLU216-Side-Chain-HB', 'Polar_SASA',
                'Molecular_Electronegativity', 'ALA300-Side-Chain-Elec', 'Aromatic_Rings', 'ASP301-Side-Chain-Elec',
                'McGowan_Molecular_Volume', 'Energy', 'aromatic', 'Heteroatoms', 'non_Polar_SASA', 'Molecular_Polarity',
                'Molecular_Weight', 'Molecular_van_der_Waals_Surface_Area', 'Globularity_Factor', 'Dipole moment',
                'Molecular_Density', 'Ovality_Index', '3D-Wiener_Index', 'LEU121-Side-Chain-Elec', 'Molecular_Softness',
                'LEU248-Side-Chain-Elec', 'CYS443-Side-Chain-Elec', 'Rotatable Bonds Entropy', 'Ionizable_Centers',
                'MatchScore', 'Stereogenic_Centers_Ratio', 'PHE247-Side-Chain-Elec', 'LEU213-Side-Chain-Elec',
                'ALA209-Backbone-HB', 'Nitrogens'],
        '3A4': ['Heavy_Atoms', 'Globularity_Factor', 'Aromatic_Rings', 'Molecular_van_der_Waals_Surface_Area',
                'aromatic', 'McGowan_Molecular_Volume', 'Ovality_Index', 'Molecular_Polarizability', 'non_Polar_SASA',
                'Molecular_Weight', 'Rings', 'Fsp3', 'Molecular_Hardness', 'Molecular_Softness', '3D-Wiener_Index',
                'Aromatic_Proportion', 'Polar Ligand Solvation Energy', 'logP', 'Polar Protein Solvation Energy',
                'Net_Charge', 'van der Waals', 'Geometric_Radius', 'Geometric_Diameter', 'tPSA', 'Molecular_Polarity',
                'PHE108-Side-Chain-vdW', 'Non-polar Solvation Energy', 'PHE108-Backbone-vdW', 'Molecular_Electronegativity',
                'Nitrogens', 'Ionizable_Centers', 'Energy', 'Stereogenic_Centers_Ratio', 'Molecular_Density',
                'PHE213-Backbone-vdW', 'Water van der Waals', 'Oxygens', 'Hydrogen_Bond_Donors', 'LEU482-Side-Chain-vdW',
                'HEM508-Side-Chain-Elec', 'Polar_SASA', 'RankScore', 'ALA305-Side-Chain-Elec', 'ILE369-Side-Chain-Elec',
                'Quantitative_Estimate_of_Druglikeness', 'ALA370-Side-Chain-Elec', 'CYS442-Side-Chain-Elec',
                'VAL240-Side-Chain-vdW', 'Dipole moment']}


    # selecting active site residues and import interacting residues identified
    # select_features_residues = {'1A2' : [], '2C9' : [], '2C19' : [], '2D6' : [], '3A4' : []}
    select_features_residues = {
        '1A2': ['ILE117', 'THR118', 'SER122', 'THR124', 'PHE125', 'THR223', 'PHE226', 'VAL227', 'PHE256', 'ASN257',
                'PHE260', 'ASN312', 'ASP313', 'GLY316', 'ALA317', 'GLY318', 'PHE319', 'ASP320', 'THR321', 'LEU382',
                'ILE386', 'LEU497', 'THR498', 'HEM', 'ALA230', 'ILE314', 'ALA230', 'PHE315'], #rf agreed gly318, ala317, phe226, gly316, leu497, thr321, ala317, asp320 . rf found ala230!!!, ile314, 230ala, phe315
        '2C9': ['ARG97', 'ARG108', 'VAL113', 'PHE114', 'LEU201', 'ASN204', 'ILE205', 'LEU208', 'LEU233', 'VAL237',
                'MET240', 'VAL292', 'ASP293', 'LEU294', 'PHE295', 'GLY296', 'ALA297', 'GLY298', 'THR301', 'LEU362',
                'SER365', 'LEU366', 'PHE476', 'HEM', 'PHE100', 'LEU208', 'ILE205', 'ILE207', 'SER209'],  # rf agreed val113, ala297, leu366 found phe100, leu208, ile205, ile207, mc weight, aeromatic, plarity
        '2C19': ['PHE100', 'LEU102', 'ALA103', 'ASN107', 'VAL113', 'PHE114', 'ILE205', 'VAL208', 'ALA292', 'ASP293',
                 'LEU294', 'GLY296', 'ALA297', 'GLY298', 'GLU300', 'THR301', 'THR304', 'LEU361', 'ILE362', 'SER365',
                 'LEU366', 'PHE476', 'ALA477', 'HEM', 'ARG97', 'CYS435'],  #  rf agreed val113, ala 297, asp293, leu366, ile205, val208, ile362, gly298, aeromatic, found arg 97, cys435.
        '2D6': ['PHE112', 'PHE120', 'ALA209', 'GLY212', 'LEU213', 'LYS214', 'GLU216', 'GLN244', 'LEU248', 'ILE297',
                'ALA300', 'ASP301', 'SER304', 'ALA305', 'THR309', 'VAL374', 'PHE483', 'HEM', 'VAL308'], #rf agreed ala305, ser304, glu 216, phe120, glu216, asp301, aeromatic groups. Found val308
        '3A4': ['PHE57', 'ASP76', 'ARG105', 'ARG106', 'PHE108', 'SER119', 'PHE213', 'PHE215', 'ILE301', 'PHE304',
                'ALA305', 'GLY306', 'THR309', 'ILE369', 'ALA370', 'ARG372', 'LEU373', 'GLU374', 'CYS442', 'HEM', 'VAL240']} #rf agreed with all, leu482 found
    # if features = 0 then rank/match score
    # if feature = 1 then use all ligand and docked feature
    # if feature = 2 then uses ligand features
    # if featuer = 3 then uses docked features
    data1 = data[select_features_RF[name]] #ligand data
    # Select columns that do not start with the values in select_features_residues
    data1 = data1[data1.columns[~data1.columns.str.contains('|'.join(select_features_residues[name]))]]

    data2 = data[data.columns[data.columns.str.startswith(tuple(select_features_residues[name]))]] #docking data
    if features == 0:
        data = pandas.concat([data00], axis=1)

    if features == 1:
        data = pandas.concat([data00, data1, data2], axis=1)  # uses data0 as done early when data include activity

    if features == 2:
        data = pandas.concat([data00, data1], axis=1)

    if features == 3:
        data = pandas.concat([data00, data2], axis=1)

    # data2 = data.filter(like=select_features_residues[name])
    # print('data2:', data2)

    # data = pandas.concat([data0, data2], axis=1) #This is for interaction/docking features only!
    # data = pandas.concat([data0, data1, data2], axis=1)  #This is all selected features after dropping
    # data = pandas.concat([data0], axis=1)  #Just Rank and Match score

    #print header of data



    return data


def get_max_Tanimoto(train, test, id_to_smiles):
    ########

    # Import the required libraries
    import statistics as stat
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import DataStructs
    import pubchempy as pcp
    import time

    # Convert the smiles column to a list
    train = [x.split('_')[0] for x in train]
    test = [x.split('_')[0] for x in test]
    train = list(set(train))
    test = list(set(test))

    def get_smiles_from_pub(ids):
        smile_list_from_pcp = str(pcp.Compound.from_cid(ids).isomeric_smiles)
        smiles = smile_list_from_pcp.replace("\\\\", "\\")
        smiles = smiles.replace("\\\\", "\\")
        smiles = smiles.replace("\\\\", "\\")
        smiles = smiles.replace("\\\\", "\\")
        smiles = smiles.replace("\\\\", "\\")

        return smiles

    def convert_to_smiles(list_of_id, id_to_smiles_func):

        # currently
        smiles_list = []
        id_list = []
        i = 1
        for iddd in list_of_id:
            iddd = int(iddd)
            if iddd in id_to_smiles_func['cid'].values:
                index = id_to_smiles_func[id_to_smiles_func['cid'] == iddd].index.values
                if "InChI=" in str(id_to_smiles_func.loc[index, 'isosmiles']):
                    smiles_list.append(get_smiles_from_pub(iddd))
                elif 'C' not in str(id_to_smiles_func.loc[index, 'isosmiles']):
                    smiles_list.append(get_smiles_from_pub(iddd))
                else:
                    smiles_list.append(id_to_smiles_func.loc[index, 'isosmiles'].values)  # print('Here')
            else:

                smiles_list.append(get_smiles_from_pub(iddd))
                # print('From pcp: ', smile_list_from_pcp)
                if iddd % 4 == 0:
                    time.sleep(1)

        # print(smiles_list)
        # smiles_list = list(set(smiles_list))

        return smiles_list

    train_smiles = convert_to_smiles(train, id_to_smiles)
    print('Converted train molecules to SMILES')

    test_smiles = convert_to_smiles(test, id_to_smiles)
    print('Converted test molecules to SMILES')
    test_number = len(test_smiles)
    print('Number of test smiles : ', test_number)

    # Create a list to store the Tanimoto coefficients
    tanimoto_coefficients = []
    tanimoto_max_avg = []
    i = 0
    c = 1
    # Iterate over the test list
    for test_compound in test_smiles:
        # Convert the test compound to a mol object
        test_compound = str(test_compound).replace("['", '')
        test_compound = test_compound.replace("']", '')
        # print('Test: ', test_compound)

        test_mol = Chem.MolFromSmiles(test_compound)
        tanimoto_per_test = []
        c = c + 1
        percent_done = c / test_number
        if c % 100 == 0:
            print(percent_done * 100, '%')
        if test_mol is None:
            print('test_mol is none')
            i = i + 1
        # print(test_mol)
        # Iterate over the train list

        else:
            test_bit = AllChem.GetMorganFingerprintAsBitVect(test_mol, radius=2, nBits=2048)
            for train_compound in train_smiles:
                # Convert the train compound to a mol object
                train_compound = str(train_compound).replace("['", '')
                train_compound = train_compound.replace("']", '')
                train_mol = Chem.MolFromSmiles(train_compound)
                # print('Train: ', train_compound)
                if train_mol is None:
                    print('train_mol is none')

                else:
                    # Calculate the Tanimoto similarity coefficient between the test and train compounds
                    # print('Test: ', test_mol, '.... Train: ', train_mol)
                    tanimoto_coefficient = DataStructs.TanimotoSimilarity(test_bit,
                                                                          AllChem.GetMorganFingerprintAsBitVect(
                                                                              train_mol, radius=2, nBits=2048))
                    if tanimoto_coefficient > 0.9:
                        print('Tan greater than 0.9 : ', tanimoto_coefficient)
                        print(train_compound, test_compound)
                    # print(tanimoto_coefficient)
                    # Append the Tanimoto coefficient to the list
                    tanimoto_coefficients.append(tanimoto_coefficient)
                    tanimoto_per_test.append(tanimoto_coefficient)
        tanimoto_max_avg.append(max(tanimoto_per_test))

    # Print the max Tanimoto coefficient found
    avg_max_tan = stat.mean(tanimoto_max_avg)
    maxx = max(tanimoto_coefficients)
    mean = stat.mean(tanimoto_coefficients)
    std = stat.stdev(tanimoto_coefficients)
    print("i = ", i, ' - amount of test mols not considered due to bad smiles')
    print('The max Tanimoto coefficient found is:', maxx, ' avg: ', mean, ' std: ', std)
    return maxx, mean, std, avg_max_tan


######## Rewrite get max tanimoto for when already have smiles list ######
def convert_to_smiles_only(list_of_id, id_to_smiles_func):

    # currently

    smiles_list = []
    id_list = []
    i = 0
    for iddd in list_of_id:
        iddd = int(iddd)
        if iddd in id_to_smiles_func['id'].values:
            index = id_to_smiles_func[id_to_smiles_func['id'] == iddd].index.values
            smiles_list.append(id_to_smiles_func.loc[index, 'smiles'].values)
        else:
            i = i+1
            #print('Smiles UNKNOWN: ' + str(iddd))
    percent_unknown = i/len(list_of_id)
    print('Percent of unknown smiles: ', percent_unknown)
    return smiles_list

def get_max_Tanimoto_from_smiles(train, test, id_to_smiles):
    ########
    global id_in_test_train
    # Import the required libraries
    import statistics as stat
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import DataStructs

    # Convert the smiles column to a list

    #train = train.iloc[:,0].tolist()
    #test = test.iloc[:,0].tolist()

    #delete duplicates and any id at interesection of train and test
    #puts deleted id into a global variable id_in_train_test
    train = list(set(train))
    test = list(set(test))
    id_in_test_train = list(set(train).intersection(test))



    if id_in_test_train :
        for value in id_in_test_train :
            if value in train:
                train.remove(value)
                print('id in train and test ', value)
    else:
        print("No id to remove")

    train_smiles = convert_to_smiles_only(train, id_to_smiles)
    print('Converted train molecules to SMILES')

    test_smiles = convert_to_smiles_only(test, id_to_smiles)
    print('Converted test molecules to SMILES')
    test_number = len(test_smiles)
    print('Number of test smiles : ', test_number)



    # Create a list to store the Tanimoto coefficients
    tanimoto_coefficients = []
    tanimoto_max_avg = []
    i = 0
    c = 1
    # Iterate over the test list
    for test_compound in test_smiles:
        # Convert the test compound to a mol object
        test_compound = str(test_compound).replace("['", '')
        test_compound = test_compound.replace("']", '')

        test_mol = Chem.MolFromSmiles(test_compound)
        tanimoto_per_test = []
        c = c + 1
        percent_done = c / test_number
        if c % 100 == 0:
            print(percent_done * 100, '%')
        if test_mol is None:
            print('test_mol is none')
            i = i + 1
        # print(test_mol)
        # Iterate over the train list

        else:
            test_bit = AllChem.GetMorganFingerprintAsBitVect(test_mol, radius=2, nBits=2048)
            for train_compound in train_smiles:
                # Convert the train compound to a mol object
                train_compound = str(train_compound).replace("['", '')
                train_compound = train_compound.replace("']", '')
                train_mol = Chem.MolFromSmiles(train_compound)
                # print('Train: ', train_compound)
                if train_mol is None:
                    print('train_mol is none')

                else:
                    # Calculate the Tanimoto similarity coefficient between the test and train compounds
                    # print('Test: ', test_mol, '.... Train: ', train_mol)
                    tanimoto_coefficient = DataStructs.TanimotoSimilarity(test_bit,
                                                                          AllChem.GetMorganFingerprintAsBitVect(
                                                                              train_mol, radius=2, nBits=2048))
                    if tanimoto_coefficient > 0.9:
                        print('Tan greater than 0.9 : ', tanimoto_coefficient)
                        print(train_compound, test_compound)
                    # print(tanimoto_coefficient)
                    # Append the Tanimoto coefficient to the list
                    tanimoto_coefficients.append(tanimoto_coefficient)
                    tanimoto_per_test.append(tanimoto_coefficient)
        tanimoto_max_avg.append(max(tanimoto_per_test))

    # Print the max Tanimoto coefficient found
    avg_max_tan = stat.mean(tanimoto_max_avg)
    maxx = max(tanimoto_coefficients)
    mean = stat.mean(tanimoto_coefficients)
    std = stat.stdev(tanimoto_coefficients)
    print("i = ", i, ' - amount of test mols not considered due to bad smiles')
    print('The max Tanimoto coefficient found is:', maxx, ' avg: ', mean, ' std: ', std)
    return maxx, mean, std, avg_max_tan
