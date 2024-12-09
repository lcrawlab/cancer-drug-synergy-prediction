from functools import partial
from os.path import exists
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
#from string_preprocessing import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import torch

# Preprocess drug data from NCI ALMANAC, create a dictionary that maps the cell line name to a cell
# line ID, and create a dictionary that maps the drug NSC ID to the drug name. Filter out na values
# and unnecessary columns. Mean center/scale the concentration columns. Save two dictionaries and the
# filtered dataframe. Also get the ComboScore for all of the unique drug and cell line pairs.
# INPUT:
#   mean_ctr_scale: bool - whether to mean center and scale the concentration columns, defaults to False
# OUTPUT:
#   None
def preprocess_drug_data(
    mean_ctr_scale=False,
):
    nci_almanac_combo_df = pd.read_csv("../data/NCI-ALMANAC/ComboDrugGrowth_Nov2017.csv")
    original_shape = nci_almanac_combo_df.shape

    # Create file of cell line name to ID mapping
    cl_names = nci_almanac_combo_df["CELLNAME"].unique()
    # zip together the cell line names and a range of numbers
    almanac_cl_name_to_id = dict(zip(cl_names, range(len(cl_names))))

    # Write the cell line name to ID mapping to a file
    with open("../data_processed/almanac_cell_line_to_id.csv", "w") as f:
        f.write("Cell Line Name,Cell Line ID\n")
        for key in sorted(almanac_cl_name_to_id.keys()):
            f.write("%s,%s\n" % (key, almanac_cl_name_to_id[key]))

    # Filter out NA values
    drug1_filter = nci_almanac_combo_df["NSC1"].notna()
    conc1_filter = nci_almanac_combo_df["CONC1"].notna()
    concunit1_filter = nci_almanac_combo_df["CONCUNIT1"].notna()
    drug2_filter = nci_almanac_combo_df["NSC2"].notna()
    conc2_filter = nci_almanac_combo_df["CONC2"].notna()
    concunit2_filter = nci_almanac_combo_df["CONCUNIT2"].notna()
    synergy_filter = nci_almanac_combo_df["SCORE"].notna()
    cancer_type_filter = nci_almanac_combo_df["PANEL"].notna()
    percentgrowth_filter = nci_almanac_combo_df["PERCENTGROWTH"].notna()

    # Filter the nci_almanac_combo_df dataframe
    nci_almanac_filtered_df = nci_almanac_combo_df[drug1_filter & conc1_filter & concunit1_filter & drug2_filter & conc2_filter & concunit2_filter & synergy_filter & cancer_type_filter & percentgrowth_filter]
    nci_almanac_filtered_df = nci_almanac_filtered_df[["NSC1", "CONC1", "NSC2", "CONC2", "CELLNAME", "PERCENTGROWTH", "SCORE", "PANEL"]]

    # Convert NSC1 and NSC2 to ints
    nci_almanac_filtered_df["NSC1"] = nci_almanac_filtered_df["NSC1"].astype(int)
    nci_almanac_filtered_df["NSC2"] = nci_almanac_filtered_df["NSC2"].astype(int)
    
    if mean_ctr_scale:
        # Mean center and scale here the concentration columns and the synergy score column
        nci_almanac_filtered_df["CONC1"] = (nci_almanac_filtered_df["CONC1"] - nci_almanac_filtered_df["CONC1"].mean()) / nci_almanac_filtered_df["CONC1"].std()
        nci_almanac_filtered_df["CONC2"] = (nci_almanac_filtered_df["CONC2"] - nci_almanac_filtered_df["CONC2"].mean()) / nci_almanac_filtered_df["CONC2"].std()
        
    print("Original shape: " + str(original_shape) + " Preprocessed shape: " + str(nci_almanac_filtered_df.shape))

    # Get the ComboScore for all of the unique drug and cell line pairs
    almanac_comboscore_df = nci_almanac_filtered_df.groupby(["CELLNAME", "NSC1", "NSC2", "PANEL"]).agg({"SCORE": "sum"}).reset_index()
    
    # Write the filtered dataframe to a csv file
    nci_almanac_filtered_df.to_csv("../data_processed/almanac_df.csv", index=False)

    # Write the ComboScore dataframe to a csv file
    almanac_comboscore_df.to_csv("../data_processed/almanac_comboscore_df.csv", index=False)


# Get the NSC id to drug name dictionary
# INPUT:
#   None
# OUTPUT:
#   drug_nsc_to_name: dict - drug NSC ID to drug name
def get_nsc_to_drug_name_dict():
    # Read in the file ComboCompoundNames_small.txt which contains the drug names for the drug IDs and 
    # create a dictionary that maps the drug ID to the drug name
    drug_nsc_to_name = {}
    with open("../data/NCI-ALMANAC/ComboCompoundNames_small.txt", "r") as f:
        for line in f:
            line = line.strip()
            line_split = line.split("\t")
            drug_id = line_split[0]
            if drug_id in drug_nsc_to_name:
                drug_nsc_to_name[drug_id] = drug_nsc_to_name[drug_id] + "/" + line_split[1]
            else:
                drug_nsc_to_name[line_split[0]] = line_split[1]
    return drug_nsc_to_name


# Get the DNA exome sequencing data for the cell lines, remove NA values and unnecessary columns/rows
# Save list of identifiers to gene names and entrez IDs, save dataframe mapping cell line to identifier
# Save dna cell lines for finding intersection
# INPUT:
#   None
# OUTPUT:
#   None
def preprocess_dna_data():
    nci_dna_fp = "../data/NCI-ALMANAC/nci60_DNA__Exome_Seq_none/DNA__Exome_Seq_none.xlsx"
    nci_dna_df = pd.read_excel(nci_dna_fp)

    original_shape = nci_dna_df.shape
    print("Original shape: " + str(original_shape))

    # Drop unnecessary columns and rows
    nci_dna_df = nci_dna_df.drop(nci_dna_df.columns[3:16], axis=1)
    nci_dna_df = nci_dna_df.iloc[12:]
    nci_dna_df = nci_dna_df.iloc[:140186]
    # replace '-' with nan
    nci_dna_df.replace('-', np.nan, inplace=True)
    # drop any rows that have an na value in the first 3 columns
    nci_dna_df = nci_dna_df.dropna(subset=nci_dna_df.columns[:3])

    # Get the column indices with proper cell line names
    column_indices = nci_dna_df.iloc[0]
    nci_dna_df = nci_dna_df.drop(nci_dna_df.index[0]) #drop this row

    new_column_indices = []
    for column_index in column_indices:
        if ':' in str(column_index):
            cell_line = column_index.split(':')[1]
            if cell_line == 'MDA-MB-231':
                new_column_indices.append('MDA-MB-231/ATCC')
            else:
                new_column_indices.append(cell_line)
        elif column_index == 12:
            new_column_indices.append("")
        else:
            new_column_indices.append(column_index)
    nci_dna_df.columns = new_column_indices
    
    nci_dna_df = nci_dna_df.reset_index(drop=True) #reset the index

    # Get cell lines from dna data and save to file
    dna_cell_lines = sorted(list(nci_dna_df.columns[4:]))
    with open("../data_processed/dnaexome_cell_lines.txt", "w") as fp:
        for cl in dna_cell_lines:
            fp.write(cl + "\n")
    
    # Get mapping of identifier to gene name and entrez ID, which are the first 3 columns
    identifier_gene_name_entrez_id_df = nci_dna_df.iloc[:, :3]
    identifier_gene_name_entrez_id_df.to_csv("../data_processed/dnaexome_identifier_gene_name_entrez_id.csv", index=False)

    # Drop the gene name and entrez ID columns
    nci_dna_df = nci_dna_df.drop(nci_dna_df.columns[1:3], axis=1)

    # Transpose the dataframe so that the cell lines are the rows and the identifiers are the columns
    nci_dna_df = nci_dna_df.set_index('Identifier (c)')
    nci_dna_df = nci_dna_df.transpose()
    # sort the columns by the identifier
    nci_dna_df = nci_dna_df.reindex(sorted(nci_dna_df.columns), axis=1)
    print("Transposed shape: " + str(nci_dna_df.shape))

    # Save transposed dataframe to file
    nci_dna_df.to_csv("../data_processed/dnaexome_df.csv", index=True)


# Get the RNA sequencing data for the cell lines, remove NA values and unnecessary columns/rows
# Save list of gene names and entrez IDs, save dataframe mapping cell line to entrez id gene
# expression value. Save rna cell lines for finding intersection
# INPUT:
#   None
# OUTPUT:
#   None
def preprocess_rna_data():
    
    nci_rna_fp = '../data/NCI-ALMANAC/nci60_RNA__5_Platform_Gene_Transcript_Average_z_scores/RNA__5_Platform_Gene_Transcript_Average_z_scores.xls'
    rna_df = pd.read_excel(nci_rna_fp)

    original_shape = rna_df.shape
    print("Original shape: " + str(original_shape))

    # Drop unnecessary columns and rows
    rna_df = rna_df.iloc[9:]
    rna_df = rna_df.drop(rna_df.columns[2:6], axis=1)
    # replace '-' with nan
    rna_df.replace('-', np.nan, inplace=True)
    # drop any rows that have an na value in the first 2 columns
    rna_df = rna_df.dropna(subset=rna_df.columns[:2])

    # Get the column indices with proper cell line names
    column_indices = rna_df.iloc[0]
    rna_df = rna_df.drop(rna_df.index[0]) #drop this row
    
    new_column_indices = []
    for column_index in column_indices:
        if ':' in column_index:
            cell_line = column_index.split(':')[1]
            if cell_line == 'MDA-MB-231':
                new_column_indices.append('MDA-MB-231/ATCC')
            else:
                new_column_indices.append(cell_line)
        else:
            new_column_indices.append(column_index)
    rna_df.columns = new_column_indices
    rna_df = rna_df.reset_index(drop=True) #reset the index

    # Get cell lines from rna data and save to file
    rna_cell_lines = sorted(list(rna_df.columns[2:]))
    with open("../data_processed/rna_cell_lines.txt", "w") as fp:
        for cl in rna_cell_lines:
            fp.write(cl + "\n")
    
    # Get mapping of gene name to entrez id, which are the first 2 columns
    gene_name_entrez_id_df = rna_df.iloc[:, :2]
    gene_name_entrez_id_df.to_csv("../data_processed/rna_gene_name_entrez_id.csv", index=False)

    # Drop the entrez id column
    rna_df = rna_df.drop(rna_df.columns[1], axis=1)

    # Transpose the dataframe so that the cell lines are rows and the entrez ids are columns
    rna_df = rna_df.set_index('Gene name d')
    rna_df = rna_df.transpose()
    # Sort columns by gene name
    rna_df = rna_df.reindex(sorted(rna_df.columns), axis=1)
    transposed_shape = rna_df.shape
    print("Transposed shape: " + str(transposed_shape))

    # Save transposed dataframe to file
    rna_df.to_csv("../data_processed/rna_df.csv", index=True)


# Get the protein expression data for the cell lines, remove NA values and unnecessary columns/rows
# Check if there are duplicates of the same entrez gene ID. Save list of identifiers, gene names,
# and entrez IDs, save dataframe mapping cell line to identifier expression value. Save protein cell
# lines for finding intersection
# INPUT:
#   None
# OUTPUT:
#   None
def preprocess_protein_data():
    
    protein_fp = '../data/NCI-ALMANAC/nci60_Protein__SWATH_(Mass_spectrometry)_Protein/Protein__SWATH_(Mass_spectrometry)_Protein.xls'
    protein_df = pd.read_excel(protein_fp)

    original_shape = protein_df.shape
    print("Original shape: " + str(original_shape))

    # Drop unnecessary columns and rows
    protein_df = protein_df.iloc[9:]
    protein_df = protein_df.drop(protein_df.columns[3:9], axis=1)

    # replace '-' with nan
    protein_df.replace('-', np.nan, inplace=True)

    # drop any rows that have an na value in the first 3 columns
    protein_df = protein_df.dropna(subset=protein_df.columns[:3])

    # Get the column indices with proper cell line names
    column_indices = protein_df.iloc[0]
    protein_df = protein_df.drop(protein_df.index[0]) #drop this row
    
    new_column_indices = []
    for column_index in column_indices:
        if ':' in column_index:
            cell_line = column_index.split(':')[1]
            if cell_line == 'MDA-MB-231':
                new_column_indices.append('MDA-MB-231/ATCC')
            else:
                new_column_indices.append(cell_line)
        else:
            new_column_indices.append(column_index)
    protein_df.columns = new_column_indices
    protein_df = protein_df.drop('MDA-N', axis=1) #drop column MDA-N, filled with NA values

    protein_df = protein_df.reset_index(drop=True) #reset the index

    # Get cell lines from rna data and save to file
    protein_cell_lines = sorted(list(protein_df.columns[3:]))
    with open("../data_processed/protein_cell_lines.txt", "w") as fp:
        for cl in protein_cell_lines:
            fp.write(cl + "\n")
    
    # Get mapping of identifier to gene name to entrez id, which are the first 3 columns
    identifier_gene_name_entrez_id_df = protein_df.iloc[:, :3]
    identifier_gene_name_entrez_id_df.to_csv("../data_processed/protein_identifier_gene_name_entrez_id.csv", index=False)

    # Drop the gene name and entrez_id column
    protein_df = protein_df.drop(protein_df.columns[1:3], axis=1)

    # Transpose the dataframe so that the cell lines are rows and the entrez ids are columns
    protein_df = protein_df.set_index('Identifier c')
    protein_df = protein_df.transpose()
    # Sort columns by identifier c
    protein_df = protein_df.reindex(sorted(protein_df.columns), axis=1)
    transposed_shape = protein_df.shape
    print("Transposed shape: " + str(transposed_shape))

    # Save transposed dataframe to file
    protein_df.to_csv("../data_processed/protein_df.csv", index=True)


# Get the SMILES and Morgan Fingerprints for the NCI ALMANAC drugs, save to file
# INPUT:
#   fp_len: int - length of the Morgan Fingerprints, defaults to 1024
# OUTPUT:
#   nsc_to_morgan_fingerprints: dict - drug NSC ID to drug Morgan Fingerprints
def get_smiles_and_fingerprints(
    fp_len=1024,
):
    drug_nsc_to_name = get_nsc_to_drug_name_dict()
    nci_sdf = Chem.SDMolSupplier("../data/NCI-ALMANAC/ComboCompoundSet.sdf") # Use RDKit
    sdf_names = set()
    nsc_to_smiles = {}
    nsc_to_morgan_fingerprints = {}
    for mol in nci_sdf:
        if mol:
            nsc = mol.GetProp("_Name")
            sdf_names.add(nsc)
            nsc_to_smiles[nsc] = Chem.MolToSmiles(mol)
            mf = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_len)
            np_mf = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(mf, np_mf)
            nsc_to_morgan_fingerprints[nsc] = np_mf

    # Find difference between the nci_almanac_nsc_to_name dictionary and the sdf_names set
    almanac_nscs = set(drug_nsc_to_name.keys())
    missing_almanac_nscs = almanac_nscs - sdf_names
    # extra_sdfs = sdf_names - almanac_nscs
    # print("Number of NSCs in NCI ALMANAC but not in SDF file: {}".format(len(missing_almanac_nscs)))
    # print(missing_almanac_nscs)
    # print("Number of molecules in SDF file but not in NCI ALMANAC: {}".format(len(extra_sdfs)))

    # Hard coded canonical SMILES for the missing NSCs, found manually by searching name in PubChem
    # and selecting the best match. Theoretically PubChemPy should be able to do this automatically
    # but the URL seems to not be working.
    hardcoded_nsc_to_smiles = {
        '119875': 'N.N.Cl[Pt]Cl',
        '266046': 'C1CCC(C(C1)[NH-])[NH-].C(=O)(C(=O)O)O.[Pt+2]',
        '753082': 'CCCS(=O)(=O)NC1=C(C(=C(C=C1)F)C(=O)C2=CNC3=C2C=C(C=N3)C4=CC=C(C=C4)Cl)F',
    }

    # Find remaining SMILES and Morgan Fingerprints for the missing NSCs
    for nsc in missing_almanac_nscs:
        mol = Chem.MolFromSmiles(hardcoded_nsc_to_smiles[nsc])
        if mol:
            nsc_to_smiles[nsc] = Chem.MolToSmiles(mol)
            mf = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_len)
            np_mf = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(mf, np_mf)
            nsc_to_morgan_fingerprints[nsc] = np_mf
        else:
            print("Cant find molecule from SMILES for NSC {}".format(nsc))

    # Print how many SMILES and Morgan Fingerprints we have
    # print("Number of SMILES: {}".format(len(nsc_to_smiles)))
    # print("Number of Morgan Fingerprints: {}".format(len(nsc_to_morgan_fingerprints)))

    # Write the SMILES to a file
    with open("../data_processed/almanac_nsc_to_smiles.txt", "w") as f:
        for nsc in nsc_to_smiles:
            f.write(nsc + "\t" + nsc_to_smiles[nsc] + "\n")

    # Write the Morgan Fingerprints to a file
    with open("../data_processed/almanac_nsc_to_morgan_fingerprints" + str(fp_len) + ".tsv", "w") as f:
        for nsc in nsc_to_morgan_fingerprints:
            f.write(nsc)
            for elem in nsc_to_morgan_fingerprints[nsc]:
                f.write('\t' + str(int(elem)))
            f.write('\n')

    return nsc_to_morgan_fingerprints


# Get the physicochemical properties of the NCI ALMANAC drugs, save to file
# INPUT:
#   None
# OUTPUT:
#   nsc_to_properties: dict - drug NSC ID to drug physicochemical properties
def get_physicochemical_properties():
    drug_nsc_to_name = get_nsc_to_drug_name_dict()
    print("Is 66847 in drug_nsc_to_name? {}".format('66847' in drug_nsc_to_name))
    nci_sdf = Chem.SDMolSupplier("../data/NCI-ALMANAC/ComboCompoundSet.sdf") # Use RDKit
    sdf_names = set()
    nsc_to_properties = {}
    for mol in nci_sdf:
        if mol:
            nsc = mol.GetProp("_Name")
            sdf_names.add(nsc)
            properties = {
                "MolWt": Descriptors.MolWt(mol),
                "TPSA": Descriptors.TPSA(mol),
                "LogP": Descriptors.MolLogP(mol),
                "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
                "NumAromaticRings": Descriptors.NumAromaticRings(mol),
                "NumHDonors": Descriptors.NumHDonors(mol),
                "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            }
            nsc_to_properties[nsc] = properties

    # Find difference between the nci_almanac_nsc_to_name dictionary and the sdf_names set
    almanac_nscs = set(drug_nsc_to_name.keys())
    print("Is 66847 in sdf_names? {}".format('66847' in sdf_names))
    missing_almanac_nscs = almanac_nscs - sdf_names
    extra_sdfs = sdf_names - almanac_nscs
    print("Number of NSCs in NCI ALMANAC but not in SDF file: {}".format(len(missing_almanac_nscs)))
    print(missing_almanac_nscs)
    print("Number of molecules in SDF file but not in NCI ALMANAC: {}".format(len(extra_sdfs)))

    # Hard coded canonical SMILES for the missing NSCs, found manually by searching name in PubChem
    # and selecting the best match. Theoretically PubChemPy should be able to do this automatically
    # but the URL seems to not be working.
    hardcoded_nsc_to_smiles = {
        '119875': 'N.N.Cl[Pt]Cl',
        '266046': 'C1CCC(C(C1)[NH-])[NH-].C(=O)(C(=O)O)O.[Pt+2]',
        '753082': 'CCCS(=O)(=O)NC1=C(C(=C(C=C1)F)C(=O)C2=CNC3=C2C=C(C=N3)C4=CC=C(C=C4)Cl)F',
    }

    # Find remaining molecules for the missing NSCs
    for nsc in missing_almanac_nscs:
        mol = Chem.MolFromSmiles(hardcoded_nsc_to_smiles[nsc])
        if mol:
            properties = {
                "MolWt": Descriptors.MolWt(mol),
                "TPSA": Descriptors.TPSA(mol),
                "LogP": Descriptors.MolLogP(mol),
                "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
                "NumAromaticRings": Descriptors.NumAromaticRings(mol),
                "NumHDonors": Descriptors.NumHDonors(mol),
                "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            }
            nsc_to_properties[nsc] = properties
        else:
            print("Cant find molecule from SMILES for NSC {}".format(nsc))

    # Write the properties to a file
    with open("../data_processed/almanac_nsc_to_properties.tsv", "w") as f:
        f.write("NSC\tName\tMolWt\tTPSA\tLogP\tNumAliphaticRings\tNumAromaticRings\tNumHDonors\tNumHAcceptors\n")
        nsc_ints = sorted([int(nsc) for nsc in nsc_to_properties.keys()])
        for nsc_int in nsc_ints:
            nsc = str(nsc_int)
            f.write(nsc + "\t" + drug_nsc_to_name[nsc] + \
                    "\t" + str(nsc_to_properties[nsc]["MolWt"]) + \
                    "\t" + str(nsc_to_properties[nsc]["TPSA"]) + \
                    "\t" + str(nsc_to_properties[nsc]["LogP"]) + \
                    "\t" + str(nsc_to_properties[nsc]["NumAliphaticRings"]) + \
                    "\t" + str(nsc_to_properties[nsc]["NumAromaticRings"]) + \
                    "\t" + str(nsc_to_properties[nsc]["NumHDonors"]) + \
                    "\t" + str(nsc_to_properties[nsc]["NumHAcceptors"]) + "\n")

    return nsc_to_properties
