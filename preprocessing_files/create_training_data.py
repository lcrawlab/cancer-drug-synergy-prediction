from preprocessing_files.string_preprocessing import *
from preprocessing_files.preprocess_data import *
import pandas as pd
import torch


# Assume that everything has been filtered already from the preprocess_data.py script
# Just load in the data from the files you want

# Get the drug data from the NCI ALMANAC combo data
# INPUT:
#  None
# OUTPUT:
#   almcomb_combo_df: pandas dataframe - NCI ALMANAC comboscore drug data
#   almcomb_pg_df: pandas dataframe - NCI ALMANAC percent growth drug data
#   almanac_cl_name_to_id: dict - cell line name to cell line ID
def get_drug_data():
    almcomb_combo_df = pd.read_csv("data_processed/almcomb_comboscore_hsa_zip.csv")
    almcomb_pg_df = pd.read_csv("data_processed/almcomb_concentrations.csv")
    almanac_cl_name_to_id = {}
    with open("data_processed/almanac_cell_line_to_id.csv", "r") as f:
        next(f) #skip header
        # for each line, split by comma and add to the dictionary
        for line in f:
            cl_name, cl_id = line.strip().split(",")
            almanac_cl_name_to_id[cl_name] = int(cl_id)
    
    return almcomb_combo_df, almcomb_pg_df, almanac_cl_name_to_id


# Get the DNA data
# INPUT:
#  None
# OUTPUT:
#   dna_cl_names: set - list of cell line names that have DNA data
#   identifier_to_entrez: dict - identifier to entrez gene ID
#   dna_cl_to_exome_mut: dataframe - dataframe containing DNA mutation data
def get_dna_data():
    dna_cl_names = set()
    dna_identifier_to_entrez = {}
    dna_cl_to_exome_mut = pd.read_csv("data_processed/dnaexome_df.csv")

    with open("data_processed/dnaexome_cell_lines.txt", "r") as f:
        for line in f:
            dna_cl_names.add(line.strip('\n'))
    
    with open("data_processed/dnaexome_identifier_gene_name_entrez_id.csv", "r") as f:
        next(f) #skip header
        for line in f:
            identifier, _, entrez_id = line.strip().split(",")
            if identifier in dna_identifier_to_entrez:
                raise Exception("Duplicate identifier")
            dna_identifier_to_entrez[identifier] = int(entrez_id)

    return dna_cl_names, dna_identifier_to_entrez, dna_cl_to_exome_mut


# Get the RNA data
# INPUT:
#  None
# OUTPUT:
#   rna_cl_names: set - list of cell line names that have RNA data
#   rna_gene_name_to_entrez: dict - gene name to entrez gene ID
#   rna_cl_to_expr: dataframe - dataframe containing RNA data
def get_rna_data():
    rna_cl_names = set()
    rna_gene_name_to_entrez = {}
    rna_cl_to_expr = pd.read_csv("data_processed/rna_df.csv")

    with open("data_processed/rna_cell_lines.txt", "r") as f:
        for line in f:
            rna_cl_names.add(line.strip('\n'))
    
    with open("data_processed/rna_gene_name_entrez_id.csv", "r") as f:
        next(f) #skip header
        for line in f:
            gene_name, entrez_id = line.strip().split(",")
            if gene_name in rna_gene_name_to_entrez:
                raise Exception("Duplicate gene name")
            rna_gene_name_to_entrez[gene_name] = int(entrez_id)

    return rna_cl_names, rna_gene_name_to_entrez, rna_cl_to_expr


# Get the protein data
# INPUT:
#   None
# OUTPUT:
#   protein_cl_names: set - list of cell line names that have protein data
#   identifier_to_entrez: dict - protein identifier to entrez gene ID
#   prot_cl_to_expr: dataframe - dataframe containing RNA data
def get_protein_data():
    protein_cl_names = set()
    protein_identifier_to_entrez = {}
    protein_cl_to_expr = pd.read_csv("data_processed/protein_df.csv")

    with open("data_processed/protein_cell_lines.txt", "r") as f:
        for line in f:
            protein_cl_names.add(line.strip('\n'))
    
    with open("data_processed/protein_identifier_gene_name_entrez_id.csv", "r") as f:
        next(f) #skip header
        for line in f:
            identifier, _, entrez_id = line.strip().split(",")
            if identifier in protein_identifier_to_entrez:
                raise Exception("Duplicate identifier")
            protein_identifier_to_entrez[identifier] = int(entrez_id)

    return protein_cl_names, protein_identifier_to_entrez, protein_cl_to_expr


# Get the STRING data
# INPUT:
#   None
# OUTPUT:
#   string_entrez_ids: set - set of entrez IDs that have STRING data
#   string_interactions_df: pandas dataframe - dataframe containing STRING edge list
#   string_prot_entrez_df: pandas dataframe - dataframe containing STRING protein to entrez ID mapping
def get_string_data():
    string_interactions_df = get_known_STRING_df()
    string_prot_entrez_df = pd.read_csv("data_processed/string_ids_prot_entrez.csv")
    string_entrez_ids = set(string_prot_entrez_df["Entrez_Gene_ID"].unique())
    
    return string_entrez_ids, string_interactions_df, string_prot_entrez_df


# Find intersection of cell lines that have DNA, RNA, protein, and almanac data, return the
# filtered almanac dataframe
# INPUT:
#   almanac_cl_names: set - list of cell line names that have almanac data
#   dna_cl_names: set - list of cell line names that have DNA data
#   rna_cl_names: set - list of cell line names that have RNA data
#   protein_cl_names: set - list of cell line names that have protein data
#   almanac_df: pandas dataframe - dataframe containing almanac data
# OUTPUT:
#   intersection_cl: set - list of cell line names at the intersection of all data modalities
#   filtered_almanac_df: pandas dataframe - dataframe containing only intersection cell lines
def find_intersection_of_cell_lines(
    almanac_cl_names,
    dna_cl_names,
    rna_cl_names,
    protein_cl_names,
    almanac_df,
):
    intersection_cl = (
        almanac_cl_names
        & dna_cl_names
        & rna_cl_names
        & protein_cl_names
    )
    print("Number of cell lines in intersection: " + str(len(intersection_cl)))

    # Filter the almanac_df dataframe, removing any rows that do not have cell lines in the intersection
    original_shape = almanac_df.shape
    cell_line_filter = almanac_df["CELLNAME"].isin(intersection_cl)
    filtered_almanac_df = almanac_df[cell_line_filter]
    filtered_shape = filtered_almanac_df.shape
    print("Original shape: " + str(original_shape) + " Filtered shape: " + str(filtered_shape))

    return intersection_cl, filtered_almanac_df


# Filter dataframe for any cell line modality to only include cell lines present in the intersection
# drop any columns (Genes) with NANs 
# INPUT:
#  df: pandas dataframe - dataframe containing any modality
#  intersection_cl: set - list of cell line names at the intersection of all data modalities
# OUTPUT:
#  df: pandas dataframe - filtered dataframe
def filter_cldf_for_intersection(
    df,
    intersection_cl,
):
    # Filter the df dataframe, removing any rows that do not have cell lines in the intersection
    original_shape = df.shape
    # create cell line filter based on first column of df and whether it is in the intersection_cl
    cell_line_filter = df.iloc[:,0].isin(intersection_cl)
    df = df[cell_line_filter]

    # Drop any columns with NANs
    df = df.dropna(axis=1)

    filtered_shape = df.shape
    print("Original shape: " + str(original_shape) + " Filtered shape: " + str(filtered_shape))

    return df


# Get the entrez ID for any cell line dataframe
# INPUT:
#  df: pandas dataframe - dataframe containing any modality
#  identifier_to_entrez: dict - identifier to entrez gene ID
# OUTPUT:
#  entrez_ids: list - list of entrez IDs
def get_entrez_ids(
    df,
    identifier_to_entrez,
):
    identifiers = df.columns[1:]
    entrez_ids = set([identifier_to_entrez[identifier] for identifier in identifiers])
    return entrez_ids


# Filter dataframe for any cell line modality based on entrez IDs
# INPUT:
#  df: pandas dataframe - dataframe containing any modality
#  identifier_to_entrez: dict - identifier to entrez gene ID
#  entrez_ids: set - set of entrez IDs
# OUTPUT:
#  df: pandas dataframe - filtered dataframe
def filter_entrezdf_for_intersection(
    df,
    identifier_to_entrez,
    entrez_ids,
):
    original_shape = df.shape

    # Filter columns based on Entrez IDs
    first_col = [df.columns[0]]
    columns_to_keep = [col for col in df.columns[1:] if identifier_to_entrez[col] in entrez_ids]

    df = df[first_col + columns_to_keep]

    filtered_shape = df.shape
    print("Original shape: " + str(original_shape) + " Filtered shape: " + str(filtered_shape))

    return df


# Filter STRING interactions based on entrez IDs
# INPUT:
#   string_interactions_df: pandas dataframe - dataframe containing STRING interactions
#   entrez_ids: set - set of entrez IDs to filter by
#   string_prot_entrez_df: pandas dataframe - dataframe containing STRING protein to entrez ID mapping
# OUTPUT:
#   df: pandas dataframe - dataframe containing STRING interactions
def filter_string_for_intersection(
    string_interactions_df,
    entrez_ids,
    string_prot_entrez_df,
):
    # for each entrez ID, get the corresponding protein identifier from string_prot_entrez_df
    string_ids_to_filter_by = []
    for entrez_id in entrez_ids:
        string_id = string_prot_entrez_df[string_prot_entrez_df["Entrez_Gene_ID"] == entrez_id]["STRING_ID"].iloc[0]
        if string_id not in string_ids_to_filter_by:
            string_ids_to_filter_by.append(string_id)
    filtered_string = get_protein_subnetwork(string_interactions_df, string_ids_to_filter_by, both_proteins=False)

    return filtered_string


# DEPRECATED
# Get a tensor determining whether it is the first anchor/library drug order or swapped, add to 
# almanac_df as a new column
# INPUT:
#   almanac_df: pandas dataframe - almanac dataframe
# OUTPUT:
#   swapped_order: tensor - 1 if the anchor/library drug order is the first ordering or swapped
def get_swapped_order(
    almanac_df,
):
    num_samples = almanac_df.shape[0]
    swapped_order = torch.empty(num_samples, dtype=torch.float64)
    original_anchor_library_pairs = set()
    swapped_anchor_library_pairs = set()

    # Iterate through each row in almanac_df and update the swapped_order tensor,
    # original_anchor_library_pairs set, and swapped_anchor_library_pairs set
    for index in range(num_samples):
        row = almanac_df.iloc[index]
        anchor = row["NSC1"]
        library = row["NSC2"]
        anchor_library_pair = (anchor, library)
        swapped_anchor_library_pair = (library, anchor)
        if anchor_library_pair in swapped_anchor_library_pairs:
            swapped_order[index] = 1.0
        else:
            swapped_order[index] = 0.0
            original_anchor_library_pairs.add(anchor_library_pair)
            swapped_anchor_library_pairs.add(swapped_anchor_library_pair)

    # add swapped_order to almanac_df as a new column if interested in using drug order as a feature
    almanac_df["DRUGORDER"] = swapped_order

    return almanac_df


# Filter cell datasets by common cell lines: find intersection of cell lines, then remove any of the
# genes/proteins that have empty values. Find the intersection of the entrez ids for all datasets
# and then filter the datasets by the intersection of entrez ids. Save the filtered datasets.
# INPUT:
#   None
# OUTPUT:
#   filtered_almcomb_combo_df: dataframe of drug data with comboscores
#   filtered_almcomb_pg_df: dataframe of drug data with percent growth
#   filtered_dna_df: dataframe of DNA mutation data
#   filtered_rna_df: dataframe of RNA expression data
#   filtered_protein_df: dataframe of protein expression data
#   filtered_string_df: dataframe of STRING interactions
#   intersection_entrez_ids: set of entrez ids that are in all datasets
def get_filtered_data():
    # Get original data
    almcomb_combo_df, almcomb_pg_df, almanac_cl_name_to_id= get_drug_data()
    dna_cl_names, dna_identifier_to_entrez, dna_df = get_dna_data()
    rna_cl_names, rna_gene_name_to_entrez, rna_df = get_rna_data()
    protein_cl_names, protein_identifier_to_entrez, protein_df = get_protein_data()

    # Find intersection of cell lines
    intersection_cl, filtered_almcomb_combo_df = find_intersection_of_cell_lines(
        set(almanac_cl_name_to_id.keys()),
        dna_cl_names,
        rna_cl_names,
        protein_cl_names,
        almcomb_combo_df
    )

    # Filter the almcomb_combo_df to only include the cell lines in the intersection
    cell_line_filter = almcomb_pg_df["CELLNAME"].isin(intersection_cl)
    filtered_almcomb_pg_df = almcomb_pg_df[cell_line_filter]
    
    # Filter cell data by intersection of cell lines
    cl_filt_dna_df = filter_cldf_for_intersection(dna_df, intersection_cl)
    print("DNA NANs: " + str(cl_filt_dna_df.isnull().values.any()))
    cl_filt_rna_df = filter_cldf_for_intersection(rna_df, intersection_cl)
    print("RNA NANs: " + str(cl_filt_rna_df.isnull().values.any()))
    cl_filt_protein_df = filter_cldf_for_intersection(protein_df, intersection_cl)
    print("Protein NANs: " + str(cl_filt_protein_df.isnull().values.any()))

    # Get entrez ids
    dna_entrez = get_entrez_ids(cl_filt_dna_df, dna_identifier_to_entrez)
    print("Number of DNA entrez ids: " + str(len(dna_entrez)))
    rna_entrez = get_entrez_ids(cl_filt_rna_df, rna_gene_name_to_entrez)
    print("Number of RNA entrez ids: " + str(len(rna_entrez)))
    protein_entrez = get_entrez_ids(cl_filt_protein_df, protein_identifier_to_entrez)
    print("Number of protein entrez ids: " + str(len(protein_entrez)))
    string_entrez, string_interactions_df, string_prot_entrez_df = get_string_data()
    print("Number of string entrez ids: " + str(len(string_entrez)))
    
    # Find intersection of entrez ids
    intersection_entrez_ids = sorted(
        dna_entrez
        & rna_entrez
        & protein_entrez
        & string_entrez
    )
    print("Number of entrez ids in intersection: " + str(len(intersection_entrez_ids)))


    # Filter cell data by intersection of entrez ids
    filtered_dna = filter_entrezdf_for_intersection(cl_filt_dna_df, dna_identifier_to_entrez, intersection_entrez_ids)
    filtered_rna = filter_entrezdf_for_intersection(cl_filt_rna_df, rna_gene_name_to_entrez, intersection_entrez_ids)
    filtered_protein = filter_entrezdf_for_intersection(cl_filt_protein_df, protein_identifier_to_entrez, intersection_entrez_ids)
    filtered_string = filter_string_for_intersection(string_interactions_df, intersection_entrez_ids, string_prot_entrez_df)
    
    # Normalize each of the -omic dataframes using z-score normalization
    dna_cell_lines = filtered_dna.iloc[:, 0]
    numeric_dna_df = filtered_dna.iloc[:, 1:]
    normalized_dna_df = (numeric_dna_df - numeric_dna_df.mean(axis=0)) / numeric_dna_df.std(axis=0)
    filtered_dna = pd.concat([dna_cell_lines, normalized_dna_df], axis=1)
    # RNA data is already normalized by z-score normalization
    protein_cell_lines = filtered_protein.iloc[:, 0]
    numeric_protein_df = filtered_protein.iloc[:, 1:]
    normalized_protein_df = (numeric_protein_df - numeric_protein_df.mean(axis=0)) / numeric_protein_df.std(axis=0)
    filtered_protein = pd.concat([protein_cell_lines, normalized_protein_df], axis=1)

    # Save filtered data, figure out indices
    filtered_almcomb_pg_df.to_csv("data_processed/filtered_almcomb_pg.csv", index=False)
    filtered_almcomb_combo_df.to_csv("data_processed/filtered_almcomb_combo.csv", index=False)
    filtered_dna.to_csv("data_processed/filtered_dna_df.csv", index=False)
    filtered_rna.to_csv("data_processed/filtered_rna_df.csv", index=False)
    filtered_protein.to_csv("data_processed/filtered_protein_df.csv", index=False)
    filtered_string.to_csv("data_processed/filtered_string_df.csv", index=False)

    with open("data_processed/intersection_entrez_ids.txt", "w") as fp:
        for entrez_id in intersection_entrez_ids:
            fp.write(str(entrez_id) + "\n")

    return filtered_almcomb_combo_df, filtered_almcomb_pg_df, filtered_dna, filtered_rna, filtered_protein, filtered_string, intersection_entrez_ids


# Split the data up by cancer type
# INPUT:
#  nci_almanac_combo_df: pandas dataframe - NCI ALMANAC combo data
# OUTPUT:
#  cancer_type_to_df: dict - cancer type to pandas dataframe containing only that cancer type
def split_data_by_cancer_type(
    nci_almanac_combo_df,
):
    cancer_types = nci_almanac_combo_df["PANEL"].unique()
    cancer_type_to_df = {}
    for cancer_type in cancer_types:
        cancer_type_to_df[cancer_type] = nci_almanac_combo_df[nci_almanac_combo_df["PANEL"] == cancer_type]

    # print the number of rows in each cancer type
    for cancer_type in cancer_type_to_df:
        print(cancer_type + ": " + str(len(cancer_type_to_df[cancer_type])))
        print(cancer_type_to_df[cancer_type].columns)

    # Save each cancer type dataframe to different file
    for cancer_type in cancer_type_to_df:
        output_ct = cancer_type.lower().replace(" ", "_")
        cancer_type_to_df[cancer_type].to_csv("data_processed/almanac_by_cancertype/filtered_almanac_df_" + output_ct + ".csv", index=False)

    return cancer_type_to_df

