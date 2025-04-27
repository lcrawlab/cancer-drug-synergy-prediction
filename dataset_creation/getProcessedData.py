import numpy as np
import pandas as pd
import torch

# Get the filtered PPIN network (assumes all interactions are known) and creates a mask for the first
# degree neighbors of each gene. Assumes tsv file uses protein1 and protein2 as the column names
# INPUT:
#   ppin_file (str) - path to tab separated file with the PPIN network
# OUTPUT:
#   ppin_mask (Pytorch tensor) - mask of the PPIN network where 0 if not 1st degree neighbors and 1
#                                if 1st degree neighbors
def get_ppin(ppin_file):
    ppin_df = pd.read_csv(ppin_file, sep="\t")
    # if the ppin_df does not have the correct column names, throw an error
    if not ('protein1' in ppin_df.columns and 'protein2' in ppin_df.columns):
        raise ValueError("ppin_df does not have the correct column names")
    elif ppin_df.shape[0] == 0:
        raise ValueError("ppin_df is empty")
    
    nodes = np.sort(np.unique(ppin_df[['protein1', 'protein2']].values))

    node_dict = {}
    for i in range(len(nodes)):
        node_dict[nodes[i]] = i

    first_degree_mask = np.zeros((len(nodes), len(nodes)))
    for i in range(len(ppin_df)):
        node1 = ppin_df.iloc[i]['protein1']
        node2 = ppin_df.iloc[i]['protein2']
        first_degree_mask[node_dict[node1], node_dict[node2]] = 1
        first_degree_mask[node_dict[node2], node_dict[node1]] = 1

    return torch.tensor(first_degree_mask)


# Get the filtered DNA data
# INPUT:
#   dna_df_fp (str) - path to the filtered DNA data file
#   dna_identifier_fp (str) - path to the filtered DNA identifier data file
# OUTPUT:
#   dna_df (Pandas dataframe) - filtered DNA data
#   dna_identifier_df (Pandas dataframe) - filtered DNA identifier data to gene name to entrez id
def get_dna_data(
    dna_df_fp='data_processed/filtered_dna_df.csv',
    dna_identifier_fp='data_processed/dnaexome_identifier_gene_name_entrez_id.csv',
):
    dna_df = pd.read_csv(dna_df_fp, index_col='Unnamed: 0')
    dna_identifier_df = pd.read_csv(dna_identifier_fp)

    dna_identifier_df = dna_identifier_df.rename(
        columns={
            'Identifier (c)': 'Identifier',
            'Gene name (d)': 'Gene',
            'Entrez gene id (e)': 'Entrez',
        }
    )

    return dna_df, dna_identifier_df


# Get the filtered RNA data
# INPUT:
#   rna_df_fp (str) - path to the filtered RNA data file
#   rna_identifier_fp (str) - path to the filtered RNA gene name to entrez id data file
# OUTPUT:
#   rna_df (Pandas dataframe) - filtered RNA data
#   rna_identifier_df (Pandas dataframe) - filtered RNA gene name to entrez id
def get_rna_data(
    rna_df_fp='data_processed/filtered_rna_df.csv',
    rna_identifier_fp='data_processed/rna_gene_name_entrez_id.csv',
):
    rna_df = pd.read_csv(rna_df_fp, index_col='Unnamed: 0')
    rna_identifier_df = pd.read_csv(rna_identifier_fp)

    rna_identifier_df = rna_identifier_df.rename(
        columns={
            'Gene name d': 'Gene',
            'Entrez gene id e': 'Entrez',
        }
    )

    return rna_df, rna_identifier_df


# Get the filtered protein data
# INPUT:
#   protein_df_fp (str) - path to the filtered protein data file
#   protein_identifier_fp (str) - path to the filtered protein identifier data file
# OUTPUT:
#   protein_df (Pandas dataframe) - filtered protein data
#   protein_identifier_df (Pandas dataframe) - filtered protein gene name to entrez id
def get_protein_data(
    protein_df_fp='data_processed/filtered_protein_df.csv',
    protein_identifier_fp='data_processed/protein_identifier_gene_name_entrez_id.csv',
):
    protein_df = pd.read_csv(protein_df_fp, index_col='Unnamed: 0')
    protein_identifier_df = pd.read_csv(protein_identifier_fp)

    protein_identifier_df = protein_identifier_df.rename(
        columns={
            'Identifier c': 'Identifier',
            'Gene name d': 'Gene',
            'Entrez gene id e': 'Entrez',
        }
    )

    return protein_df, protein_identifier_df


# Get the processed drug data
# INPUT:
#   drug_df_fp (str) - path to the filtered drug data file
#   nsc_to_mfp_fp (str) - path to the NSC to MFP bit vector data file
#   nsc_to_prop_fp (str) - path to the NSC to physiochemical property data file
#   cancer_type (str) - cancer type to filter the drug data by
# OUTPUT:
#   drug_df (Pandas dataframe) - filtered drug data
#   nsc_to_mfp (dict) - dictionary mapping NSC drug ID to MFP bit vector
#   nsc_to_prop_df (Pandas dataframe) - dictionary mapping NSC drug ID to physiochemical properties
def get_processed_drug_data(
    drug_df_fp='data_processed/filtered_almcomb_pg.csv',
    drug_comboscore_fp='data_processed/filtered_almcomb_combo.csv',
    nsc_to_mfp_fp='data_processed/almanac_nsc_to_morgan_fingerprints256.tsv',
    nsc_to_prop_fp='data_processed/almanac_nsc_to_properties.tsv',
    cancer_type='ALL',
):
    drug_pg_df = pd.read_csv(drug_df_fp)
    if cancer_type != 'ALL':
        # Check if cancer type belongs in PANEL column
        if cancer_type not in drug_pg_df['PANEL'].unique():
            raise ValueError(f'{cancer_type} is not a valid cancer type.')
        print("Old shape was", drug_pg_df.shape)
        drug_pg_df = drug_pg_df[drug_pg_df['PANEL'] == cancer_type]
        print("New shape is", drug_pg_df.shape)
    # Convert NSC1 and NSC2 columns to ints then strings
    drug_pg_df['NSC1'] = drug_pg_df['NSC1'].astype(int).astype(str)
    drug_pg_df['NSC2'] = drug_pg_df['NSC2'].astype(int).astype(str)
    # Convert CONC1, CONC2, PERCENTGROWTH columns to floats
    drug_pg_df['CONC1'] = drug_pg_df['CONC1'].astype(float)
    drug_pg_df['CONC2'] = drug_pg_df['CONC2'].astype(float)
    drug_pg_df['PERCENTGROWTH'] = drug_pg_df['PERCENTGROWTH'].astype(float)

    # CREATING INVARIANCE TO DRUG COMBINATION ORDER
    # For all rows, add a new row with the NSC1 and CONC1 swapped with NSC2 and CONC2
    drug_pg_df_swapped = drug_pg_df.copy()
    drug_pg_df_swapped['NSC1'] = drug_pg_df['NSC2']
    drug_pg_df_swapped['NSC2'] = drug_pg_df['NSC1']
    drug_pg_df_swapped['CONC1'] = drug_pg_df['CONC2']
    drug_pg_df_swapped['CONC2'] = drug_pg_df['CONC1']
    drug_pg_df = pd.concat([drug_pg_df, drug_pg_df_swapped], ignore_index=True)

    drug_comboscore_df = pd.read_csv(drug_comboscore_fp)
    if cancer_type != 'ALL':
        print("Old shape was", drug_comboscore_df.shape)
        drug_comboscore_df = drug_comboscore_df[drug_comboscore_df['PANEL'] == cancer_type]
        print("New shape is", drug_comboscore_df.shape)
    # Convert NSC1 and NSC2 columns to ints then strings
    drug_comboscore_df['NSC1'] = drug_comboscore_df['NSC1'].astype(int).astype(str)
    drug_comboscore_df['NSC2'] = drug_comboscore_df['NSC2'].astype(int).astype(str)
    drug_comboscore_df['COMBOSCORE'] = drug_comboscore_df['COMBOSCORE'].astype(float)
    drug_comboscore_df['ZIP'] = drug_comboscore_df['ZIP'].astype(float)
    drug_comboscore_df['HSA'] = drug_comboscore_df['HSA'].astype(float)

    # CREATING INVARIANCE TO DRUG COMBINATION ORDER
    # For all rows, add a new row with the NSC1 and NSC2 swapped
    drug_comboscore_df_swapped = drug_comboscore_df.copy()
    drug_comboscore_df_swapped['NSC1'] = drug_comboscore_df['NSC2']
    drug_comboscore_df_swapped['NSC2'] = drug_comboscore_df['NSC1']
    drug_comboscore_df = pd.concat([drug_comboscore_df, drug_comboscore_df_swapped], ignore_index=True)

    nsc_to_mfp = {}
    for line in open(nsc_to_mfp_fp):
        line = line.strip('\n').split('\t')
        nsc_to_mfp[str(int(line[0]))] = [int(val) for val in line[1:]]

    nsc_to_prop_df = pd.read_csv(nsc_to_prop_fp, sep='\t', index_col=0)

    return drug_pg_df, drug_comboscore_df, nsc_to_mfp, nsc_to_prop_df


# Remove the low variance columns from the dataframe
# INPUT:
#   identifier_df (Pandas dataframe) - dataframe mapping the identifier and entrez ids
#   df (Pandas dataframe) - dataframe to remove low variance columns from
#   intersection_entrez_fn - filename containing entrez ids that are in the intersection of all modalities
#   threshold (int) - threshold for what top percentage highest variance to retain (0-100)
# OUTPUT:
#   df (Pandas dataframe) - dataframe with low variance columns removed
def remove_low_var_columns(identifier_df, df, intersection_entrez_fn='data_processed/intersection_entrez_ids.txt', threshold=100):
    intersection_entrez_ids = set()
    with open (intersection_entrez_fn, 'r') as f:
        for line in f:
            intersection_entrez_ids.add(int(line.strip()))
    print('Original intersection entrez IDs:', len(intersection_entrez_ids))
    print('Original number of features:', df.shape[1])

    df_var = df.var()
    sorted_var = df_var.sort_values(ascending=False)
    
    # Calculate how many features to keep based on the percentage
    num_features_to_keep = int(len(sorted_var) * threshold / 100)
    print(f'Keeping top {threshold}% of features = {num_features_to_keep} features')
    
    # Get the identifiers of the top variance features
    high_var_identifiers = sorted_var.index[:num_features_to_keep]
    
    # Get the entrez IDs for these high variance features
    high_var_entrez = set()
    for identifier in high_var_identifiers:
        entrez_values = identifier_df[identifier_df['Identifier'] == identifier]['Entrez'].values
        if len(entrez_values) > 0:
            high_var_entrez.add(entrez_values[0])
    
    # Find intersection with original entrez IDs
    new_intersection_entrez = high_var_entrez.intersection(intersection_entrez_ids)

    # Create dataframe with only high variance columns
    high_var_df = df[high_var_identifiers]

    print('Number of features after removing low variance columns:', high_var_df.shape[1])
    print('Number of unique entrez IDs after removing low variance columns:', len(new_intersection_entrez))

    return high_var_df, new_intersection_entrez

