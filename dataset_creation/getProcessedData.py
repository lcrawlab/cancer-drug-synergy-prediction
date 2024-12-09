import numpy as np
import pandas as pd
import torch
# from torch.utils.data import random_split, Dataset, DataLoader
# from torchmetrics.regression import MeanSquaredError
# from torchmetrics.regression import PearsonCorrCoef


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
    dna_df_fp='../data_processed/filtered_dna_df.csv',
    dna_identifier_fp='../data_processed/dnaexome_identifier_gene_name_entrez_id.csv',
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
    rna_df_fp='../data_processed/filtered_rna_df.csv',
    rna_identifier_fp='../data_processed/rna_gene_name_entrez_id.csv',
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
    protein_df_fp='../data_processed/filtered_protein_df.csv',
    protein_identifier_fp='../data_processed/protein_identifier_gene_name_entrez_id.csv',
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
    drug_df_fp='../data_processed/filtered_almanac_df.csv',
    drug_comboscore_fp='../data_processed/filtered_almanac_comboscore_df.csv',
    nsc_to_mfp_fp='../data_processed/almanac_nsc_to_morgan_fingerprints256.tsv',
    nsc_to_prop_fp='../data_processed/almanac_nsc_to_properties.tsv',
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
    # Convert CONC1, CONC2, SCORE, DRUGORDER columns to floats
    drug_pg_df['CONC1'] = drug_pg_df['CONC1'].astype(float)
    drug_pg_df['CONC2'] = drug_pg_df['CONC2'].astype(float)
    drug_pg_df['SCORE'] = drug_pg_df['SCORE'].astype(float)
    # Drop the SCORE column
    drug_pg_df = drug_pg_df.drop(columns=['SCORE'])
    drug_pg_df['PERCENTGROWTH'] = drug_pg_df['PERCENTGROWTH'].astype(float)
    drug_pg_df['DRUGORDER'] = drug_pg_df['DRUGORDER'].astype(float)

    drug_comboscore_df = pd.read_csv(drug_comboscore_fp)
    if cancer_type != 'ALL':
        print("Old shape was", drug_comboscore_df.shape)
        drug_comboscore_df = drug_comboscore_df[drug_comboscore_df['PANEL'] == cancer_type]
        print("New shape is", drug_comboscore_df.shape)
    # Convert NSC1 and NSC2 columns to ints then strings
    drug_comboscore_df['NSC1'] = drug_comboscore_df['NSC1'].astype(int).astype(str)
    drug_comboscore_df['NSC2'] = drug_comboscore_df['NSC2'].astype(int).astype(str)
    drug_comboscore_df['SCORE'] = drug_comboscore_df['SCORE'].astype(float)

    nsc_to_mfp = {}
    for line in open(nsc_to_mfp_fp):
        line = line.strip('\n').split('\t')
        nsc_to_mfp[str(int(line[0]))] = [int(val) for val in line[1:]]

    nsc_to_prop_df = pd.read_csv(nsc_to_prop_fp, sep='\t', index_col=0)

    return drug_pg_df, drug_comboscore_df, nsc_to_mfp, nsc_to_prop_df


# def get_drug_data(
#     drug_df_fp='data_processed/filtered_almanac_df.csv',
#     nsc_to_mfp_fp='data_processed/almanac_nsc_to_morgan_fingerprints256.tsv',
#     nsc_to_prop_fp='data_processed/almanac_nsc_to_properties.tsv',
#     binary_classification=False,
# ):
#     drug_df = pd.read_csv(drug_df_fp)
#     # Convert NSC1 and NSC2 columns to ints then strings
#     drug_df['NSC1'] = drug_df['NSC1'].astype(int).astype(str)
#     drug_df['NSC2'] = drug_df['NSC2'].astype(int).astype(str)
#     # Convert CONC1, CONC2, SCORE, DRUGORDER columns to floats
#     drug_df['CONC1'] = drug_df['CONC1'].astype(float)
#     drug_df['CONC2'] = drug_df['CONC2'].astype(float)
#     drug_df['SCORE'] = drug_df['SCORE'].astype(float)
#     drug_df['DRUGORDER'] = drug_df['DRUGORDER'].astype(float)

#     if binary_classification:
#         # Drop rows with SCORE == 0
#         drug_df = drug_df[drug_df['SCORE'] != 0]
#         # Convert SCORE to 1 if >0, and 0 if < 0
#         drug_df['SCORE'] = drug_df['SCORE'].apply(lambda x: 1.0 if x > 0 else 0.0)

#     nsc_to_mfp = {}
#     for line in open(nsc_to_mfp_fp):
#         line = line.strip('\n').split('\t')
#         nsc_to_mfp[str(int(line[0]))] = [int(val) for val in line[1:]]

#     nsc_to_prop_df = pd.read_csv(nsc_to_prop_fp, sep='\t', index_col=0)

#     return drug_df, nsc_to_mfp, nsc_to_prop_df


# # Order cell line data based on the order of the rows of the drug data
# # INPUT:
# #   ordered_cl (list) - list of cell lines tested on in order of the rows of Z (corresponding drug data)
# #   omic_df (Pandas dataframe) - omic data
# #   omic_identifier_df (Pandas dataframe) - omic identifier data
# # OUTPUT:
# #   ordered_omic_features (Pytorch tensor) - cell line features matrix corresponding to the ordered_cl list
# #   omic_col_identifiers (Pandas dataframe) - dataframe mapping the column index to the -omic id, gene, entrez
# def ordered_omic_features(
#     ordered_cl,
#     omic_df,
#     omic_identifier_df,
# ):
#     # Extract omic features using tensor operations
#     ordered_omic_features = torch.tensor(omic_df.loc[ordered_cl].values, dtype=torch.float64)

#     # Extract identifiers directly using pandas operations
#     identifiers = omic_df.columns
#     genes = None
#     if 'Identifier' in omic_identifier_df.columns:
#         gene_info = omic_identifier_df.set_index('Identifier').loc[identifiers, ['Gene', 'Entrez']].reset_index()
#         genes = gene_info['Gene'].values
#     else:
#         gene_info = omic_identifier_df.set_index('Gene').loc[identifiers, ['Entrez']].reset_index()
#         genes = gene_info.index.values

#     # Create the DataFrame with column information
#     omic_col_identifiers = pd.DataFrame({
#         'Column': range(ordered_omic_features.shape[1]),
#         'Identifier': identifiers,
#         'Gene': genes,
#         'Entrez': gene_info['Entrez'].values,
#     })

#     return ordered_omic_features, omic_col_identifiers


# Get mask between data modalities by entrez id
# INPUT:
#   modality1_df (Pandas dataframe) - modality 1 data
#   modality1_identifier_df (Pandas dataframe) - modality 1 identifier data with entrez id
#   modality2_df (Pandas dataframe) - modality 2 data
#   modality2_identifier_df (Pandas dataframe) - modality 2 identifier data with entrez id
# OUTPUT:
#   mask (Pytorch tensor) - mask of the omic data where 0 if not in both DNA and RNA data and 1 if in both
def get_mask_by_entrez_id(
    modality1_df,
    modality1_identifier_df,
    modality2_df,
    modality2_identifier_df,
):
    # check if modality1_identifier_df and modality2_identifier_df have the correct column names
    if not (
        'Identifier' in modality1_identifier_df.columns and 
        'Identifier' in modality2_identifier_df.columns and
        'Entrez' in modality1_identifier_df.columns and
        'Entrez' in modality2_identifier_df.columns
    ):
        raise ValueError("modality1_identifier_df and modality2_identifier_df do not have the correct column names")
    elif modality1_identifier_df.shape[0] == 0 or modality2_identifier_df.shape[0] == 0:
        raise ValueError("modality1_identifier_df or modality2_identifier_df is empty")
    raise NotImplementedError


# # Assemble the drug data, split into training, tuning, and testing sets, and center and scale the
# # data if specified
# # INPUT:
# #   drug_df (Pandas dataframe) - filtered drug data
# #   random_seed (int) - random seed for shuffling the rows of the drug_df
# #   train_tune_test_split (list of float) - proportion of the data to use for training, tuning, and testing
# #   include_conc (bool) - whether or not to include drug concentration data
# #   include_order (bool) - whether or not to include drug order data
# #   include_mfp (bool) - whether or not to include morgan fingerprint data
# #   include_pcp (bool) - whether or not to include physiochemical property data
# #   nsc_to_mfp (dict) - dictionary mapping NSC drug ID to MFP bit vector
# #   nsc_to_prop_df (Pandas dataframe) - dictionary mapping NSC drug ID to physiochemical properties
# #   center_scale (bool) - whether or not to center and scale the data
# #   enforce_balanced_class (bool) - whether or not to enforce balanced class in the training set
# # OUTPUT:
# #   Z_train (Pytorch tensor) - drug covariates matrix for training
# #   Z_tune (Pytorch tensor) - drug covariates matrix for tuning
# #   Z_test (Pytorch tensor) - drug covariates matrix for testing
# #   ordered_cl_train (list) - list of cell lines tested on in order of the rows of Z train
# #   ordered_cl_tune (list) - list of cell lines tested on in order of the rows of Z tune
# #   ordered_cl_test (list) - list of cell lines tested on in order of the rows of Z test
# #   Y_train (Pytorch tensor) - labels matrix for training
# #   Y_tune (Pytorch tensor) - labels matrix for tuning
# #   Y_test (Pytorch tensor) - labels matrix for testing
# def assemble_drug_data(
#     drug_df,
#     random_seed=42,
#     train_tune_test_split=[0.8, 0.1, 0.1],
#     include_conc=True,
#     include_order=True,
#     include_mfp=True,
#     nsc_to_mfp=None,
#     include_pcp=False,
#     nsc_to_prop_df=None,
#     center_scale=False,
#     enforce_balanced_class=False,
# ):
    
#     # Error check train_tune_test_split
#     if sum(train_tune_test_split) != 1.0:
#         raise ValueError("train_tune_test_split must sum to 1")
    
#     # Error check nsc_to_mfp
#     if include_mfp and nsc_to_mfp is None:
#         raise ValueError("nsc_to_mfp must be specified if include_mfp is True")
    
#     # Error check nsc_to_prop_df
#     if include_pcp and nsc_to_prop_df is None:
#         raise ValueError("nsc_to_prop_df must be specified if include_pcp is True")

#     if enforce_balanced_class:
#         print("Enforcing balanced synergy vs non-synergy class")
#         # Split the drug_df into synergistic and not synergistic
#         drug_df_synergistic = drug_df[drug_df['SCORE'] > 0]
#         drug_df_not_synergistic = drug_df[drug_df['SCORE'] <= 0]

#         # Get the number of samples in each class
#         num_samples_synergistic = drug_df_synergistic.shape[0]
#         num_samples_not_synergistic = drug_df_not_synergistic.shape[0]

#         min_samples = min(num_samples_synergistic, num_samples_not_synergistic)

#         # Randomly sample the minimum number of samples from each class
#         print("Getting balanced classes")
#         drug_df_synergistic = drug_df_synergistic.sample(n=min_samples)
#         drug_df_not_synergistic = drug_df_not_synergistic.sample(n=min_samples)

#         # Concatenate the balanced classes back together
#         drug_df = pd.concat([drug_df_synergistic, drug_df_not_synergistic], ignore_index=True)
    
#     # Randomly shuffle the rows of the drug_df
#     print("Shuffling the drug data...")
#     drug_df = drug_df.sample(frac=1).reset_index(drop=True)
    
#     num_samples = drug_df.shape[0]

#     # Get the number of columns in the Z matrix based on include flags
#     z_col = 0
#     mfp_len = 0
#     if include_conc:
#         print("Including concentration data...")
#         z_col += 2
#     if include_order:
#         print("Including drug order data...")
#         z_col += 1
#     if include_mfp:
#         print("Including morgan fingerprint data...")
#         mfp_len = len(nsc_to_mfp['740'])
#         z_col += 2 * mfp_len
#     if include_pcp:
#         print("Including physiochemical property data...")
#         z_col += 7 * 2

#     Z = torch.zeros(num_samples, z_col, dtype=torch.float64)
    
#     #Convert the drug_df to a pytorch tensor including columns based on the include flags
#     col_index = 0
#     if include_conc:
#         Z[:, col_index] = torch.tensor(drug_df['CONC1'], dtype=torch.float64)
#         Z[:, col_index + 1] = torch.tensor(drug_df['CONC2'], dtype=torch.float64)
#         col_index += 2
#     if include_order:
#         Z[:, col_index] = torch.tensor(drug_df['DRUGORDER'], dtype=torch.float64)
#         col_index += 1
#     if include_mfp:
#         drug1s = drug_df['NSC1']
#         drug2s = drug_df['NSC2']
#         # map the NSC drug IDs to the MFP bit vectors
#         for i in range(num_samples):
#             mfp1 = nsc_to_mfp[str(drug1s[i])]
#             mfp2 = nsc_to_mfp[str(drug2s[i])]
#             Z[i, col_index:col_index + mfp_len] = torch.tensor(mfp1, dtype=torch.float64)
#             Z[i, col_index + mfp_len:col_index + 2 * mfp_len] = torch.tensor(mfp2, dtype=torch.float64)
#     if include_pcp:
#         drug1s = drug_df['NSC1']
#         drug2s = drug_df['NSC2']
#         # map the NSC drug IDs to the physiochemical properties
#         for i in range(num_samples):
#             prop1 = nsc_to_prop_df.loc[int(drug1s[i]), 'MolWt':'NumHAcceptors']
#             prop2 = nsc_to_prop_df.loc[int(drug2s[i]), 'MolWt':'NumHAcceptors']
#             Z[i, col_index:col_index + 7] = torch.tensor(prop1, dtype=torch.float64)
#             Z[i, col_index + 7:col_index + 14] = torch.tensor(prop2, dtype=torch.float64)


#     # Get the corresponding SCORE column and convert to a pytorch tensor, this is Y
#     Y = torch.tensor(drug_df['SCORE'], dtype=torch.float64)

#     # Get the corresponding CELLNAME column and convert to a list, this is ordered_cl
#     ordered_cl = drug_df['CELLNAME'].tolist()

#     # Center and scale each column in Z, but NOT the synergy score since we need to keep it as is for accuracy metrics
#     if center_scale:
#         print("Centering and scaling the drug data...")
#         for i in range(Z.shape[1]):
#             if i != col_index:
#                 Z[:, i] = (Z[:, i] - torch.mean(Z[:, i])) / torch.std(Z[:, i])

#     # Get number of rows of drug_df to split into training, tuning, and testing sets
#     num_train = int(num_samples * train_tune_test_split[0])
#     num_tune = int(num_samples * train_tune_test_split[1])

#     # Split the data into training, tuning, and testing sets
#     Z_train = Z[:num_train, :]
#     print("Z_train shape: {}".format(Z_train.shape))
#     Z_tune = Z[num_train:num_train + num_tune, :]
#     print("Z_tune shape: {}".format(Z_tune.shape))
#     Z_test = Z[num_train + num_tune:, :]
#     print("Z_test shape: {}".format(Z_test.shape))
#     Y_train = Y[:num_train]
#     print("Y_train shape: {}".format(Y_train.shape))
#     Y_tune = Y[num_train:num_train + num_tune]
#     print("Y_tune shape: {}".format(Y_tune.shape))
#     Y_test = Y[num_train + num_tune:]
#     print("Y_test shape: {}".format(Y_test.shape))
#     ordered_cl_train = ordered_cl[:num_train]
#     print("Number of ordered_cl_train: {}".format(len(ordered_cl_train)))
#     ordered_cl_tune = ordered_cl[num_train:num_train + num_tune]
#     print("Number of ordered_cl_tune: {}".format(len(ordered_cl_tune)))
#     ordered_cl_test = ordered_cl[num_train + num_tune:]
#     print("Number of ordered_cl_test: {}".format(len(ordered_cl_test)))

#     return Z_train, Z_tune, Z_test, ordered_cl_train, ordered_cl_tune, ordered_cl_test, Y_train, Y_tune, Y_test


# # Assemble cell line data
# # INPUT:
# #   ordered_cl (list) - list of cell lines tested on in order of the rows of Z (corresponding drug data)
# #   include_dna (bool) - whether or not to include DNA data
# #   include_rna (bool) - whether or not to include RNA data
# #   include_protein (bool) - whether or not to include protein data
# #   num_topvar (int) - number of top variance features to return, optional
# #   dna_index_filter (list) - list of indices to filter the dna data by, optional
# # OUTPUT:
# #   X (Pytorch tensor) - cell line features matrix for training
# #   feature_list (Pandas dataframe) - dataframe mapping the column index to the -omic id, gene,
# #       entrez, and IF use_topvar, also previous column index
# #   indices (list) - list of indices of the top variance features, empty list if use_topvar is false
# #       and index_filter is empty
# def assemble_cell_line_data(
#     ordered_cl,
#     include_dna=True,
#     include_rna=True,
#     include_protein=True,
#     num_topvar=0,
#     dna_index_filter=[],
# ):
#     if not (include_dna or include_rna or include_protein):
#         raise ValueError("At least one of include_dna, include_rna, or include_protein must be True")
    
#     if num_topvar > 0 and len(dna_index_filter) > 0:
#         raise ValueError("num_topvar and index_filter cannot both be True, either find the top \
#                          variance features or filter by indices for DNA data, not both")
    
#     data_types = []
#     if include_dna:
#         print("Including DNA data...")
#         omic_df, identifier_df = get_dna_data()
#         data_types.append(('DNA', get_dna_data()))
#     if include_rna:
#         print("Including RNA data...")
#         data_types.append(('RNA', get_rna_data()))
#     if include_protein:
#         print("Including protein data...")
#         data_types.append(('Protein', get_protein_data()))

#     X_list = []
#     feature_list = []
#     indices = [] # don't pass through DNA indices by default unless they were used or generated
    
#     for omic_type, (omic_df, identifier_df) in data_types:
#         ordered_features, ordered_identifiers = ordered_omic_features(ordered_cl, omic_df, identifier_df)
#         if omic_type == 'DNA' and num_topvar > 0:
#             # Note that DNA is always the first to be included, so indices_to_append can start from 0
#             ordered_features, ordered_identifiers, indices = get_top_variance_features(num_topvar, ordered_features, ordered_identifiers)
#         elif omic_type == 'DNA' and len(dna_index_filter) > 0:
#             ordered_features = ordered_features[:, dna_index_filter] # only keep the columns that match the dna index filter
#             ordered_identifiers = ordered_identifiers.iloc[dna_index_filter].reset_index(drop=True) # reset the index of the identifiers
#             indices = dna_index_filter # pass on the dna indices that were filtered for
#         X_list.append(ordered_features)
#         feature_list.append(ordered_identifiers)

#     X = torch.cat(X_list, dim=1)
#     feature_list = pd.concat(feature_list, axis=0)
    
#     return X, feature_list, indices


# Remove the low variance columns from the dataframe
# INPUT:
#   identifier_df (Pandas dataframe) - dataframe mapping the identifier and entrez ids
#   df (Pandas dataframe) - dataframe to remove low variance columns from
#   intersection_entrez_fn - filename containing entrez ids that are in the intersection of all modalities
#   threshold (float) - threshold for variance to remove columns
# OUTPUT:
#   df (Pandas dataframe) - dataframe with low variance columns removed
def remove_low_var_columns(identifier_df, df, intersection_entrez_fn='../data_processed/intersection_entrez_ids.txt', threshold=0):
    intersection_entrez_ids = set()
    with open (intersection_entrez_fn, 'r') as f:
        for line in f:
            intersection_entrez_ids.add(int(line.strip()))
    print('Original intersection entrez IDs:', len(intersection_entrez_ids))
    print('Original number of features:', df.shape[1])

    df_var = df.var()
    high_var = df_var[df_var > threshold]
    high_var_entrez = set(identifier_df[identifier_df['Identifier'] == identifier]['Entrez'].values[0] for identifier in high_var.index)
    new_intersection_entrez = high_var_entrez.intersection(intersection_entrez_ids)

    high_var_df = df[high_var.index]

    print('Number of features after removing low variance columns:', high_var_df.shape[1])
    print('Number of unique entrez IDs after removing low variance columns:', len(new_intersection_entrez))

    return high_var_df, new_intersection_entrez


# # Get highest variance features
# # INPUT:
# # Find the top mostly high variance features in the dataset
# # INPUT:
# #   num_top_var (int): Number of top variance features to return
# #   X (tensor): Tensor of data to get the top variance features from
# #   feature_list (dataframe): Dataframe of feature list to get the top variance features from
# # OUTPUT:
# #   X_filtered (tensor): Tensor of data with only the top variance features
# #   feature_list_filtered (dataframe): Dataframe of feature list with only the top variance features
# #   topvar_indices (tensor): Tensor of indices of the top variance features
# def get_top_variance_features(
#     num_top_var,
#     X,
#     feature_list,
# ):
#     print("Getting the top " + str(num_top_var) + " high variance features")

#     # Set feature_list index to 'Column'
#     feature_list.set_index('Column', inplace=True)

#     #print("X shape: " + str(X.shape))
#     #print("Feature list shape: " + str(feature_list.shape))

#     X_var = torch.var(X, dim=0)

#     # Rank the features by variance, then print top 10 features and their variance
#     variance, indices = torch.sort(X_var, descending=True)

#     # Save the top high variance features to a file and the indices that they originally came from
#     topvar_indices = indices[:num_top_var]
#     topvar_X = X[:, topvar_indices]
#     #print("Top Highest Variance Shape: " + str(topvar_X.shape))

#     # Save the rows of feature_list that correspond to the top 1000 features to a tsv
#     topvar_feature_list = feature_list.iloc[topvar_indices]
#     topvar_feature_list.reset_index(inplace=True)
#     topvar_feature_list.rename(columns={'Column': 'Old_Column_Unsorted_By_Variance'}, inplace=True)

#     #print("X_filtered for var shape: " + str(topvar_X.shape))
#     #print("feature_list_filtered for var shape: " + str(topvar_feature_list.shape))

#     return topvar_X, topvar_feature_list, topvar_indices


# # Get the accuracy, mse, and pearson correlation of the predictions
# # INPUT:
# #   Y_pred (Pytorch tensor) - predicted synergy scores
# #   Y_true (Pytorch tensor) - true synergy scores
# #   binary_classification (bool) - whether or not to use binary classification accuracy
# #   syn_threshold (float) - threshold for classifying syn
# #   ant_threshold (float) - threshold for classifying ant
# # OUTPUT:
# #   accuracy (float) - accuracy of the predictions
# #   mse (float) - mean squared error of the predictions
# #   pearson (float) - pearson correlation coefficient of the predictions
# def get_accuracy_metrics(
#         Y_pred,
#         Y_true,
#         binary_classification=False,
#         syn_threshold=10,
#         ant_threshold=-10,
#     ):

#     # Check if Y_pred and Y_true are the same shape
#     if Y_pred.shape != Y_true.shape:
#         # Try flattening the tensors
#         Y_pred = Y_pred.flatten()
#         Y_true = Y_true.flatten()
#         if Y_pred.shape != Y_true.shape:
#             # If still not the same shape, raise an error
#             print("Y_pred shape: {}".format(Y_pred.shape))
#             print("Y_true shape: {}".format(Y_true.shape))
#             raise ValueError("Y_pred and Y_true are not the same shape")

#     # Check if Y_pred and Y_true are the correct dtype
#     if Y_pred.dtype != torch.float64:
#         Y_pred = Y_pred.type(torch.float64)
#     if Y_true.dtype != torch.float64:
#         Y_true = Y_true.type(torch.float64)
    
#     # MSE is the sum of the squared differences between the predicted and true values
#     mean_squared_error = MeanSquaredError()
#     mse = mean_squared_error(Y_pred, Y_true)
   
#     # Pearson correlation coefficient is the covariance of the predicted and true values divided by the product of their standard deviations
#     pearson = PearsonCorrCoef()
#     corr = pearson(Y_pred, Y_true)

#     if binary_classification:
#         # assumes that the output has been forced through sigmoid so gives value between 0 to 1
#         # Scores are considered synergistic if > 0.5, and not if <= 0.5
#         Y_pred_synergistic = torch.where(Y_pred > 0.5, 1, 0)
#         Y_true_synergistic = torch.where(Y_true > 0.5, 1, 0)
#     else:
#         # If regression, scores are considered synergistic if > 10, antagonistic if < -10, and 
#         # additive if between -10 and 10
#         Y_pred_synergistic = torch.where(Y_pred >= syn_threshold, 1, torch.where(Y_pred <= ant_threshold, -1, 0))
#         Y_true_synergistic = torch.where(Y_true >= syn_threshold, 1, torch.where(Y_true <= ant_threshold, -1, 0))

#     # Accuracy is the number of correct predictions divided by the total number of values
#     correct_predictions = torch.sum(Y_pred_synergistic == Y_true_synergistic)
#     categorical_acc = correct_predictions / Y_true.shape[0]

#     return categorical_acc, mse, corr


