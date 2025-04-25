import argparse
from os.path import exists
import pandas as pd

# Read in full filtered (known) STRING file, protein symbols not converted yet
# INPUT:
#   None
# OUTPUT:
#   string_df (Pandas Dataframe) - the dataframe containing the full PPIN
def get_known_STRING_df():
    full_string_fp='data_processed/STRING_full_filtered.tsv'
    if exists(full_string_fp):
        string_df = pd.read_csv(full_string_fp, sep='\t')
        return string_df
    else:
        raise FileNotFoundError("File " + full_string_fp + " does not exist. Please run with --from_original first.")


# Read in full filtered (known) STRING file, protein symbols converted
# INPUT:
#   None
# OUTPUT:
#   string_df (Pandas Dataframe) - the dataframe containing the full PPIN
def get_known_STRING_protein_df():
    full_string_fp='data_processed/STRING_full_filtered_protsym.tsv'
    if exists(full_string_fp):
        string_df = pd.read_csv(full_string_fp, sep='\t')
        return string_df
    else:
        raise FileNotFoundError("File " + full_string_fp + " does not exist. Please run with --from_known or --from_original first.")


# Read in STRING proteins
# INPUT:
#   None
# OUTPUT:
#   string_protein_list (list) - list of proteins that are represented in the STRING PPIN
def get_string_protein_list():
    string_protein_fp='data_processed/STRING_full_filtered_protsym_unique_proteins.txt'
    if exists(string_protein_fp):
        # Read in line separated file of genes
        with open(string_protein_fp, 'r') as f:
            string_protein_list = f.read().splitlines()
        print("Number of STRING proteins: " + str(len(string_protein_list)))
        return string_protein_list
    else:
        raise FileNotFoundError("File " + string_protein_fp + " does not exist.")


# Filter STRING PPIN df to just include known interactions rather than predicted interactions
# INPUT:
#   df (Pandas Dataframe) - the dataframe containing the full PPIN
# OUTPUT:
#   df (Pandas Dataframe) - the dataframe containing PPIN without predicted interactions
def get_known_interactions(
    df,
):
    # Print the columns of the dataframe
    full_column_headers = df.columns
    print(full_column_headers)

    full_protected_columns = ['protein1', 'protein2', 'experimental', 'database', 'combined_score']

    print("Original full STRING shape: " + str(df.shape))
    # Remove any columns that don't belong to the 'experimentally_determined_interaction' or 'database_annotated' categories
    for column in full_column_headers:
        if column not in full_protected_columns:
            df.drop(column, axis=1, inplace=True)

    # Print the shape of the dataframe now
    print("Removed irrelevant column: " + str(df.shape))

    df.drop(df[(df.experimental == 0) & (df.database == 0)].index, inplace=True)

    # Print the shape of the dataframe now
    print("Known interactions df shape: " + str(df.shape))

    return df


# Filter dataframe to only use the interactions that include proteins in a certain list. Options
# for both the first and second protein in the interaction to be needed, or just one of them.
# INPUT:
#   df (Pandas Dataframe) - the dataframe containing the full PPIN
#   protein_list (list) - list of proteins to filter the dataframe to
#   both_proteins (bool) - whether to filter to interactions that include both proteins in the list
# OUTPUT:
#   df (Pandas Dataframe) - the dataframe containing PPIN without predicted interactions
def get_protein_subnetwork(
    df,
    protein_list,
    both_proteins=True,
):
    print("Original dataframe shape: " + str(df.shape))

    # Filters for protein 1 and protein 2
    protein1_filter = df['protein1'].isin(protein_list)
    protein2_filter = df['protein2'].isin(protein_list)

    # if both_proteins is true, then filter to only include interactions where both protein 1 and 2
    # are in the protein list, otherwise filter to include interactions where either protein 1 or 2
    # are in the protein list
    if both_proteins:
        df = df[protein1_filter & protein2_filter]
    else:
        df = df[protein1_filter | protein2_filter]
    
    # Print the shape of the dataframe now
    print("Protein subnetwork shape: " + str(df.shape))

    return df


# Get the protein symbols for the proteins in the dataframe
# INPUT:
#   None
# OUTPUT:
#   prot_id_to_symbol_dict (dict) - dictionary mapping protein id to protein symbol
def get_protein_id_to_symbol_dict():
    prot_id_to_symbol_dict = {}
    possible_dict_fp = 'data_processed/prot_id_to_symbol.csv'
    if exists(possible_dict_fp):
        with open(possible_dict_fp, 'r') as f:
            for line in f:
                line_split = line.split(',')
                prot_id_to_symbol_dict[line_split[0]] = line_split[1].strip()
        return prot_id_to_symbol_dict
    
    # If not already created, create the dictionary from original STRING file
    with open('data/STRING_homosapiens/9606.protein.info.v11.5.txt', 'r') as f:
        f.readline()
        for line in f:
            line_split = line.split('\t')
            prot_id_to_symbol_dict[line_split[0]] = line_split[1].strip()
    
    # Save the dictionary to a file
    with open(possible_dict_fp, 'w') as f:
        f.write('protein_id,protein_symbol\n')
        for key, value in prot_id_to_symbol_dict.items():
            f.write(key + ',' + value + '\n')

    return prot_id_to_symbol_dict


# Convert the string protein id to the protein symbol
# INPUT:
#   string_df (Pandas Dataframe) - the dataframe containing the unconverted PPIN
#   prot_id_to_symbol (dict) - dictionary mapping protein id to protein symbol
# OUTPUT:
#   string_df (Pandas Dataframe) - the dataframe containing the converted PPIN
def convert_string_to_protein_symbol(
    string_df,
    prot_id_to_symbol,
):
    # Replace the protein ids with the protein symbols
    string_df['protein1'] = string_df['protein1'].map(prot_id_to_symbol)
    string_df['protein2'] = string_df['protein2'].map(prot_id_to_symbol)



    return string_df


# Save the dataframe to a file using tab separation
# INPUT:
#   df (Pandas dataframe) - dataframe to save
#   file_path (str) - where to save the file
# OUTPUT:
#   None
def save_dataframe(
    df,
    file_path,
):
    df.to_csv(file_path, sep='\t', index=False)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_original', action="store_true", help='Should start from original file')
    parser.add_argument('--use_protsym', action="store_true", help='Should use protein symbols when generating from original file')
    parser.add_argument('--from_known', action="store_true", help='Should start from known interactions file, protein symbols not converted yet')
    parser.add_argument('--from_known_proteins', action="store_true", help='Should start from known interactions file, protein symbols converted')
    parser.add_argument('--dna_mutation', action="store_true", help='Should use DNA mutation data')
    parser.add_argument('--rna_expression', action="store_true", help='Should use RNA expression data')
    parser.add_argument('--protein_expression', action="store_true", help='Should use protein expression data')
    args = parser.parse_args()
    
    full_string_df = None
    prot_id_to_symbol = get_protein_id_to_symbol_dict()
    output_file_prefix = 'data_processed/STRING_full_filtered'
    output_file_prostym_suffix = '_protsym'
    # Keep track of which data modalities are being used to filter the PPIN
    of_filter = ''
    of_suffix = '.tsv'

    if args.from_original:
        full_string_detailed_file='data/STRING_homosapiens/9606.protein.links.detailed.v11.5.txt'
        full_string_detailed_df = pd.read_csv(full_string_detailed_file, sep=' ')
        print("Read the CSV")
        known_interactions_df = get_known_interactions(full_string_detailed_df)
        print("Got known interactions")
        if args.use_protsym:
            full_string_df = convert_string_to_protein_symbol(known_interactions_df, prot_id_to_symbol)
            output_file_prefix += output_file_prostym_suffix
        full_string_detailed_df.to_csv(output_file_prefix + of_suffix, sep='\t', index=False)
    elif args.from_known:
        known_interactions_df = get_known_STRING_df()
        full_string_df = convert_string_to_protein_symbol(known_interactions_df, prot_id_to_symbol)
    elif args.from_known_proteins:
        full_string_df = get_known_STRING_protein_df()
    else:
        raise NotImplementedError("Please specify --from_original or --from_known or --from_known_proteins")


    # Print the first 5 rows of the dataframe
    print(full_string_df.head())

    # Save the dataframe to a new file
    save_dataframe(full_string_df, output_file_prefix + of_filter + of_suffix)

    # Save the unique proteins to a file
    unique_protein1 = full_string_df['protein1'].unique()
    unique_protein2 = full_string_df['protein2'].unique()
    unique_proteins = sorted(set(unique_protein1).union(set(unique_protein2)))
    print("Number of unique proteins for this run: " + str(len(unique_proteins)))
    
    with open(output_file_prefix + of_filter + '_unique_proteins.txt', 'w') as f:
        for protein in unique_proteins:
            f.write(protein + '\n')


   