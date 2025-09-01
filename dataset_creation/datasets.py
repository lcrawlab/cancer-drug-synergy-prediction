import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

# Dataset class loading data from h5 file based on a dataframe:
# INPUT:
#   h5_path: str - path to the h5 file with the Morgan fingerprints, DNA, RNA, and protein features
#   data_path: str - path to the csv file with the data
#   target_column: str - the column name for the target variable (e.g. 'COMBOSCORE', 'PERCENTGROWTH', 'ZIP', or 'HSA')
#   binary_classification: bool - whether to convert the target variable to binary
#   balance_classes: bool - whether to balance the classes for binary classification
#   cancer_type: str - the cancer type to filter the dataset by (e.g. 'breast', 'cns', 'colon', 'leukemia', 'melanoma', 'nsclc', 'ovarian', 'prostate', 'renal')
#   drug_class: str - the drug class to filter the dataset by (e.g. 'chemo_chemo', 'chemo_targeted', 'chemo_other', 'targeted_targeted', 'targeted_other', 'other_other')
#   use_mfp: bool - whether to use Morgan fingerprints
#   use_dna: bool - whether to use DNA SNP data
#   use_rna: bool - whether to use RNA expression data
#   use_prot: bool - whether to use protein expression data
# OUTPUT:
#   Dataset class with the X and y torch tensors
class H5Dataset(Dataset):

    def __init__(self, h5_path, data_path, target_column, binary_classification=False, balance_classes=False,
                 cancer_type='all_cancer', drug_class='all_drugs',  # Optional parameters for filtering
                 use_mfp=True, use_dna=True, use_rna=True, use_prot=True, device='cpu'):
        if not binary_classification and balance_classes:
            raise ValueError('Cannot balance classes if not binary classification')
        if not (use_mfp or use_dna or use_rna or use_prot):
            raise ValueError('Must use at least one omics layer')
        if target_column not in ['COMBOSCORE', 'PERCENTGROWTH', 'ZIP', 'HSA']:
            raise ValueError('target_column should be one of COMBOSCORE, PERCENTGROWTH, ZIP, or HSA')
        if target_column == 'PERCENTGROWTH' and binary_classification:
            raise ValueError('Cannot use binary classification on PERCENTGROWTH')
        if cancer_type not in ['all_cancer', 'breast', 'cns', 'colon', 'leukemia', 'melanoma', 'nsclc', 'ovarian', 'prostate', 'renal']:
            raise ValueError('cancer_type should be one of all_cancer, breast, cns, colon, leukemia, melanoma, nsclc, ovarian, prostate, renal')
        if drug_class not in ['all_drugs', 'chemo_chemo', 'chemo_targeted', 'chemo_other', 'targeted_targeted', 'targeted_other', 'other_other']:
            raise ValueError('drug_class should be one of all_drugs, chemo_chemo, chemo_targeted, chemo_other, targeted_targeted, targeted_other, other_other')
        if cancer_type != 'all_cancer' and drug_class != 'all_drugs':
            raise ValueError('Can only filter by one of cancer_type or drug_class')
        if device not in ['cpu', 'cuda']:
            raise ValueError("device should be 'cpu' or 'cuda'")
        
        self.h5_path = h5_path
        self.data_path = data_path
        self.target_column = target_column
        self.binary_classification = binary_classification
        self.balance_classes = balance_classes
        self.cancer_type = cancer_type
        self.cancer_type_indices = None
        self.drug_class = drug_class
        self.drug_class_indices = None
        self.use_mfp = use_mfp
        self.use_dna = use_dna
        self.use_rna = use_rna
        self.use_prot = use_prot
        self.device = device

        # Get the indices for the cancer type and drug class
        if self.cancer_type != 'all_cancer' or self.drug_class != 'all_drugs':
            self._get_subset_indices()

        # Load the mappings from the H5 file
        self._load_mapping()

        # Process the dataframe to create the feature matrix
        self._process_dataframe()

        self.n_samples = len(self.x)
        self.num_features = self._calculate_feature_dimensions()

    def _get_subset_indices(self):
        # Get the indices for the cancer type
        cancer_type_indices_fp = 'data/ASP_dataset_slices/'+ self.cancer_type
        if self.target_column == 'PERCENTGROWTH':
            cancer_type_indices_fp += '_pg_indices.txt'
        else:
            cancer_type_indices_fp += '_comboscore_indices.txt'
        
        drug_class_indices_fp = 'data/ASP_dataset_slices/all_cancer_' + self.drug_class
        if self.target_column == 'PERCENTGROWTH':
            drug_class_indices_fp += '_pg_indices.txt'
        else:
            drug_class_indices_fp += '_cs_indices.txt'

        if self.cancer_type != 'all_cancer':
            with open(cancer_type_indices_fp, 'r') as f:
                self.cancer_type_indices = [int(line.rstrip('\n')) for line in f]
        elif self.drug_class != 'all_drugs':
            with open(drug_class_indices_fp, 'r') as f:
                self.drug_class_indices = [int(line.rstrip('\n')) for line in f]

    def _calculate_feature_dimensions(self):
        """Calculate total feature dimensions to pre-allocate arrays"""
        total_features = 0
        
        if self.use_mfp:
            total_features += 512  # 256 * 2 drugs
        if self.use_dna:
            total_features += 1168
        if self.use_rna:
            total_features += 786
        if self.use_prot:
            total_features += 786
        
        # Note: Concentration features only apply to PERCENTGROWTH
        if self.target_column == 'PERCENTGROWTH':
            total_features += 2     # CONC1, CONC2

        self.num_features = total_features
        print(f"Total feature dimensions: {total_features}")
            
        return total_features
    
    def _load_mapping(self):
        # Load the index mapping from the H5 file
        with h5py.File(self.h5_path, 'r') as f:
            nscs = f['drug_mfp']['nscs'][:]
            nscs = [int(nsc.decode('utf-8')) for nsc in nscs]
            self.nsc_to_idx = {nsc: i for i, nsc in enumerate(nscs)}

            self.cell_lines_dna = [cl.decode('utf-8') for cl in f['dna']['cell_lines'][:]]
            self.cell_lines_rna = [cl.decode('utf-8') for cl in f['rna']['cell_lines'][:]]
            self.cell_lines_prot = [cl.decode('utf-8') for cl in f['protein']['cell_lines'][:]]

            self.cl_to_dna_idx = {cl: i for i, cl in enumerate(self.cell_lines_dna)}
            self.cl_to_rna_idx = {cl: i for i, cl in enumerate(self.cell_lines_rna)}
            self.cl_to_prot_idx = {cl: i for i, cl in enumerate(self.cell_lines_prot)}

    def _process_dataframe(self):
        # Process the dataframe to create the feature matrix and targets

        # Load the data from the CSV file
        # Should be 'data/ASP_dataset_slices/drug_comboscore_hsa_zip.csv'
        # or 'data/ASP_dataset_slices/drug_PERCENTGROWTH.csv'
        data_df = pd.read_csv(self.data_path)
        # Check if the target column is in the dataframe
        if self.target_column not in data_df.columns:
            raise ValueError(f'{self.target_column} not in dataframe columns')
        # Filter the dataframe by the cancer type or drug class
        if self.cancer_type != 'all_cancer':
            if self.cancer_type_indices is None:
                raise ValueError('cancer_type_indices is None')
            data_df = data_df.iloc[self.cancer_type_indices].reset_index(drop=True)
        elif self.drug_class != 'all_drugs':
            if self.drug_class_indices is None:
                raise ValueError('drug_class_indices is None')
            data_df = data_df.iloc[self.drug_class_indices].reset_index(drop=True)
        # Check if the target column has NaN values
        if data_df[self.target_column].isnull().any():
            # Filter out rows with NaN in the target column and print a new shape
            print(f'Found NaN values in {self.target_column}, filtering them out')
            data_df = data_df.dropna(axis='index', subset=[self.target_column]).reset_index(drop=True)
        # Check if the dataframe is empty
        if data_df.empty:
            raise ValueError('Dataframe is empty after filtering by cancer type, drug class, or NaN values')
        
        n_samples_og = len(data_df)
        y = np.zeros(n_samples_og, dtype=np.float32)

        # Extract targets
        y[:] = data_df[self.target_column].values

        # Check if need to convert to binary classification and balance classes
        if self.binary_classification:
            if self.target_column == 'PERCENTGROWTH':
                raise ValueError('Did not use PERCENTGROWTH for binary classification')
            y = (y > 0).astype(np.float32)
            if self.balance_classes:
                print('Initial dataset size:', n_samples_og)
                print('Balancing classes')
                # check if the dataset in y are only 0s and 1s
                assert np.all(np.logical_or(y == 0, y == 1)), 'Labels should be binary if balancing classes'
                positive_indices = [i for i, label in enumerate(y) if label == 1]
                negative_indices = [i for i, label in enumerate(y) if label == 0]
                n_positive = len(positive_indices)
                n_negative = len(negative_indices)
                if n_positive < n_negative:
                    negative_indices = np.random.choice(negative_indices, n_positive, replace=False)
                else:
                    positive_indices = np.random.choice(positive_indices, n_negative, replace=False)
                indices = np.concatenate([positive_indices, negative_indices]) # Just make sure to shuffle this later
                data_df = data_df.iloc[indices].reset_index(drop=True)
                y = y[indices]
        
        n_samples = len(data_df)
        total_features = self._calculate_feature_dimensions()
        X = np.zeros((n_samples, total_features), dtype=np.float32)
        
        with h5py.File(self.h5_path, 'r') as f:
            # Load data from H5 file
            mfp_data = f['drug_mfp']['mfp'][:] if self.use_mfp else None
            dna_data = f['dna']['features'][:] if self.use_dna else None
            rna_data = f['rna']['features'][:] if self.use_rna else None
            prot_data = f['protein']['features'][:] if self.use_prot else None

        # Get indices for vectorized access
        drug1_indices = data_df['NSC1'].map(self.nsc_to_idx).values
        drug2_indices = data_df['NSC2'].map(self.nsc_to_idx).values
        cell_dna_indices = data_df['CELLNAME'].map(self.cl_to_dna_idx).values
        cell_rna_indices = data_df['CELLNAME'].map(self.cl_to_rna_idx).values
        cell_prot_indices = data_df['CELLNAME'].map(self.cl_to_prot_idx).values
        
        # Fill features using vectorized operations
        feature_idx = 0
        
        if self.use_mfp:
            # Drug 1 MFP
            X[:, feature_idx:feature_idx+256] = mfp_data[drug1_indices]
            feature_idx += 256
            
            # Drug 1 concentration (if PERCENTGROWTH)
            if self.target_column == 'PERCENTGROWTH':
                X[:, feature_idx] = data_df['CONC1'].values
                feature_idx += 1
            
            # Drug 2 MFP
            X[:, feature_idx:feature_idx+256] = mfp_data[drug2_indices]
            feature_idx += 256
            
            # Drug 2 concentration (if PERCENTGROWTH)
            if self.target_column == 'PERCENTGROWTH':
                X[:, feature_idx] = data_df['CONC2'].values
                feature_idx += 1
        
        if self.use_dna:
            X[:, feature_idx:feature_idx+1168] = dna_data[cell_dna_indices]
            feature_idx += 1168
        
        if self.use_rna:
            X[:, feature_idx:feature_idx+786] = rna_data[cell_rna_indices]
            feature_idx += 786
        
        if self.use_prot:
            X[:, feature_idx:feature_idx+786] = prot_data[cell_prot_indices]
            feature_idx += 786
        
        # Convert to torch tensors
        self.x = torch.from_numpy(X).to(self.device)
        self.y = torch.from_numpy(y).unsqueeze(1).to(self.device)

        print(f"Created feature matrix of shape {self.x.shape}")

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


# Deprecated version of dataset creation
'''
class MorganFingerprintDataset(Dataset):

    def __init__(self, filename, balance_classes=False, indices_filter_fn=None):
        # data loading
        xy = np.loadtxt(filename, delimiter=',', dtype=np.float32)
        print("Original Dataset shape:", xy.shape)

        # Check if the indices_filter_fn, if provided, exists
        if indices_filter_fn is not None:
            try:
                with open(indices_filter_fn, 'r') as f:
                    print(f'Reading indices from {indices_filter_fn}')
                    ct_indices = [int(line.rstrip('\n')) for line in f]
                    xy = xy[ct_indices]
                    print("Filtered Dataset shape:", xy.shape)
            except FileNotFoundError:
                raise FileNotFoundError(f'{indices_filter_fn} not found')
        
        self.x = torch.from_numpy(xy[:, :-1])
        self.y = torch.from_numpy(xy[:, [-1]]) # n_samples, 1

        # balance classes
        if balance_classes:
            print('Initial dataset size:', self.x.shape[0])
            print('Balancing classes')
            # check if the dataset in y are only 0s and 1s
            assert torch.all(torch.logical_or(self.y == 0, self.y == 1)), 'Labels should be binary if balancing classes'
            positive_indices = [i for i, label in enumerate(self.y) if label == 1]
            negative_indices = [i for i, label in enumerate(self.y) if label == 0]
            n_positive = len(positive_indices)
            n_negative = len(negative_indices)
            if n_positive < n_negative:
                negative_indices = np.random.choice(negative_indices, n_positive, replace=False)
            else:
                positive_indices = np.random.choice(positive_indices, n_negative, replace=False)
            indices = np.concatenate([positive_indices, negative_indices])
            self.x = self.x[indices]
            self.y = self.y[indices]
            print('Balanced dataset size:', self.x.shape[0])

        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
'''


# Split data into train, tune, and test sets
# INPUT:
#   ds: torch.utils.data.Dataset
#   train_size: float, default=0.8
#   tune_size: float, default=0.1
#   test_size: float, default=0.1
#   random_state: int, default=42
# OUTPUT:
#   train_subset: torch.utils.data.Subset
#   tune_subset: torch.utils.data.Subset
#   test_subset: torch.utils.data.Subset
def split_data(ds, train_size=0.8, tune_size=0.1, test_size=0.1, random_state=42):
    if train_size + tune_size + test_size != 1.0:
        raise ValueError('train_size + tune_size + test_size should be 1.0')
    
    n_samples = len(ds)
    train_idx, tune_and_test_idx = train_test_split(range(n_samples), test_size=tune_size+test_size, train_size=train_size, random_state=random_state, shuffle=True)
    tune_idx, test_idx = train_test_split(tune_and_test_idx, test_size=test_size/(tune_size+test_size), train_size=tune_size/(tune_size+test_size), random_state=random_state, shuffle=True)
    train_subset = Subset(ds, train_idx)
    tune_subset = Subset(ds, tune_idx)
    test_subset = Subset(ds, test_idx)

    print('Train size:', len(train_subset))
    print('Tune size:', len(tune_subset))
    print('Test size:', len(test_subset))

    return train_subset, tune_subset, test_subset

# Deprecated version of getting the filename for the all cancer dataset
'''
# Return the filename for use for the all cancer dataset. If you want to get a different cancer type
# use the other method to retrieve the cancer type indices that you can filter the dataset from
# INPUT:
#   use_mfp: bool
#   use_dna: bool
#   use_rna: bool
#   use_prot: bool
#   use_bc: bool
#   use_csreg: bool
#   use_pgreg: bool
#   mfp_len: int, default=256
#   bc_cutoff: int, default=0
# OUTPUT:
#   fn_prefix: str
def get_all_cancer_dataset_filename(use_mfp, use_dna, use_rna, use_prot, use_bc, use_csreg, use_pgreg, mfp_len=256, bc_cutoff=0):
    fn_prefix = 'data/ASP_dataset_slices/all_cancer' + '_'
    if use_mfp:
        fn_prefix += str(mfp_len) + '_mfp'
    if use_dna:
        fn_prefix += 'dna'
    if use_rna:
        fn_prefix += 'rna'
    if use_prot:
        fn_prefix += 'prot'
    if use_bc:
        fn_prefix += '_bc' + str(bc_cutoff) + '_comboscore.csv'
    if use_csreg:
        fn_prefix += '_comboscore.csv'
    if use_pgreg:
        fn_prefix += '_percgrowth.csv'
    print("Using filename: ", fn_prefix)
    return fn_prefix
'''

# Unneeded function, built this into the H5Dataset class
'''
# Get the filename for the cancer type indices to filter the all cancer dataset by
# INPUT:
#   cancer_type: str
#   use_bc: bool
#   use_csreg: bool
#   use_pgreg: bool
# OUTPUT:
#   filename: str
def get_cancer_type_indices_filename(cancer_type, use_bc, use_csreg, use_pgreg):
    valid_cancer_types = ['breast', 'cns', 'colon', 'leukemia', 'melanoma', 'nsclc', 'ovarian', 'prostate', 'renal']
    if cancer_type not in valid_cancer_types:
        raise ValueError(f'cancer_type should be one of {valid_cancer_types}')
    
    fn_prefix = 'data/ASP_dataset_slices/' + cancer_type + '_'
    if (use_bc or use_csreg) and use_pgreg:
        raise ValueError('Cannot use both ComboScore and Percent Growth')
    elif not(use_bc or use_csreg or use_pgreg):
        raise ValueError('Must use either ComboScore or Percent Growth')
    if use_bc or use_csreg: # Binary Classification or ComboScore Regression both use the same samples (rows)
        fn_prefix += 'comboscore'
    if use_pgreg:
        fn_prefix += 'pg'
    return fn_prefix + '_indices.txt'


# Get the filename for the drug class indices to filter the all cancer dataset by
# INPUT:
#   drug_class: str
#   use_bc: bool
#   use_csreg: bool
#   use_pgreg: bool
# OUTPUT:
#   filename: str
def get_drug_class_indices_filename(drug_class, use_bc, use_csreg, use_pgreg):
    valid_drug_classes = ['chemo_chemo', 'chemo_targeted', 'chemo_other', 'targeted_targeted', 'targeted_other', 'other_other']
    if drug_class not in valid_drug_classes:
        raise ValueError(f'drug_class should be one of {valid_drug_classes}')
    
    fn_prefix = 'data/ASP_dataset_slices/all_cancer_' + drug_class + '_'
    if (use_bc or use_csreg) and use_pgreg:
        raise ValueError('Cannot use both ComboScore and Percent Growth')
    elif not(use_bc or use_csreg or use_pgreg):
        raise ValueError('Must use either ComboScore or Percent Growth')
    if use_bc or use_csreg: # Binary Classification or ComboScore Regression both use the same samples (rows)
        fn_prefix += 'cs'
    if use_pgreg:
        fn_prefix += 'pg'
    return fn_prefix + '_indices.txt'
'''

# Deprecated version of loading the dataloader
'''
# Load the data from the filename
# INPUT:
#   filename: str
#   balance_classes: bool, default=False
#   batch_size: int, default=128
#   indices_filter_fn: str, default=None
# OUTPUT:
#  train_dataloader: DataLoader
#  tune_dataloader: DataLoader
#  test_dataloader: DataLoader
def load_dataloader(filename, balance_classes=False, batch_size=128, indices_filter_fn=None):
    data = MorganFingerprintDataset(filename=filename, balance_classes=balance_classes, indices_filter_fn=indices_filter_fn)

    train_subset, tune_subset, test_subset = split_data(data, train_size=0.8, tune_size=0.1, test_size=0.1, random_state=42)

    # Create a dataloader for each subset
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    tune_dataloader = DataLoader(tune_subset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)

    # Print out the size of the dataloaders
    print(f"Train loader size: {len(train_dataloader)}")
    print(f"Tune loader size: {len(tune_dataloader)}")
    print(f"Test loader size: {len(test_dataloader)}")

    return train_dataloader, tune_dataloader, test_dataloader
'''

# Get the mask for the model. Note, do not need to take in tissue because mask is same for all
# tissue types as the columns are the same
# INPUT:
#   use_mfp: bool - whether to use morgan fingerprints
#   use_dna: bool - whether to use dna data
#   use_rna: bool - whether to use rna data
#   use_prot: bool - whether to use protein data
#   output_fp: str - the file path to save the mask to
#   use_cs: bool - whether to use combo score (without concentration features)
#   use_pg: bool - whether to use percent growth (with concentration features)
#   mfp_len: int, default=256 - the length of the morgan fingerprint
# OUTPUT:
#   mask: torch.Tensor, shape (num_genes, num_features) - to match input layer weights matrix
def get_mask1gl(use_mfp, use_dna, use_rna, use_prot, output_fp, use_cs, use_pg, mfp_len=256):
    fn_prefix = 'data/ASP_dataset_slices/'
    mask_suffix = '_mask1gl.csv'
    mask_fns_to_append = []
    if use_mfp:
        mfp_fn = fn_prefix + str(mfp_len) + '_mfp'
        if use_cs and not use_pg:
            mfp_fn += '_cs'
        elif use_pg and not use_cs:
            mfp_fn += '_pg'    
        else:
            raise ValueError('Cannot use both cs and pg or neither, choose whether to include conc')
        mfp_fn += mask_suffix
        mask_fns_to_append.append(mfp_fn)
    if use_dna:
        dna_fn = fn_prefix + 'dna' + mask_suffix
        mask_fns_to_append.append(dna_fn)
    if use_rna:
        rna_fn = fn_prefix + 'rna' + mask_suffix
        mask_fns_to_append.append(rna_fn)
    if use_prot:
        prot_fn = fn_prefix + 'prot' + mask_suffix
        mask_fns_to_append.append(prot_fn)
    
    if len(mask_fns_to_append) == 0:
        raise ValueError('Must use at least one type of data')
    
    mask = pd.DataFrame()
    for i, mask_fn in enumerate(mask_fns_to_append):
        if i == 0:
            mask = pd.read_csv(mask_fn, header=0, index_col=0)
        else:
            mask_to_append = pd.read_csv(mask_fn, header=0, index_col=0)
            mask = pd.concat([mask, mask_to_append], axis=0)

    # Transpose mask
    mask = mask.T

    # Save the mask to a file
    mask.to_csv(output_fp, header=True, index=True)

    mask = torch.tensor(mask.values, dtype=torch.float32)
    print(mask.shape)
    return mask


# Get the masks for the different omics layers depending on the data modalities chosen
# INPUT:
#   use_mfp: bool - whether to use morgan fingerprints
#   use_dna: bool - whether to use dna data
#   use_rna: bool - whether to use rna data
#   use_prot: bool - whether to use protein data
#   use_bc: bool - whether to use binary comboscore for prediction task
#   use_csreg: bool - whether to use regression on comboscore for prediction task
#   use_pgreg: bool - whether to use regression on percentage growth for prediction task
#   mfp_len: int - the length of the morgan fingerprint
# OUTPUT:
#   masks: list of torch.Tensor - the masks for the different omics layers
def get_masks_concat(use_mfp, use_dna, use_rna, use_prot, use_bc, use_csreg, use_pgreg, mfp_len=256):
    # Error check
    if (use_bc and use_csreg) or (use_bc and use_pgreg) or (use_csreg and use_pgreg):
        raise ValueError("Cannot use both two prediction tasks simultaneously")
    elif not (use_bc or use_csreg or use_pgreg):
        raise ValueError("Must use at least one prediction task")
    if not (use_mfp or use_dna or use_rna or use_prot):
        raise ValueError("Must use at least one omics layer")
    
    dir_path = 'data/ASP_dataset_slices/concat_masks/'
    ppin_prefix = 'ppin_'
    mfp_prefix = str(mfp_len)+'_mfp'
    dna_prefix = 'dna'
    rna_prefix = 'rna'
    prot_prefix = 'protein'
    cs_prefix = '_cs'
    pgreg_prefix = '_pg'
    mask_suffix = '_maskconcat.csv'

    # Get the pairs of (in_layer, out_layer) for the masks based on if use_mfp, use_dna, use_rna, use_prot
    masks = []
    in_layers = []
    out_layers = []
    usage = [1 if use_mfp else 0, 1 if use_dna else 0, 1 if use_rna else 0, 1 if use_prot else 0]
    omics = [mfp_prefix, dna_prefix, rna_prefix, prot_prefix]
    for index in range(len(omics)):
        if usage[index] == 1:
            if len(in_layers) >= 1:
                out_layers.append(omics[index])
            in_layers.append(omics[index])
    out_layers.append('output')

    for l in range(len(in_layers)):
        filename = dir_path
        ppin_combos = [('dna', 'rna'), ('dna', 'protein'), ('rna', 'protein')]
        if (in_layers[l], out_layers[l]) in ppin_combos:
            filename += ppin_prefix
            print("using ppin mask")
        filename += in_layers[l] + '_' + out_layers[l]

        if in_layers[l] == mfp_prefix:
            if use_bc or use_csreg:
                filename += cs_prefix
            elif use_pgreg:
                filename += pgreg_prefix
        filename += mask_suffix

        mask = torch.Tensor(pd.read_csv(filename, index_col=0).values)
        masks.append(mask)

    return masks

