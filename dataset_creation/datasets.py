import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

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

