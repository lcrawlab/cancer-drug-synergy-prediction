from dataset_creation.datasets import *
from dataset_creation.getProcessedData import *
from models.src.modelPCNNGL import *
from sklearn.metrics import *
import argparse
import os
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Create PCNNGLSyndeepModel is a subclass of PCNNGLModel for handling the partially connected cell
# feature layers. This model structure will handle different -omics types by connecting them all to 
# a central gene layer, having additional layers that follows the SYNDEEP model, and then
# connecting that to the output layer
# INPUT:
#   input_size: int - the number of features in the input
#   hidden_sizes: list of ints - the number of nodes in each of the hidden layers
#   mask: torch.Tensor - the mask to enforce partial connections from input features to the gene layer
#   num_epochs: int - the number of epochs to train the model
#   learning_rate: float - the learning rate for the optimizer
#   loss_fxn: function - the loss function to use
#   device: torch.device - the device to train the model on
#   binary_classification: bool - whether the model is for binary classification or not
#   dropout_rate: float - the dropout rate for the model, if 0 no dropout is used
# OUTPUT:
#   model: nn.Module - PCNNGLSyndeep model
class PCNNGLSyndeepModel(PCNNGLModel):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        mask,
        num_epochs,
        learning_rate,
        loss_fxn,
        binary_classification=False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dropout_rate=0.0,
    ):
        super().__init__(input_size, hidden_sizes[0], mask, num_epochs, learning_rate, loss_fxn, binary_classification)

        # Initialize extra parameters not covered by PCNNGLModel
        self.hidden_sizes = hidden_sizes
        self.use_dropout = dropout_rate > 0.0

        if len(hidden_sizes) == 0:
            raise ValueError('Must have at least one hidden layer, corresponding to the gene layer')
        gene_layer_size = hidden_sizes[0]


        # Initialize layers, hidden size is the number of genes for the gene layer
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.batch_norms.append(nn.BatchNorm1d(self.input_size, device=device, dtype=torch.float32))
        for i in range(len(self.hidden_sizes)):
            if i == len(self.hidden_sizes) - 1:
                self.batch_norms.append(nn.BatchNorm1d(self.hidden_sizes[i], device=device, dtype=torch.float32))
                self.output = nn.Linear(self.hidden_sizes[i], 1, device=device, dtype=torch.float32)
            else:
                hidden_layer = nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1], device=device, dtype=torch.float32)
                batch_norm_layer = nn.BatchNorm1d(self.hidden_sizes[i], device=device, dtype=torch.float32)
                self.hidden_layers.append(hidden_layer)
                self.batch_norms.append(batch_norm_layer)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Move model to device
        self.to(device)
        
    # Forward pass through the model, overwriting the forward pass in PCNNGLModel
    # INPUT:
    #   x: torch.Tensor - the input data
    # OUTPUT:
    #   out: torch.Tensor - the output data
    def forward(self, x):
        x = x.to(self.device)
        out = self.batch_norms[0](x)
        out = self.input(x)
        with torch.no_grad():
            self.input.weight = nn.Parameter(self.input.weight * self.mask) # 0 out connections not in mask again

        for i in range(len(self.hidden_layers) + 1): # +1 to include output layer
            out = self.ReLU(out)
            out = self.batch_norms[i + 1](out)
            if self.use_dropout:
                out = self.dropout(out)

            if i == len(self.hidden_layers):
                out = self.output(out)
            else:
                out = self.hidden_layers[i](out)
        
        if self.binary_classification:
            out = self.sigmoid(out)

        return out


# Take in a job array index and return the corresponding hyperparameters. Modify this to run 
# different hyperparameter combinations, hidden layer variations for PCNNGLSyndeep model
# INPUT:
#   job_index: int - the job array index (1 indexed)
#   device: torch.device - the device to train the model on
# OUTPUT:
#   hyperparameters: dict - the hyperparameters for the model
def get_PCNNGLSyndeep_hyperparameters(job_index):
    num_epochs = [300]
    add_hidden_layers = [[722, 128, 32]]
    batch_sizes = [128]
    learning_rates = [2e-4]
    loss_fxn = ['MSELoss']
    optimizer = ['Adam']
    dropout_rates = [0.8]
    hyperparameter_array = []
    for hl in add_hidden_layers:
        for ne in num_epochs:
            for bs in batch_sizes:
                for lr in learning_rates:
                    for lf in loss_fxn:
                        for opt in optimizer:
                            for dr in dropout_rates:
                                hyperparam_dict = {
                                    'hidden_sizes': hl,
                                    'num_epochs': ne,
                                    'batch_size': bs,
                                    'learning_rate': lr,
                                    'loss_fxn': lf,
                                    'optimizer': opt,
                                    'dropout_rate': dr,
                                }
                                hyperparameter_array.append(hyperparam_dict)
    return hyperparameter_array[job_index - 1]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_fp', type=str, help='Output file path')
    parser.add_argument('--job_index', type=int, default=0, help='Job array index (1 indexed)')
    parser.add_argument('--checkpoint_weights', type=str, default=None, help='File containing checkpoint weights')
    parser.add_argument('--loss_file', type=str, default=None, help='File containing loss over time till checkpoint')
    parser.add_argument('--use_mfp', action='store_true', help='Use morgan fingerprint data')
    parser.add_argument('--use_dna', action='store_true', help='Include DNA data')
    parser.add_argument('--use_rna', action='store_true', help='Include RNA data')
    parser.add_argument('--use_prot', action='store_true', help='Include protein data')
    parser.add_argument('--use_bc', action='store_true', help='Use binary comboscore for prediction task')
    parser.add_argument('--use_csreg', action='store_true', help='Use regression on comboscore for prediction task')
    parser.add_argument('--use_pgreg', action='store_true', help='Use regression on percentage growth for prediction task')
    parser.add_argument('--tissue', type=str, default='all_cancer', help='Tissue type to use')
    parser.add_argument('--drug_class', type=str, default='all_drugs', help='Drug pair classes to filter the dataset by')
    parser.add_argument('--mfp_len', type=int, default=256, help='Length of Morgan fingerprint')
    parser.add_argument('--bc_cutoff', type=int, default=0, help='Cutoff for binary comboscore')
    parser.add_argument('--folds', type=int, default=10, help='Number of folds for cross validation')
    args = parser.parse_args()

    # Error check
    if args.use_mfp and args.mfp_len == 0:
        raise ValueError('Must specify mfp length if using mfp')
    if not (args.use_mfp or args.use_dna or args.use_rna or args.use_prot):
        raise ValueError('Must use at least one type of data')
    if not (args.use_bc or args.use_csreg or args.use_pgreg):
        raise ValueError('Must use at least one prediction task')
    if args.use_bc and args.use_csreg:
        raise ValueError('Cannot use both bc and csreg')
    if args.use_bc and args.use_pgreg:
        raise ValueError('Cannot use both bc and pgreg')
    if args.use_csreg and args.use_pgreg:
        raise ValueError('Cannot use both csreg and pgreg')
    if args.tissue != 'all_cancer' and args.drug_class != 'all_drugs':
        raise ValueError('Cannot use both tissue and drug class filtering')
    if args.tissue != 'all_cancer':
        valid_cancer_types = ['breast', 'cns', 'colon', 'leukemia', 'melanoma', 'nsclc', 'ovarian', 'prostate', 'renal']
        if args.tissue not in valid_cancer_types:
            raise ValueError(f'tissue should be one of {valid_cancer_types}')
    if args.drug_class != 'all_drugs':
        valid_drug_classes = ['chemo_chemo', 'chemo_targeted', 'chemo_other', 'targeted_targeted', 'targeted_other', 'other_other']
        if args.drug_class not in valid_drug_classes:
            raise ValueError(f'drug_class should be one of {valid_drug_classes}')
    
    # Device to train the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Get the model hyperparameters
    params = get_PCNNGLSyndeep_hyperparameters(args.job_index)
    print("Hyperparameters: ", params)

    lf = None
    if params["loss_fxn"] == "MSELoss":
        lf = nn.MSELoss()
    else:
        raise Exception("Error: Unsupported loss function, only MSE implemented")
    if params["optimizer"] != 'Adam':
        raise Exception("Error: optimizer not supported, only Adam implemented")

    # Get the data
    print("Loading data...")
    data_fn = get_all_cancer_dataset_filename(args.use_mfp, args.use_dna, args.use_rna, args.use_prot, args.use_bc, args.use_csreg, args.use_pgreg, args.mfp_len, args.bc_cutoff)
    filter_indices_fn = None
    if args.tissue != 'all_cancer':
        filter_indices_fn = get_cancer_type_indices_filename(args.tissue, args.use_bc, args.use_csreg, args.use_pgreg)
    elif args.drug_class != 'all_drugs':
        filter_indices_fn = get_drug_class_indices_filename(args.drug_class, args.use_bc, args.use_csreg, args.use_pgreg)
    data = MorganFingerprintDataset(filename=data_fn, balance_classes=args.use_bc, indices_filter_fn=filter_indices_fn)
    

    # Get the mask for the model
    mask_fp = args.output_fp + str(args.job_index) + 'mask.csv'
    mask = get_mask1gl(args.use_mfp, args.use_dna, args.use_rna, args.use_prot, mask_fp, args.use_bc or args.use_csreg, args.use_pgreg, args.mfp_len) # Should be number of features by number of genes
    hidden_layer_size = mask.shape[0]

    # K fold cross validation
    kf = KFold(n_splits=args.folds, shuffle=True)
    all_fold_metrics=pd.DataFrame()
    if args.use_bc:
        all_fold_metrics = pd.DataFrame(columns=['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'MCC', 'AUC', 'Kappa'])
    elif args.use_csreg or args.use_pgreg:
        all_fold_metrics = pd.DataFrame(columns=['MSE', 'RMSE', 'MAE', 'R2', 'Pearson', 'Spearman'])

    for fold_index, (train_tune_index, test_index) in enumerate(kf.split(data.x)):
        print(f"Fold {fold_index+1}")

        train_index, tune_index = train_test_split(train_tune_index, test_size=0.1)
        train_subset = Subset(data, train_index)
        train_dataloader = DataLoader(train_subset, batch_size=128, shuffle=True)
        tune_subset = Subset(data, tune_index)
        tune_dataloader = DataLoader(tune_subset, batch_size=128, shuffle=True)
        test_subset = Subset(data, test_index)
        test_dataloader = DataLoader(test_subset, batch_size=128, shuffle=True)

        input_size = train_dataloader.dataset[0][0].shape[0]

        # Create the model
        pcnn_gene_layer_model = PCNNGLSyndeepModel(
            input_size = input_size,
            mask = mask,
            hidden_sizes = params['hidden_sizes'],
            num_epochs = params["num_epochs"],
            learning_rate = params["learning_rate"],
            loss_fxn = lf,
            device = device,
            binary_classification = args.use_bc,
            dropout_rate = params["dropout_rate"],
        )
        pcnn_gene_layer_model = pcnn_gene_layer_model.to(device)
        optimizer = optim.Adam(pcnn_gene_layer_model.parameters(), lr=params["learning_rate"])

        hidden_layers_str = ''
        for hl_idx in range(len(params["hidden_sizes"])):
            hidden_layers_str += str(params["hidden_sizes"][hl_idx]) + ','

        # Check which device pcnn_gene_layer_model is on
        print("pcnn_gene_layer_model device: " + str(next(pcnn_gene_layer_model.parameters()).device))
        
        output_file_prefix = args.output_fp +'hl' + hidden_layers_str + '_ne' + str(params["num_epochs"]) \
            + '_bs'  + str(params["batch_size"]) + '_lr' + str(params["learning_rate"]) + '_lf' \
            + params["loss_fxn"] + '_opt' + params["optimizer"] + '_kf' + str(fold_index+1) + '/'
        
        print(output_file_prefix)
        
        if not os.path.exists(output_file_prefix):
            os.makedirs(output_file_prefix)
        
        start_time = time.time()
        # Train the model
        train_cov_model(
            pcnn_gene_layer_model,
            train_dataloader,
            optimizer,
            output_file_prefix,
            tune_dataloader=tune_dataloader,
            checkpoint_weights=args.checkpoint_weights,
            loss_file=args.loss_file,
            epsilon=-1e-4,
        )
        # Print last training loss
        print("Last training loss: " + str(pcnn_gene_layer_model.train_loss_over_time[-1]))

        train_end_time = time.time()

        print_model_params(pcnn_gene_layer_model)

        # Test the model
        fold_metrics = test_cov_model(pcnn_gene_layer_model, test_dataloader, output_file_prefix)
        all_fold_metrics.loc[fold_index] = fold_metrics
        print(fold_metrics)

        # Print time it took for training and testing
        print("Train time: " + str(train_end_time - start_time))
    
    all_fold_metrics.to_csv(args.output_fp + 'all_fold_metrics.csv')
    print(all_fold_metrics.mean())


