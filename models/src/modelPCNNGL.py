from dataset_creation.datasets import *
from dataset_creation.getProcessedData import *
from prettytable import PrettyTable
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import *
from sklearn.model_selection import KFold, train_test_split
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Create PCNNGLModel for handling the partially connected cell feature layers
# This model structure will handle different -omics types by connecting them all to 
# a central gene layer and then connecting that to the output layer
# INPUT:
#   input_size: int - the number of features in the input
#   hidden_size: int - the number of nodes in hidden gene layer (how many genes)
#   mask: torch.Tensor - the mask to enforce partial connections from input features to the gene layer
#   num_epochs: int - the number of epochs to train the model
#   learning_rate: float - the learning rate for the optimizer
#   loss_fxn: function - the loss function to use
#   device: torch.device - the device to train the model on
#   binary_classification: bool - whether the model is for binary classification or not
# OUTPUT:
#   model: nn.Module - PCNNGLModel model
class PCNNGLModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        mask,
        num_epochs,
        learning_rate,
        loss_fxn,
        binary_classification=False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(PCNNGLModel, self).__init__()

        # Initialize parameters
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mask = mask.to(device)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.binary_classification = binary_classification
        self.train_loss_over_time = []
        self.tune_loss_over_time = []

        if mask.shape[1] != input_size and mask.shape[0] != hidden_size:
            raise ValueError('Mask shape must be hidden_size x input_size, got: ' + str(mask.shape) + ' when input_size is ' + str(input_size) + ' and hidden_size is ' + str(hidden_size))
        
        if binary_classification:
            print("Actually using BCE Loss")
            self.loss_fxn = nn.BCELoss()
            self.sigmoid=nn.Sigmoid()
            self.train_accuracy_over_time = []
            self.tune_accuracy_over_time = []
        else:
            self.loss_fxn = loss_fxn


        # Initialize layers, hidden size is the number of genes for the gene layer
        self.input = nn.Linear(self.input_size, self.hidden_size, device=device, dtype=torch.float32)
        with torch.no_grad():
            self.input.weight = nn.Parameter(self.input.weight * self.mask) # element wise multiplication
        self.output = nn.Linear(self.hidden_size, 1, device=device, dtype=torch.float32)
        self.batch_norms = nn.ModuleList()
        bn1 = nn.BatchNorm1d(self.input_size, device=device, dtype=torch.float32)
        self.batch_norms.append(bn1)
        bn2 = nn.BatchNorm1d(self.hidden_size, device=device, dtype=torch.float32)
        self.batch_norms.append(bn2)
        self.ReLU = nn.ReLU()

        # Move model to device
        self.to(device)
        
    # Forward pass through the model
    # INPUT:
    #   x: torch.Tensor - the input data
    # OUTPUT:
    #   out: torch.Tensor - the output data
    def forward(self, x):
        x = x.to(self.device)
        out = self.batch_norms[0](x)
        out = self.input(out)
        with torch.no_grad():
            self.input.weight = nn.Parameter(self.input.weight * self.mask) # 0 out connections not in mask again
        out = self.ReLU(out)
        out = self.batch_norms[1](out) # 1 to skip input layer
        out = self.output(out)
        
        if self.binary_classification:
            out = self.sigmoid(out)

        return out
    
    # Load weights of trained model from file
    # INPUT:
    #   model: nn.Module - the model to load weights into
    #   weights_file: str - the file to load weights from
    # OUTPUT:
    #   None
    def load_weights(self, weights_file, loss_file):
        self.load_state_dict(torch.load(weights_file))
        print("Weights loaded from file")
        self.train_loss_over_time = pickle.load(open(loss_file, 'rb'))
        print("Loss loaded from file")
    
    # Train the model
    # INPUT:
    #   train_dataloader: torch.utils.data.DataLoader - the training data
    #   tune_dataloader: torch.utils.data.DataLoader - the tuning data (optional)
    #   optimizer: torch.optim - the optimizer to use
    #   epochs: int - the number of epochs to train the model
    #   epsilon: float - if tuning loss does not improve by epsilon, stop training
    #   output_file_prefix: str - the prefix for the output files
    def fit(
        self,
        train_dataloader,
        optimizer,
        tune_dataloader=None,
        epochs=300,
        epsilon=-9999.0,
        output_file_prefix='',
    ):
        train_dir = output_file_prefix + '/training'
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        else:
            train_dir = train_dir + '_cont'
        checkpoint_dir = output_file_prefix + '/checkpoint'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checked_state_dict = False
        checked_batch_device = False
        
        for epoch in range(epochs):
            train_loss = 0.0
            tune_loss = 0.0
            if self.binary_classification:
                train_correct = 0
                train_total = 0
                tune_correct = 0
                tune_total = 0

            self.train()
            for x_batch, y_batch in train_dataloader:
                # if the batch size is 1, the model will throw an error, so just skip it
                if x_batch.size(0) == 1:
                    continue
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                if not checked_batch_device:
                    checked_batch_device = True
                y_pred = self.forward(x_batch)
                loss = self.loss_fxn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                if self.binary_classification:
                    train_total += y_batch.size(0)
                    train_correct += (torch.round(y_pred) == y_batch).sum().item() # Have to round the output to get 0 or 1

            epoch_train_loss = train_loss / len(train_dataloader)
            self.train_loss_over_time.append(epoch_train_loss)

            if self.binary_classification:
                epoch_train_acc = 100 * train_correct / train_total
                self.train_accuracy_over_time.append(epoch_train_acc)
            
            if tune_dataloader is not None:
                self.eval()
                with torch.no_grad():
                    for x_batch, y_batch in tune_dataloader:
                        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                        y_pred = self(x_batch)
                        loss = self.loss_fxn(y_pred, y_batch)

                        tune_loss += loss.item()
                        if self.binary_classification:
                            tune_total += y_batch.size(0)
                            tune_correct += (torch.round(y_pred) == y_batch).sum().item()
                epoch_tune_loss = tune_loss / len(tune_dataloader)
                self.tune_loss_over_time.append(epoch_tune_loss)

                if self.binary_classification:
                    epoch_tune_acc = 100 * tune_correct / tune_total
                    self.tune_accuracy_over_time.append(epoch_tune_acc)
           
                # Early stopping with tuning loss
                # If tuning loss does not improve by epsilon, stop training
                # if epoch > 50 and self.tune_loss_over_time[-2] - self.tune_loss_over_time[-1] < epsilon:
                #     torch.save(self.state_dict(), checkpoint_dir + '/wts_trep' + str(epoch) + '.pth')
                #     with open(checkpoint_dir + '/loss_trep' + str(epoch) + '.pkl', 'wb') as f:
                #         pickle.dump(self.train_loss_over_time, f)
                #     print(f'Early stopping at epoch {epoch}, Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_acc:.2f}%, Tune Loss: {epoch_tune_loss:.4f}, Tune Accuracy: {epoch_tune_acc:.2f}%')
                #     break

            if (epoch) % 50 == 0:
                # Save weights of trained model to output file
                torch.save(self.state_dict(), checkpoint_dir + '/wts_trep' + str(epoch) + '.pth')
                if not checked_state_dict:
                    # Print layer names and sizes from the model state dictionary
                    for name, param in self.state_dict().items():
                        print(f"Layer: {name}, Size: {param.size()}")
                    checked_state_dict = True
                
                with open(checkpoint_dir + '/loss_trep' + str(epoch) + '.pkl', 'wb') as f:
                    pickle.dump(self.train_loss_over_time, f)

            if (epoch) % 25 == 0:
                if self.binary_classification:
                    print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_acc:.2f}%')
                    if tune_dataloader is not None:
                        print(f'Epoch [{epoch}/{epochs}], Tune Loss: {epoch_tune_loss:.4f}, Tune Accuracy: {epoch_tune_acc:.2f}%')
                else:
                    print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_train_loss:.4f}')
                    if tune_dataloader is not None:
                        print(f'Epoch [{epoch}/{epochs}, Tune Loss: {epoch_tune_loss:.4f}')

    # Plot the loss of the model over time
    # INPUT:
    #   output_file: str - the file to save the plot to
    # OUTPUT:
    #   None
    def plot_loss(self, output_file=None):
        plt.close()
        # Plot training loss over time
        plt.plot(range(len(self.train_loss_over_time)), self.train_loss_over_time, label='Training Loss')
        plt.title('Training Loss over Time')
        if len(self.tune_loss_over_time) > 0:
            # Plot tuning loss over time
            plt.plot(range(len(self.tune_loss_over_time)), self.tune_loss_over_time, label='Tune Loss')
            plt.title('Training and Tuning Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
        
    # Plot the accuracy of the model over time training
    # INPUT:
    #   output_file: str - the file to save the plot to
    # OUTPUT:
    #   None
    def plot_accuracy(self, output_file=None):
        plt.close()
        if not self.binary_classification:
            raise Exception("Error: Cannot plot accuracy for regression model")
        # Plot training accuracy over time
        plt.plot(range(len(self.train_accuracy_over_time)), self.train_accuracy_over_time, label='Training Accuracy')
        plt.title('Training Accuracy over Time')
        if len(self.tune_accuracy_over_time) > 0:
            # Plot tuning accuracy over time
            plt.plot(range(len(self.tune_accuracy_over_time)), self.tune_accuracy_over_time, label='Tune Accuracy')
            plt.title('Training and Tuning Accuracy over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper right")
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()

    # Predict the output of the model
    # INPUT:
    #   X: torch.Tensor - the input data
    # OUTPUT:
    #   y_pred: torch.Tensor - the predicted output
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            y_pred = self(X)
            y_pred = torch.round(y_pred)
        return y_pred
    

# Find and print the number of parameters (trainable and non) of a model
# INPUT:
#   model: nn.Module - the model to find the number of trainable parameters of
# OUTPUT:
#   None
def print_model_params(model):
    table_to_print = PrettyTable(["Modules", "Trainable Parameters"])
    total_params = 0
    total_trainable_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        if not parameter.requires_grad:
            total_params += param
            continue
        table_to_print.add_row([name, param])
        total_params += param
        total_trainable_params += param
    table_to_print.add_row(["Total", total_trainable_params])
    print(table_to_print)
    print(f"Total number of all parameters: {total_params}")


# Train the model with the training data
# INPUT:
#   model: nn.Module - the model to be trained
#   train_dataloader: torch.utils.data.DataLoader - the training data
#   tune_dataloader: torch.utils.data.DataLoader - the tuning data (optional)
#   optimizer: torch.optim - the optimizer to use
#   output_file_prefix: str - the prefix for the output files
#   checkpoint_weights: str - the checkpoint file to load weights from
#   loss_file: str - the loss file to load loss from previous epochs from
# OUTPUT:
#   None
def train_cov_model(
        model,
        train_dataloader,
        optimizer,
        output_file_prefix,
        tune_dataloader=None,
        checkpoint_weights=None,
        loss_file=None,
        epsilon=None,
    ):
    print("Model is training")

    # If loading weights from a checkpoint, load the weights and train for remaining epochs
    if checkpoint_weights:
        if (checkpoint_weights is not None) and (loss_file is not None):
            model.load_weights(checkpoint_weights, loss_file)
            print("Training for " + str(model.num_epochs) + " more epochs")
        else:
            raise Exception("Error: checkpoint file, loss file and remaining epochs must be specified if loading weights")
        
    model.fit(
        train_dataloader=train_dataloader,
        tune_dataloader=tune_dataloader,
        optimizer=optimizer,
        epochs=model.num_epochs,
        epsilon=epsilon,
        output_file_prefix=output_file_prefix)
    
    model.plot_loss(output_file_prefix + '/training/loss.png')
    if model.binary_classification:
        model.plot_accuracy(output_file_prefix + '/training/accuracy.png')

    # Save weights of trained model to output file
    torch.save(model.state_dict(), output_file_prefix + '/training/wts_final.pth')
    
    return model

# Test the model with the test data
# INPUT:
#   model: nn.Module - the model to be tested
#   test_dataloader: torch.utils.data.DataLoader - the testing data
#   output_file_prefix: str - the prefix for the output files
# OUTPUT:
#   fold_metrics: list - the metrics for the model
def test_cov_model(model, test_dataloader, output_file_prefix):
    X_test = test_dataloader.dataset.dataset.x[test_dataloader.dataset.indices]
    y_test = test_dataloader.dataset.dataset.y[test_dataloader.dataset.indices]
    print("Model is testing")

    test_dir = output_file_prefix + '/testing'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    model.eval()
    with torch.no_grad():
        # Get Y_pred
        y_pred = model.predict(X_test)
        
        # Make sure Y_pred and Y_test are both on cpu for accuracy metrics and plotting
        y_pred = y_pred.cpu().detach().numpy() 
        y_test = y_test.cpu().detach().numpy()
        print(f"Y test shape: " + str(y_test.shape))
        print(f"Y pred shape: " + str(y_pred.shape))

        if model.binary_classification:
            accuracy = accuracy_score(y_test, y_pred)
            sensitivity = recall_score(y_test, y_pred)
            specificity = recall_score(y_test, y_pred, pos_label=0)
            precision = precision_score(y_test, y_pred)
            f1s = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)

            # print("Accuracy:", accuracy)
            # print("Sensitivity:", sensitivity)
            # print("Specificity:", specificity)
            # print("Precision:", precision)
            # print("F1 Score:", f1s)
            # print("MCC:", mcc)
            # print("AUC:", auc)
            # print("Kappa:", kappa)

            return [accuracy, sensitivity, specificity, precision, f1s, mcc, auc, kappa]
        else:
            y_pred = np.ndarray.flatten(y_pred)
            y_test = np.ndarray.flatten(y_test)
            print(f"Y_test flattened shape: {y_test.shape}")
            print(f"Y_pred flattened shape: {y_pred.shape}")
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            pearson = pearsonr(y_test, y_pred)[0]
            spearman = spearmanr(y_test, y_pred)[0]

            # print(f"MSE: {mse}")
            # print(f"RMSE: {rmse}")
            # print(f"MAE: {mae}")
            # print(f"R2: {r2}")
            # print(f"Pearson: {pearson}")
            # print(f"Spearman: {spearman}")

            return [mse, rmse, mae, r2, pearson, spearman]


# Take in a job array index and return the corresponding hyperparameters. Modify this to run 
# different hyperparameter combinations, no hidden layer variations for pcnngl model
# INPUT:
#   job_index: int - the job array index (1 indexed)
# OUTPUT:
#   hyperparameters: dict - the hyperparameters for the model
def get_pcnngl_hyperparameters(job_index):
    num_epochs = [300]
    batch_sizes = [128]
    learning_rates = [2e-4]
    loss_fxn = ['MSELoss']
    optimizer = ['Adam']
    hyperparameter_array = []
    for ne in num_epochs:
        for bs in batch_sizes:
            for lr in learning_rates:
                for lf in loss_fxn:
                    for opt in optimizer:
                        hyperparam_dict = {
                            'num_epochs': ne,
                            'batch_size': bs,
                            'learning_rate': lr,
                            'loss_fxn': lf,
                            'optimizer': opt,
                        }
                        hyperparameter_array.append(hyperparam_dict)
    return hyperparameter_array[job_index-1]


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
    parser.add_argument('--score', type=str, default='COMBOSCORE', help='Score to use for prediction task, should be one of COMBOSCORE, PERCENTGROWTH, ZIP, HSA')
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
    if args.score not in ['COMBOSCORE', 'PERCENTGROWTH', 'ZIP', 'HSA']:
        raise ValueError('Score should be one of COMBOSCORE, PERCENTGROWTH, ZIP, HSA')
    if args.score == 'PERCENTGROWTH' and not args.use_pgreg:
        raise ValueError('Must use percent growth regression if using PERCENTGROWTH score')
    if args.score == 'PERCENTGROWTH' and args.use_bc:
        raise ValueError('Cannot use binary classification if score is PERCENTGROWTH')
    if args.use_pgreg and args.score != 'PERCENTGROWTH':
        raise ValueError('Must use PERCENTGROWTH score if using percent growth regression')
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
    params = get_pcnngl_hyperparameters(args.job_index)
    print("Hyperparameters: ", params)

    lf = None
    if params["loss_fxn"] == "MSELoss":
        lf = nn.MSELoss()
    else:
        raise Exception("Error: Unsupported loss function, only MSE implemented")
    if params["optimizer"] != 'Adam':
        raise Exception("Error: optimizer not supported, only Adam implemented")

    # Get the filename
    h5_path = 'data/ASP_dataset_slices/all_256mfpdnarnaprot.h5'
    non_pg_data_path = 'data/ASP_dataset_slices/drug_comboscore_hsa_zip.csv'
    pg_data_path = 'data/ASP_dataset_slices/drug_percent_growth.csv'
    plot_path = args.output_fp + '_'

    if args.use_pgreg:
        data_path = pg_data_path
    else:
        data_path = non_pg_data_path

    # Load the data
    data = H5Dataset(
        h5_path=h5_path,
        data_path=data_path,
        target_column=args.score,
        binary_classification=args.use_bc,
        balance_classes=args.use_bc,
        cancer_type=args.tissue,
        drug_class=args.drug_class,
        use_mfp=args.use_mfp,
        use_dna=args.use_dna,
        use_rna=args.use_rna,
        use_prot=args.use_prot,
        device=device,
    )

    # Get the mask
    mask_fp = args.output_fp + str(args.job_index) + 'mask.csv'
    mask = get_mask1gl(args.use_mfp, args.use_dna, args.use_rna, args.use_prot, mask_fp, args.use_bc or args.use_csreg, args.use_pgreg, args.mfp_len) # Should be number of features by number of genes
    hidden_layer_size = mask.shape[0]

    # K fold cross validation
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42) # Set random_state for reproducibility
    all_fold_metrics = pd.DataFrame()
    if args.use_bc:
        all_fold_metrics = pd.DataFrame(columns=['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'MCC', 'AUC', 'Kappa'])
    elif args.use_csreg or args.use_pgreg:
        all_fold_metrics = pd.DataFrame(columns=['MSE', 'RMSE', 'MAE', 'R2', 'Pearson', 'Spearman'])

    for i, (train_tune_index, test_index) in enumerate(kf.split(data.x)):
        print(f"Fold {i+1}")

        train_index, tune_index = train_test_split(train_tune_index, test_size=0.1)
        train_subset = Subset(data, train_index)
        train_dataloader = DataLoader(train_subset, batch_size=128, shuffle=True)
        tune_subset = Subset(data, tune_index)
        tune_dataloader = DataLoader(tune_subset, batch_size=128, shuffle=True)
        test_subset = Subset(data, test_index)
        test_dataloader = DataLoader(test_subset, batch_size=128, shuffle=True)

        input_size = train_dataloader.dataset[0][0].shape[0]

        # Create the model
        pcnn_gene_layer_model = PCNNGLModel(
            input_size = input_size,
            mask = mask,
            hidden_size = hidden_layer_size,
            num_epochs = params["num_epochs"],
            learning_rate = params["learning_rate"],
            loss_fxn = lf,
            device = device,
            binary_classification = args.use_bc,
        )

        pcnn_gene_layer_model = pcnn_gene_layer_model.to(device)
        optimizer = optim.Adam(pcnn_gene_layer_model.parameters(), lr=params["learning_rate"])

        # Check which device pcnn_gene_layer_model is on
        print("pcnn_gene_layer_model device: " + str(next(pcnn_gene_layer_model.parameters()).device))
        
        output_file_prefix = args.output_fp + 'hl' + str(hidden_layer_size) + '_ne' + str(params["num_epochs"]) \
            + '_bs'  + str(params["batch_size"]) + '_lr' + str(params["learning_rate"]) \
            + '_lf' + params["loss_fxn"] + '_opt' + params["optimizer"] + '_kf' + str(i+1) + '/'
        
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
        all_fold_metrics.loc[i] = fold_metrics

        # Print time it took for training and testing
        print("Fold train time: " + str(train_end_time - start_time))

    all_fold_metrics.to_csv(args.output_fp + 'all_fold_metrics.csv')
    print(all_fold_metrics.mean())

