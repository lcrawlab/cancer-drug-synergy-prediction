import argparse
import numpy as np
import pandas as pd
import scipy.stats
from datasets import *
from functools import partial
from models_baselines import *
from multiprocessing import Pool
from sklearn.model_selection import KFold
from sklearn.metrics import *


# Fit the snn model for binary classification
# INPUT:
#   train_dataloader: DataLoader
#   num_features: int
#   plot_path: str
#   tune_dataloader: DataLoader (optional)
# OUTPUT:
#   model: SNNModelBC
def fit_snn_bc_model(train_dataloader, num_features, plot_path, tune_dataloader=None):
    # Initialize and train the model
    model = SNNModelBC(
        input_size=num_features,
        hidden_size=256,
        dropout=0.8,
        learn_rate=0.0002,
    )
    model.fit(train_dataloader, tune_dataloader, epochs=300)
    print("Model trained")

    model.plot_loss(output_file=plot_path+'loss.png')
    print("Loss plotted")

    model.plot_accuracy(output_file=plot_path+'accuracy.png')
    print("Accuracy plotted")
    return model


# Evaluate the snn model for binary classification
# INPUT:
#   model: SNNModelBC
#   test_dataloader: DataLoader
# OUTPUT:
#   fold_metrics: list
def evaluate_snn_bc_model(model, test_dataloader):
    X_test = test_dataloader.dataset.dataset.x[test_dataloader.dataset.indices]
    y_test = test_dataloader.dataset.dataset.y[test_dataloader.dataset.indices]
    # Evaluation
    with torch.no_grad():
        y_pred = np.ndarray.flatten(model.predict(X_test).cpu().detach().numpy())
        y_test = np.ndarray.flatten(y_test.cpu().detach().numpy())
        print("Y_test flattened shape: {y_test.shape}")
        print("Y_pred flattened shape: {y_pred.shape}")
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred)
        specificity = recall_score(y_test, y_pred, pos_label=0)
        precision = precision_score(y_test, y_pred)
        f1s = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        #print("Accuracy:", accuracy)
        #print("Sensitivity:", sensitivity)
        #print("Specificity:", specificity)
        #print("Precision:", precision)
        #print("F1 Score:", f1s)
        #print("MCC:", mcc)
        #print("AUC:", auc)
        #print("Kappa:", kappa)

        return [accuracy, sensitivity, specificity, precision, f1s, mcc, auc, kappa]
        

# Fit the snn model for regression on comboscore or percent growth
# INPUT:
#   train_dataloader: DataLoader
#   tune_dataloader: DataLoader
#   num_features: int
#   plot_path: str
# OUTPUT:
#   model: SNNModelRegression
def fit_snn_reg_model(train_dataloader, num_features, plot_path, tune_dataloader=None):
    # Initialize and train the model
    model = SNNModelRegression(
        input_size=num_features,
        hidden_size=256,
        dropout=0.8,
        learn_rate=0.0002,
    )
    model.fit(train_dataloader, tune_dataloader, epochs=300)
    print("Model trained")

    model.plot_loss(output_file=plot_path+'loss.png')
    print("Loss plotted")
    return model


# Evaluate the snn model for regression on comboscore or percent growth
# INPUT:
#   model: SNNModelRegression
#   test_dataloader: DataLoader
# OUTPUT:
#   fold_metrics: list
def evaluate_snn_reg_model(model, test_dataloader):
    X_test = test_dataloader.dataset.dataset.x[test_dataloader.dataset.indices]
    y_test = test_dataloader.dataset.dataset.y[test_dataloader.dataset.indices]
    # Evaluation
    with torch.no_grad():
        y_pred = np.ndarray.flatten(model.predict(X_test).cpu().detach().numpy())
        y_test = np.ndarray.flatten(y_test.cpu().detach().numpy())
        print("Y_test flattened shape: {y_test.shape}")
        print("Y_pred flattened shape: {y_pred.shape}")
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        pearson = scipy.stats.pearsonr(y_test, y_pred)[0]
        spearman = scipy.stats.spearmanr(y_test, y_pred)[0]

        # print(f"MSE: {mse}")
        # print(f"RMSE: {rmse}")
        # print(f"MAE: {mae}")
        # print(f"R2: {r2}")
        # print(f"Pearson: {pearson}")
        # print(f"Spearman: {spearman}")

        return [mse, rmse, mae, r2, pearson, spearman]
    

# Run the snn model, one fold, so can be parallelized
# INPUT:
#   i: int (fold number)
#   train_index: np.ndarray
#   test_index: np.ndarray
#   data: MorganFingerprintDataset
#   args: argparse.ArgumentParser
# OUTPUT:
#   i: int
#   fold_metrics: list
def process_fold(i, train_index, test_index, data, args):
    print(f"Fold {i+1}")

    train_subset = Subset(data, train_index)
    train_dataloader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_subset = Subset(data, test_index)
    test_dataloader = DataLoader(test_subset, batch_size=128, shuffle=True)

    # Fit the model and evaluate
    model = None
    num_features = train_dataloader.dataset[0][0].shape[0]
    if args.use_bc:
        model = fit_snn_bc_model(train_dataloader, num_features, plot_path)
        fold_metrics = evaluate_snn_bc_model(model, test_dataloader)
    elif args.use_csreg:
        model = fit_snn_reg_model(train_dataloader, num_features, plot_path)
        fold_metrics = evaluate_snn_reg_model(model, test_dataloader)
    elif args.use_pgreg:
        model = fit_snn_reg_model(train_dataloader, num_features, plot_path)
        fold_metrics = evaluate_snn_reg_model(model, test_dataloader)
    else:
        raise ValueError('No prediction task specified')
    
    return i, fold_metrics


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--use_mfp', action='store_true', help='Use Morgan fingerprints')
    args.add_argument('--use_dna', action='store_true', help='Use DNA SNP data')
    args.add_argument('--use_rna', action='store_true', help='Use RNA expression data')
    args.add_argument('--use_prot', action='store_true', help='Use protein expression data')
    args.add_argument('--use_bc', action='store_true', help='Use binary comboscore for prediction task')
    args.add_argument('--use_csreg', action='store_true', help='Use regression on comboscore for prediction task')
    args.add_argument('--use_pgreg', action='store_true', help='Use regression on percentage growth for prediction task')
    args.add_argument('--output_fp', type=str, help='Output file for metrics and plots')
    args.add_argument('--tissue', type=str, default='all_cancer', help='Tissue type to use')
    args.add_argument('--drug_class', type=str, default='all_drugs', help='Drug pair classes to filter the dataset by')
    args.add_argument('--mfp_len', type=int, default=256, help='Length of Morgan fingerprint')
    args.add_argument('--bc_cutoff', type=int, default=0, help='Cutoff for binary comboscore')
    args.add_argument('--folds', type=int, default=10, help='Number of folds for cross-validation')
    args = args.parse_args()

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
        
    # Get the filename
    filename = get_all_cancer_dataset_filename(args.use_mfp, args.use_dna, args.use_rna, args.use_prot, args.use_bc, args.use_csreg, args.use_pgreg, args.mfp_len, args.bc_cutoff)
    filter_indices_fn = None
    if args.tissue != 'all_cancer':
        filter_indices_fn = get_cancer_type_indices_filename(args.tissue, args.use_bc, args.use_csreg, args.use_pgreg)
    elif args.drug_class != 'all_drugs':
        filter_indices_fn = get_drug_class_indices_filename(args.drug_class, args.use_bc, args.use_csreg, args.use_pgreg)
    plot_path = args.output_fp + '_'

    # Load the data
    data = MorganFingerprintDataset(filename=filename, balance_classes=args.use_bc, indices_filter_fn=filter_indices_fn)

    kf = KFold(n_splits=args.folds, shuffle=True)
    all_fold_metrics = pd.DataFrame()
    if args.use_bc:
        all_fold_metrics = pd.DataFrame(columns=['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'MCC', 'AUC', 'Kappa'])
    elif args.use_csreg or args.use_pgreg:
        all_fold_metrics = pd.DataFrame(columns=['MSE', 'RMSE', 'MAE', 'R2', 'Pearson', 'Spearman'])

    # Parallelization Code
    # fold_args = [(i, train_index, test_index) for i, (train_index, test_index) in enumerate(kf.split(data.x))]
    # process_fold_partial = partial(process_fold, data=data, args=args)

    # with Pool() as pool:
    #     results = pool.starmap(process_fold_partial, fold_args)
    
    # results.sort(key=lambda x:x[0])
    # for i, fold_metrics in results:
    #     all_fold_metrics.loc[i] = fold_metrics

    # Serial Code
    for i, (train_index, test_index) in enumerate(kf.split(data.x)):
        print(f"Fold {i+1}")

        train_subset = Subset(data, train_index)
        train_dataloader = DataLoader(train_subset, batch_size=128, shuffle=True)
        test_subset = Subset(data, test_index)
        test_dataloader = DataLoader(test_subset, batch_size=128, shuffle=True)

        # Fit the model and evaluate
        model = None
        num_features = train_dataloader.dataset[0][0].shape[0]
        if args.use_bc:
            model = fit_snn_bc_model(train_dataloader, num_features, plot_path)
            fold_metrics = evaluate_snn_bc_model(model, test_dataloader)
        elif args.use_csreg:
            model = fit_snn_reg_model(train_dataloader, num_features, plot_path)
            fold_metrics = evaluate_snn_reg_model(model, test_dataloader)
        elif args.use_pgreg:
            model = fit_snn_reg_model(train_dataloader, num_features, plot_path)
            fold_metrics = evaluate_snn_reg_model(model, test_dataloader)
        else:
            raise ValueError('No prediction task specified')
        
        all_fold_metrics.loc[i] = fold_metrics

    # Save the metrics
    all_fold_metrics.to_csv(args.output_fp + "all_fold_metrics.csv", index=True, header=True)

    # Print the average metrics
    print(all_fold_metrics.mean())


    
    


    
    
    
