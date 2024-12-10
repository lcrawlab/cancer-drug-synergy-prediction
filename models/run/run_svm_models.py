import argparse
import numpy as np
import pandas as pd
import scipy.stats
from dataset_creation.datasets import *
from models.src.models_baselines import *
from sklearn.model_selection import KFold
from sklearn.metrics import *

# Fit the svm model for binary classification
# INPUT:
#   X_train: np.ndarray
#   y_train: np.ndarray
# OUTPUT:
#   model: SVMModelBC
def fit_svm_bc_model(X_train, y_train):
    # Initialize and train the model
    model = SVMModelBC()
    model.fit(X_train, y_train)
    print("Model trained")
    return model


# Evaluate the svm model for binary classification
# INPUT:
#   model: SVMModelBC
#   X_test: np.ndarray
#   y_test: np.ndarray
# OUTPUT:
#   fold_metrics: list
def evaluate_svm_bc_model(model, X_test, y_test):
    # Evaluation
    with torch.no_grad():
        y_pred = model.predict(X_test)
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


# Fit the svm model for regression on comboscore or percent growth
# INPUT:
#   X_train: np.ndarray
#   y_train: np.ndarray
# OUTPUT:
#   model: SVMModelRegression
def fit_svm_reg_model(X_train, y_train):
    # Initialize and train the model
    model = SVMModelRegression()

    # On large datasets, the SVM model says the value is too large to convert to numpy_int32
    # So let's split up the data into 5 chunks and train on each chunk
    # This is a workaround for the issue
    # Split the data into 5 chunks
    chunk_size = int(len(X_train) / 5)
    for i in range(5):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if i == 4:
            end = len(X_train)
        model.fit(X_train[start:end], y_train[start:end])
        print(f"Model trained on chunk {i}")
    
    print("Model trained")
    return model


# Evaluate the svm model for regression on comboscore or percent growth
# INPUT:
#   model: SVMModelRegression
#   X_test: np.ndarray
#   y_test: np.ndarray
# OUTPUT:
#   fold_metrics: list
def evaluate_svm_reg_model(model, X_test, y_test):
    # Evaluation
    with torch.no_grad():
        y_pred = np.ndarray.flatten(model.predict(X_test))
        y_test = np.ndarray.flatten(y_test)
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


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--use_mfp', action='store_true', help='Use Morgan fingerprints')
    args.add_argument('--use_dna', action='store_true', help='Use DNA SNP data')
    args.add_argument('--use_rna', action='store_true', help='Use RNA expression data')
    args.add_argument('--use_prot', action='store_true', help='Use protein expression data')
    args.add_argument('--use_bc', action='store_true', help='Use binary comboscore for prediction task')
    args.add_argument('--use_csreg', action='store_true', help='Use regression on comboscore for prediction task')
    args.add_argument('--use_pgreg', action='store_true', help='Use regression on percentage growth for prediction task')
    args.add_argument('--output_fp', type=str, help='Output file for metrics')
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

    # Load the data
    data = MorganFingerprintDataset(filename=filename, balance_classes=args.use_bc, indices_filter_fn=filter_indices_fn)
    X = data.x.detach().cpu().numpy().astype(np.float32)
    y = data.y.detach().cpu().numpy().astype(np.float32)
    # flatten y
    y = np.ndarray.flatten(y)


    kf = KFold(n_splits=args.folds, shuffle=True)
    all_fold_metrics = pd.DataFrame()
    if args.use_bc:
        all_fold_metrics = pd.DataFrame(columns=['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'MCC', 'AUC', 'Kappa'])
    elif args.use_csreg or args.use_pgreg:
        all_fold_metrics = pd.DataFrame(columns=['MSE', 'RMSE', 'MAE', 'R2', 'Pearson', 'Spearman'])
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {i+1}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model and evaluate
        model = None
        if args.use_bc:
            model = fit_svm_bc_model(X_train, y_train)
            fold_metrics = evaluate_svm_bc_model(model, X_test, y_test)
        elif args.use_csreg:
            model = fit_svm_reg_model(X_train, y_train)
            fold_metrics = evaluate_svm_reg_model(model, X_test, y_test)
        elif args.use_pgreg:
            model = fit_svm_reg_model(X_train, y_train)
            fold_metrics = evaluate_svm_reg_model(model, X_test, y_test)
        else:
            raise ValueError('No prediction task specified')
        
        all_fold_metrics.loc[i] = fold_metrics

    # Save the metrics
    all_fold_metrics.to_csv(args.output_fp, index=True, header=True)

    # Print the average metrics
    print(all_fold_metrics.mean())
    
