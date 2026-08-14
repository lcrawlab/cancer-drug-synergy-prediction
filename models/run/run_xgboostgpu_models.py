import argparse
import os
import numpy as np
import pandas as pd
import scipy.stats
from dataset_creation.datasets import *
from models.src.cv_splits import iter_cv_splits
from models.src.models_baselines import *
from models.src.significance import summarize_with_significance
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import *


def _safe_roc_auc(y_true, y_pred):
    if np.unique(y_true).shape[0] < 2:
        return np.nan
    return roc_auc_score(y_true, y_pred)


# Fit the GPU XGBoost model for binary classification
# INPUT:
#   X_train: Torch tensor
#   y_train: Torch tensor
# OUTPUT:
#   model: GPUXGBoostModelBC
def fit_xgboostgpu_bc_model(X_train, y_train):
    # Initialize and train the XGBoost binary classifier
    model = GPUXGBoostModelBC()
    model.fit(X_train, y_train)
    print("Model trained")
    return model


# Evaluate the GPU XGBoost model for binary classification
# INPUT:
#   model: GPUXGBoostModelBC
#   X_test: Torch tensor
#   y_test: Torch tensor
# OUTPUT:
#   fold_metrics: list
def evaluate_xgboostgpu_bc_model(model, X_test, y_test):
    # Evaluation
    y_pred = model.predict(X_test)

    y_test = y_test.cpu().numpy()

    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1s = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = _safe_roc_auc(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    return [accuracy, sensitivity, specificity, precision, f1s, mcc, auc, kappa]


# Fit the GPU XGBoost model for regression on comboscore or percent growth
# INPUT:
#   X_train: Torch tensor
#   y_train: Torch tensor
# OUTPUT:
#   model: GPUXGBoostModelRegression
def fit_xgboostgpu_reg_model(X_train, y_train):
    # Initialize and train the XGBoost regressor
    model = GPUXGBoostModelRegression()
    model.fit(X_train, y_train)
    print("Model trained")
    return model


# Evaluate the GPU XGBoost model for regression on comboscore or percent growth
# INPUT:
#   model: GPUXGBoostModelRegression
#   X_test: Torch tensor
#   y_test: Torch tensor
# OUTPUT:
#   fold_metrics: list
def evaluate_xgboostgpu_reg_model(model, X_test, y_test):
    # Evaluation
    y_pred = np.ndarray.flatten(model.predict(X_test))

    y_test = y_test.cpu().numpy()
    y_test = np.ndarray.flatten(y_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pearson = scipy.stats.pearsonr(y_test, y_pred)[0]
    spearman = scipy.stats.spearmanr(y_test, y_pred)[0]

    return [mse, rmse, mae, r2, pearson, spearman]


def fit_dummy_bc_model(y_train):
    model = DummyClassifier(strategy='most_frequent')
    X_train_dummy = np.zeros((len(y_train), 1), dtype=np.float32)
    model.fit(X_train_dummy, y_train)
    return model


def evaluate_dummy_bc_model(model, y_test):
    X_test_dummy = np.zeros((len(y_test), 1), dtype=np.float32)
    y_pred = model.predict(X_test_dummy)

    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1s = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = _safe_roc_auc(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    return [accuracy, sensitivity, specificity, precision, f1s, mcc, auc, kappa]


def fit_dummy_reg_model(y_train):
    model = DummyRegressor(strategy='mean')
    X_train_dummy = np.zeros((len(y_train), 1), dtype=np.float32)
    model.fit(X_train_dummy, y_train)
    return model


def evaluate_dummy_reg_model(model, y_test):
    X_test_dummy = np.zeros((len(y_test), 1), dtype=np.float32)
    y_pred = np.ndarray.flatten(model.predict(X_test_dummy))

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pearson = scipy.stats.pearsonr(y_test, y_pred)[0] if np.std(y_pred) > 0 else np.nan
    spearman = scipy.stats.spearmanr(y_test, y_pred)[0] if np.std(y_pred) > 0 else np.nan

    return [mse, rmse, mae, r2, pearson, spearman]


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--use_mfp', action='store_true', help='Use Morgan fingerprints')
    args.add_argument('--use_dna', action='store_true', help='Use DNA SNP data')
    args.add_argument('--use_rna', action='store_true', help='Use RNA expression data')
    args.add_argument('--use_prot', action='store_true', help='Use protein expression data')
    args.add_argument('--score', type=str, default='COMBOSCORE', help='Score to use for prediction task, should be one of COMBOSCORE, PERCENTGROWTH, ZIP, HSA')
    args.add_argument('--use_bc', action='store_true', help='Use binary classification of score for prediction task')
    args.add_argument('--use_csreg', action='store_true', help='Use regression of score for prediction task')
    args.add_argument('--use_pgreg', action='store_true', help='Use regression of percent growth for prediction task')
    args.add_argument('--output_fp', type=str, help='Output file for metrics')
    args.add_argument('--tissue', type=str, default='all_cancer', help='Tissue type to use')
    args.add_argument('--drug_class', type=str, default='all_drugs', help='Drug pair classes to filter the dataset by')
    args.add_argument('--mfp_len', type=int, default=256, help='Length of Morgan fingerprint')
    args.add_argument('--bc_cutoff', type=int, default=0, help='Cutoff for binary comboscore')
    args.add_argument('--folds', type=int, default=10, help='Number of folds for cross-validation')
    args.add_argument('--cv_scheme', type=str, default='random', help='Cross-validation scheme to use')
    args.add_argument('--loo', action='store_true', help='Use leave-one-group-out for grouped CV')
    args = args.parse_args()

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
    if args.cv_scheme not in ['random', 'leave_cell_line', 'leave_drug']:
        raise ValueError('cv_scheme should be one of random, leave_cell_line, leave_drug')
    if args.cv_scheme == 'random' and args.loo:
        raise ValueError('--loo is invalid when cv_scheme is random')
    if args.tissue != 'all_cancer':
        valid_cancer_types = ['breast', 'cns', 'colon', 'leukemia', 'melanoma', 'nsclc', 'ovarian', 'prostate', 'renal']
        if args.tissue not in valid_cancer_types:
            raise ValueError(f'tissue should be one of {valid_cancer_types}')
    if args.drug_class != 'all_drugs':
        valid_drug_classes = ['chemo_chemo', 'chemo_targeted', 'chemo_other', 'targeted_targeted', 'targeted_other', 'other_other']
        if args.drug_class not in valid_drug_classes:
            raise ValueError(f'drug_class should be one of {valid_drug_classes}')
    
    # Get the filename
    h5_path = 'data/ASP_dataset_slices/all_256mfpdnarnaprot.h5'
    non_pg_data_path = 'data/ASP_dataset_slices/drug_comboscore_hsa_zip.csv'
    pg_data_path = 'data/ASP_dataset_slices/drug_percent_growth.csv'

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
        #device='cuda',  # Use GPU
    )
    
    X = data.x
    y = torch.flatten(data.y)

    metric_columns = []
    if args.use_bc:
        metric_columns = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'MCC', 'AUC', 'Kappa']
    elif args.use_csreg or args.use_pgreg:
        metric_columns = ['MSE', 'RMSE', 'MAE', 'R2', 'Pearson', 'Spearman']

    model_fold_rows = []
    dummy_fold_rows = []
    fold_info_rows = []

    split_iterator = iter_cv_splits(
        dataset=data,
        cv_scheme=args.cv_scheme,
        n_splits=args.folds,
        loo=args.loo,
        random_state=42,
    )

    for i, (train_index, test_index, fold_info) in enumerate(split_iterator):
        print(f"Fold {i+1}")
        train_idx_list = train_index.tolist()
        test_idx_list = test_index.tolist()
        X_train_cpu, X_test_cpu = X[train_idx_list], X[test_idx_list]
        y_train_cpu, y_test_cpu = y[train_idx_list], y[test_idx_list]

        # Move to GPU
        X_train = X_train_cpu.to('cuda')
        y_train = y_train_cpu.to('cuda')
        X_test = X_test_cpu.to('cuda')
        y_test = y_test_cpu.to('cuda')

        y_train_np = np.ndarray.flatten(y_train_cpu.numpy())
        y_test_np = np.ndarray.flatten(y_test_cpu.numpy())

        # Fit the model and evaluate
        model = None
        if args.use_bc:
            model = fit_xgboostgpu_bc_model(X_train, y_train)
            fold_metrics = evaluate_xgboostgpu_bc_model(model, X_test, y_test)
            dummy_model = fit_dummy_bc_model(y_train_np)
            dummy_metrics = evaluate_dummy_bc_model(dummy_model, y_test_np)
        elif args.use_csreg:
            model = fit_xgboostgpu_reg_model(X_train, y_train)
            fold_metrics = evaluate_xgboostgpu_reg_model(model, X_test, y_test)
            dummy_model = fit_dummy_reg_model(y_train_np)
            dummy_metrics = evaluate_dummy_reg_model(dummy_model, y_test_np)
        elif args.use_pgreg:
            model = fit_xgboostgpu_reg_model(X_train, y_train)
            fold_metrics = evaluate_xgboostgpu_reg_model(model, X_test, y_test)
            dummy_model = fit_dummy_reg_model(y_train_np)
            dummy_metrics = evaluate_dummy_reg_model(dummy_model, y_test_np)
        else:
            raise ValueError('No prediction task specified')

        model_row = dict(zip(metric_columns, fold_metrics))
        model_row['n_train'] = fold_info['n_train']
        model_row['n_test'] = fold_info['n_test']
        model_fold_rows.append(model_row)

        dummy_fold_rows.append(dict(zip(metric_columns, dummy_metrics)))
        fold_info_rows.append(fold_info)

    all_fold_metrics = pd.DataFrame(model_fold_rows)
    all_dummy_metrics = pd.DataFrame(dummy_fold_rows)
    fold_info_df = pd.DataFrame(fold_info_rows)
    summary_df = summarize_with_significance(
        model_metrics_df=all_fold_metrics[metric_columns],
        dummy_metrics_df=all_dummy_metrics,
        fold_info_df=fold_info_df,
    )

    # Save the metrics
    all_fold_metrics.to_csv(args.output_fp, index=False, header=True)
    output_root, output_ext = os.path.splitext(args.output_fp)
    summary_fp = f'{output_root}_summary.csv' if output_ext else args.output_fp + '_summary.csv'
    fold_info_fp = f'{output_root}_fold_info.csv' if output_ext else args.output_fp + '_fold_info.csv'
    summary_df.to_csv(summary_fp, index=False, header=True)
    fold_info_df.to_csv(fold_info_fp, index=False, header=True)

    # Print mean ± SD and mark significance vs dummy.
    for metric in metric_columns:
        metric_row = summary_df[summary_df['metric'] == metric]
        if metric_row.empty:
            continue
        row = metric_row.iloc[0]
        significant_vs_dummy = pd.notna(row['nb_p_vs_dummy']) and row['nb_p_vs_dummy'] < 0.05
        sig_tag = ' *' if significant_vs_dummy else ''
        print(f"{metric}: {row['mean']:.4f} ± {row['std']:.4f}{sig_tag}")

    print(f"Saved fold metrics to {args.output_fp}")
    print(f"Saved significance summary to {summary_fp}")
    print(f"Saved fold group metadata to {fold_info_fp}")
