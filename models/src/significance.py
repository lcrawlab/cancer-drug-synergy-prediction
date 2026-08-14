import numpy as np
import pandas as pd
from scipy import stats


HIGHER_IS_BETTER = {
    "Accuracy": True,
    "Sensitivity": True,
    "Specificity": True,
    "Precision": True,
    "F1 Score": True,
    "MCC": True,
    "AUC": True,
    "Kappa": True,
    "R2": True,
    "Pearson": True,
    "Spearman": True,
    "MSE": False,
    "RMSE": False,
    "MAE": False,
}


NULL_BASELINES = {
    "Accuracy": 0.5,
    "AUC": 0.5,
    "MCC": 0.0,
    "Kappa": 0.0,
    "R2": 0.0,
    "Pearson": 0.0,
    "Spearman": 0.0,
}


def corrected_t_test(differences, test_train_ratio):
    diffs = np.asarray(differences, dtype=np.float64)
    diffs = diffs[~np.isnan(diffs)]
    k = len(diffs)
    if k < 2:
        return np.nan, np.nan

    mean_diff = np.mean(diffs)
    s2 = np.var(diffs, ddof=1)
    if np.isclose(s2, 0.0):
        if np.isclose(mean_diff, 0.0):
            return 0.0, 1.0
        return np.inf * np.sign(mean_diff), 0.0

    corrected_var = (1.0 / k + test_train_ratio) * s2
    if corrected_var <= 0:
        return np.nan, np.nan

    t_stat = mean_diff / np.sqrt(corrected_var)
    dof = k - 1
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), dof))
    return float(t_stat), float(p_val)


def _improvement(model_values, baseline_values, higher_is_better):
    model_values = np.asarray(model_values, dtype=np.float64)
    baseline_values = np.asarray(baseline_values, dtype=np.float64)
    if higher_is_better:
        return model_values - baseline_values
    return baseline_values - model_values


def _mean_test_train_ratio(fold_info_df):
    if fold_info_df is None or fold_info_df.empty:
        return 1.0 / 9.0
    ratios = fold_info_df["n_test"] / fold_info_df["n_train"]
    return float(np.mean(ratios))


def summarize_with_significance(model_metrics_df, dummy_metrics_df=None, fold_info_df=None):
    test_train_ratio = _mean_test_train_ratio(fold_info_df)
    rows = []

    for metric in model_metrics_df.columns:
        series = pd.to_numeric(model_metrics_df[metric], errors="coerce")
        metric_values = series.values.astype(np.float64)

        row = {
            "metric": metric,
            "mean": float(np.nanmean(metric_values)),
            "std": float(np.nanstd(metric_values, ddof=1)) if len(series.dropna()) > 1 else np.nan,
            "n_folds": int(series.notna().sum()),
            "test_train_ratio": test_train_ratio,
            "nb_t_vs_null": np.nan,
            "nb_p_vs_null": np.nan,
            "nb_t_vs_dummy": np.nan,
            "nb_p_vs_dummy": np.nan,
        }

        higher_is_better = HIGHER_IS_BETTER.get(metric, True)

        if metric in NULL_BASELINES:
            baseline = np.full_like(metric_values, NULL_BASELINES[metric], dtype=np.float64)
            improvement = _improvement(metric_values, baseline, higher_is_better)
            t_stat, p_val = corrected_t_test(improvement, test_train_ratio)
            row["nb_t_vs_null"] = t_stat
            row["nb_p_vs_null"] = p_val

        if dummy_metrics_df is not None and metric in dummy_metrics_df.columns:
            dummy_values = pd.to_numeric(dummy_metrics_df[metric], errors="coerce").values.astype(np.float64)
            if metric in ["Pearson", "Spearman"] and np.nanstd(dummy_values) == 0:
                rows.append(row)
                continue
            improvement = _improvement(metric_values, dummy_values, higher_is_better)
            t_stat, p_val = corrected_t_test(improvement, test_train_ratio)
            row["nb_t_vs_dummy"] = t_stat
            row["nb_p_vs_dummy"] = p_val

        rows.append(row)

    return pd.DataFrame(rows)
