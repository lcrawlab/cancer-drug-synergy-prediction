import numpy as np
from sklearn.model_selection import GroupKFold, KFold


def _to_str_array(values):
    return np.asarray(values).astype(str)


def _build_fold_info(fold_idx, cv_scheme, train_idx, test_idx, held_out_groups):
    return {
        "fold": fold_idx + 1,
        "cv_scheme": cv_scheme,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "held_out_groups": "|".join(held_out_groups),
        "n_held_out_groups": int(len(held_out_groups)),
    }


def _split_unique_groups(unique_groups, n_splits, random_state):
    rng = np.random.default_rng(random_state)
    shuffled = np.asarray(unique_groups, dtype=object).copy()
    rng.shuffle(shuffled)
    partitions = np.array_split(shuffled, n_splits)
    return [part.astype(str).tolist() for part in partitions if len(part) > 0]


def _iter_leave_cell_line(cell_names, n_splits, loo):
    unique_groups = np.unique(cell_names)
    if loo:
        for fold_idx, group in enumerate(unique_groups):
            test_mask = cell_names == group
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            yield fold_idx, train_idx, test_idx, [str(group)]
        return

    if n_splits > len(unique_groups):
        raise ValueError(
            f"Cannot use n_splits={n_splits} with only {len(unique_groups)} cell lines. "
            "Lower --folds or use --loo."
        )

    splitter = GroupKFold(n_splits=n_splits)
    index = np.arange(len(cell_names))
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(index, groups=cell_names)):
        held_out = sorted(np.unique(cell_names[test_idx]).astype(str).tolist())
        yield fold_idx, train_idx, test_idx, held_out


def _iter_leave_drug(nsc1, nsc2, n_splits, loo, random_state):
    unique_drugs = np.unique(np.concatenate([nsc1, nsc2]))
    if loo:
        drug_folds = [[drug] for drug in unique_drugs]
    else:
        if n_splits > len(unique_drugs):
            raise ValueError(
                f"Cannot use n_splits={n_splits} with only {len(unique_drugs)} unique drugs. "
                "Lower --folds or use --loo."
            )
        drug_folds = _split_unique_groups(unique_drugs, n_splits=n_splits, random_state=random_state)

    for fold_idx, fold_drugs in enumerate(drug_folds):
        held_out = np.asarray(fold_drugs, dtype=str)
        test_mask = np.isin(nsc1, held_out) | np.isin(nsc2, held_out)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            raise ValueError(
                f"Invalid leave_drug fold {fold_idx + 1}: "
                f"n_train={len(train_idx)}, n_test={len(test_idx)}. "
                "Try a different --folds value or use --loo."
            )
        yield fold_idx, train_idx, test_idx, sorted(held_out.tolist())


def iter_cv_splits(dataset, cv_scheme="random", n_splits=10, loo=False, random_state=42):
    n_samples = dataset.n_samples
    if cv_scheme == "random":
        if loo:
            raise ValueError("--loo is only valid for leave_cell_line and leave_drug")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(np.arange(n_samples))):
            yield train_idx, test_idx, _build_fold_info(
                fold_idx, cv_scheme, train_idx, test_idx, held_out_groups=[]
            )
        return

    if cv_scheme == "leave_cell_line":
        if dataset.cell_names is None:
            raise ValueError("Dataset is missing cell_names; update dataset creation before grouped CV.")
        cell_names = _to_str_array(dataset.cell_names)
        iterator = _iter_leave_cell_line(cell_names=cell_names, n_splits=n_splits, loo=loo)
        for fold_idx, train_idx, test_idx, held_out_groups in iterator:
            yield train_idx, test_idx, _build_fold_info(
                fold_idx, cv_scheme, train_idx, test_idx, held_out_groups
            )
        return

    if cv_scheme == "leave_drug":
        if dataset.nsc1 is None or dataset.nsc2 is None:
            raise ValueError("Dataset is missing nsc1/nsc2; update dataset creation before grouped CV.")
        nsc1 = _to_str_array(dataset.nsc1)
        nsc2 = _to_str_array(dataset.nsc2)
        iterator = _iter_leave_drug(
            nsc1=nsc1, nsc2=nsc2, n_splits=n_splits, loo=loo, random_state=random_state
        )
        for fold_idx, train_idx, test_idx, held_out_groups in iterator:
            yield train_idx, test_idx, _build_fold_info(
                fold_idx, cv_scheme, train_idx, test_idx, held_out_groups
            )
        return

    raise ValueError("cv_scheme should be one of random, leave_cell_line, or leave_drug")
