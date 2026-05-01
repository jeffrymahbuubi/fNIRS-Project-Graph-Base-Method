# pylint: disable=too-many-arguments, too-many-locals, too-many-instance-attributes
import configparser
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .utils import compute_statistical_features, pearson_correlation_matrix, coherence_matrix


class fNIRSGraphDataset(Dataset):
    LABEL_MAP = {"healthy": 0, "anxiety": 1}

    def __init__(
        self,
        root: str,
        task_type: str = "GNG",
        data_type: str = "hbo",
        max_trials: Optional[int] = None,
        directed: bool = False,
        corr_threshold: float = 0.1,
        self_loops: bool = False,
    ):
        self.root = root
        self.task_type = task_type
        self.data_type = data_type.lower()
        self.max_trials = max_trials
        self.directed = directed
        self.corr_threshold = corr_threshold
        self.self_loops = self_loops
        self._graphs: List[Data] = []
        self._load()

    def _get_fs(self, subject_dir: str) -> float:
        for cfg_file in glob.glob(os.path.join(subject_dir, "*.data")):
            cfg = configparser.ConfigParser()
            cfg.read(cfg_file)
            try:
                return float(cfg["GeneralInfo"]["SamplingRate"])
            except (KeyError, ValueError):
                pass
        return 10.0

    def _build_graph(self, trial: np.ndarray, fs: float) -> Data:
        if trial.ndim == 2 and trial.shape[0] > trial.shape[1]:
            trial = trial.T
        stats = compute_statistical_features(trial, channels_first=True)
        node_feats = np.stack([
            stats["mean"], stats["min"], stats["max"],
            stats["skewness"], stats["kurtosis"], stats["variance"],
        ], axis=1)
        node_feats = np.nan_to_num(node_feats, nan=0.0)

        corr_mat = pearson_correlation_matrix(trial, channels_first=True)
        coh_mat, _, _ = coherence_matrix(trial, fs=fs, coherence_ratio="1/3", channels_first=True)

        C = trial.shape[0]
        edge_src, edge_dst, edge_feats = [], [], []
        for i in range(C):
            j_range = range(C) if self.directed else range(i + 1, C)
            for j in j_range:
                if i == j:
                    if self.self_loops:
                        edge_src.append(i)
                        edge_dst.append(j)
                        edge_feats.append([abs(corr_mat[i, j]), float(coh_mat[i, j])])
                    continue
                if abs(corr_mat[i, j]) >= self.corr_threshold:
                    edge_src.append(i)
                    edge_dst.append(j)
                    edge_feats.append([abs(corr_mat[i, j]), float(coh_mat[i, j])])
                    if not self.directed:
                        edge_src.append(j)
                        edge_dst.append(i)
                        edge_feats.append([abs(corr_mat[i, j]), float(coh_mat[i, j])])

        if edge_src:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_attr = torch.tensor(edge_feats, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float)

        return Data(
            x=torch.tensor(node_feats, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

    def _load(self) -> None:
        for class_name, label in self.LABEL_MAP.items():
            class_dir = os.path.join(self.root, self.task_type, class_name)
            if not os.path.isdir(class_dir):
                continue
            for subject in sorted(os.listdir(class_dir)):
                subject_dir = os.path.join(class_dir, subject)
                if not os.path.isdir(subject_dir):
                    continue
                fs = self._get_fs(subject_dir)
                data_dir = os.path.join(subject_dir, self.data_type)
                if not os.path.isdir(data_dir):
                    continue
                trial_files = sorted(
                    glob.glob(os.path.join(data_dir, "*.npy")),
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
                )
                if self.max_trials is not None:
                    trial_files = trial_files[: self.max_trials]
                for trial_path in trial_files:
                    trial = np.load(trial_path)
                    if not np.all(np.isfinite(trial)):
                        continue
                    graph = self._build_graph(trial, fs)
                    graph.y = torch.tensor(label, dtype=torch.long)
                    graph.subject_id = subject
                    self._graphs.append(graph)

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> Data:
        return self._graphs[idx]

    def compute_stats(self) -> Dict[str, torch.Tensor]:
        all_x = torch.cat([g.x for g in self._graphs], dim=0)
        valid_ea = [g.edge_attr for g in self._graphs if g.edge_attr.shape[0] > 0]
        mean_x = all_x.mean(dim=0)
        std_x = all_x.std(dim=0).clamp(min=1e-8)
        if valid_ea:
            all_ea = torch.cat(valid_ea, dim=0)
            mean_ea = all_ea.mean(dim=0)
            std_ea = all_ea.std(dim=0).clamp(min=1e-8)
        else:
            mean_ea = torch.zeros(2)
            std_ea = torch.ones(2)
        return {"mean_x": mean_x, "std_x": std_x, "mean_ea": mean_ea, "std_ea": std_ea}


# ---------------------------------------------------------------------------
# SubsetWithTransform
# ---------------------------------------------------------------------------

class SubsetWithTransform(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Data:
        graph = self.subset[idx]
        if self.transform is not None:
            graph = self.transform(graph)
        return graph


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _group_indices_by_subject(
    dataset,
) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    subj_to_indices: Dict[str, List[int]] = defaultdict(list)
    subj_to_label: Dict[str, int] = {}
    for i in range(len(dataset)):
        g = dataset[i]
        sid = str(g.subject_id)
        lbl = int(g.y.item() if isinstance(g.y, torch.Tensor) else g.y)
        subj_to_indices[sid].append(i)
        if sid in subj_to_label and subj_to_label[sid] != lbl:
            raise ValueError(f"Subject {sid} has inconsistent labels.")
        subj_to_label[sid] = lbl
    return subj_to_indices, subj_to_label


def _subjects_to_indices(dataset, subjects: Sequence[str]) -> List[int]:
    targets = set(map(str, subjects))
    return [i for i in range(len(dataset)) if str(dataset[i].subject_id) in targets]


def _stratified_subject_split(
    subject_ids: List[str],
    subject_labels: List[int],
    val_ratio: float,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    idx = np.arange(len(subject_ids))
    ((train_idx, val_idx),) = sss.split(idx, subject_labels)
    return [subject_ids[i] for i in train_idx], [subject_ids[i] for i in val_idx]


def _subject_kfold_indices(
    subject_ids: List[str],
    subject_labels: List[int],
    n_splits: int,
    random_state: int = 42,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    idx = np.arange(len(subject_ids))
    for fold, (tr, va) in enumerate(skf.split(idx, subject_labels), start=1):
        yield fold, [subject_ids[i] for i in tr], [subject_ids[i] for i in va]


def _print_split_info(tag: str, dataset, indices: List[int]) -> None:
    subjects = {str(dataset[i].subject_id) for i in indices}
    label_counts = Counter(
        int(dataset[i].y.item() if isinstance(dataset[i].y, torch.Tensor) else dataset[i].y)
        for i in indices
    )
    print(f"  {tag}: {len(indices)} graphs | {len(subjects)} subjects | labels={dict(label_counts)}")


def _make_loaders(
    dataset,
    train_indices: List[int],
    val_indices: List[int],
    batch_size: int,
    shuffle_train: bool,
    num_workers: int,
    pin_memory: bool,
    train_transform,
    val_transform,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = SubsetWithTransform(Subset(dataset, train_indices), transform=train_transform)
    val_ds = SubsetWithTransform(Subset(dataset, val_indices), transform=val_transform)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Public loader functions
# ---------------------------------------------------------------------------

def get_holdout_loaders(
    dataset,
    *,
    batch_size: int = 8,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    val_subjects: Optional[Sequence[str]] = None,
    val_ratio: float = 0.2,
    random_state: int = 42,
    train_transform=None,
    val_transform=None,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    subj_to_indices, subj_to_label = _group_indices_by_subject(dataset)
    subject_ids = sorted(subj_to_indices.keys())
    subject_labels = [subj_to_label[sid] for sid in subject_ids]

    if val_subjects is None:
        train_subjects, val_subjects = _stratified_subject_split(
            subject_ids, subject_labels, val_ratio, random_state
        )
    else:
        val_subjects = list(map(str, val_subjects))
        train_subjects = [s for s in subject_ids if s not in set(val_subjects)]

    train_indices = _subjects_to_indices(dataset, train_subjects)
    val_indices = _subjects_to_indices(dataset, val_subjects)

    if verbose:
        print("=== Holdout Split (subject-level) ===")
        _print_split_info("Train", dataset, train_indices)
        _print_split_info("Val  ", dataset, val_indices)
        print("=====================================")

    return _make_loaders(
        dataset, train_indices, val_indices,
        batch_size, shuffle_train, num_workers, pin_memory,
        train_transform, val_transform,
    )


def get_kfold_loaders(
    dataset,
    *,
    n_splits: int = 5,
    batch_size: int = 8,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    random_state: int = 42,
    train_transform=None,
    val_transform=None,
    verbose: bool = True,
) -> List[Tuple[DataLoader, DataLoader]]:
    subj_to_indices, subj_to_label = _group_indices_by_subject(dataset)
    subject_ids = sorted(subj_to_indices.keys())
    subject_labels = [subj_to_label[sid] for sid in subject_ids]

    fold_loaders = []
    for fold_id, tr_subjects, va_subjects in _subject_kfold_indices(
        subject_ids, subject_labels, n_splits, random_state
    ):
        train_indices = _subjects_to_indices(dataset, tr_subjects)
        val_indices = _subjects_to_indices(dataset, va_subjects)
        if verbose:
            print(f"=== K-Fold {fold_id}/{n_splits} (subject-level) ===")
            _print_split_info("Train", dataset, train_indices)
            _print_split_info("Val  ", dataset, val_indices)
            print("=" * 45)
        fold_loaders.append(
            _make_loaders(
                dataset, train_indices, val_indices,
                batch_size, shuffle_train, num_workers, pin_memory,
                train_transform, val_transform,
            )
        )
    return fold_loaders


def get_kfold_loaders_from_json(
    dataset,
    splits_json: str,
    n_splits: int = 5,
    *,
    batch_size: int = 8,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    train_transform=None,
    val_transform=None,
    verbose: bool = True,
) -> List[Tuple["DataLoader", "DataLoader"]]:
    with open(splits_json) as f:
        data = json.load(f)
    key = f"kfold_{n_splits}"
    if key not in data:
        available = [k for k in data if k.startswith("kfold_")]
        raise ValueError(f"Key '{key}' not in splits JSON. Available: {available}")
    fold_loaders = []
    for fold_entry in data[key]:
        fold_id = fold_entry["fold"]
        train_indices = _subjects_to_indices(dataset, fold_entry["train_subjects"])
        val_indices = _subjects_to_indices(dataset, fold_entry["val_subjects"])
        if verbose:
            print(f"=== K-Fold {fold_id}/{n_splits} (from JSON, subject-level) ===")
            _print_split_info("Train", dataset, train_indices)
            _print_split_info("Val  ", dataset, val_indices)
            print("=" * 45)
        fold_loaders.append(
            _make_loaders(
                dataset, train_indices, val_indices,
                batch_size, shuffle_train, num_workers, pin_memory,
                train_transform, val_transform,
            )
        )
    return fold_loaders


def get_loso_loaders(
    dataset,
    *,
    batch_size: int = 8,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    train_transform=None,
    val_transform=None,
    verbose: bool = True,
) -> List[Tuple[DataLoader, DataLoader, str]]:
    subj_to_indices, _ = _group_indices_by_subject(dataset)
    subject_ids = sorted(subj_to_indices.keys())

    fold_loaders = []
    n = len(subject_ids)
    for fold_id, val_subject in enumerate(subject_ids, start=1):
        val_indices = subj_to_indices[val_subject]
        train_indices = [
            idx for sid, indices in subj_to_indices.items()
            if sid != val_subject
            for idx in indices
        ]
        if verbose:
            print(f"=== LOSO {fold_id}/{n} — Val subject: {val_subject} ===")
            _print_split_info("Train", dataset, train_indices)
            _print_split_info("Val  ", dataset, val_indices)
            print("=" * 50)
        train_loader, val_loader = _make_loaders(
            dataset, train_indices, val_indices,
            batch_size, shuffle_train, num_workers, pin_memory,
            train_transform, val_transform,
        )
        fold_loaders.append((train_loader, val_loader, val_subject))
    return fold_loaders
