"""
Generate deterministic k-fold subject splits and save to a JSON file.

Run once, commit the output JSON alongside your data. All subsequent training
runs load splits from the file instead of recomputing them, guaranteeing
identical fold assignments across re-runs and collaborators.

Usage:
    python data/generate_splits.py --processed_dir ./data/processed

Output:
    data/splits/kfold_splits.json
"""
import os
import json
import argparse
from datetime import date

import numpy as np
from sklearn.model_selection import StratifiedKFold


def collect_subjects(processed_dir: str, task: str) -> dict[str, str]:
    """Return {subject_id: label_name} sorted alphabetically."""
    label_map = {"healthy": 0, "anxiety": 1}
    subjects: dict[str, str] = {}
    for label_name in label_map:
        label_path = os.path.join(processed_dir, task, label_name)
        if not os.path.isdir(label_path):
            raise FileNotFoundError(f"Expected directory not found: {label_path}")
        for subject in sorted(os.listdir(label_path)):
            subjects[subject] = label_name
    return dict(sorted(subjects.items()))


def generate_kfold(subjects: dict[str, str], k: int, seed: int) -> list[dict]:
    label_map = {"healthy": 0, "anxiety": 1}
    sorted_subjects = list(subjects.keys())
    labels = [label_map[subjects[s]] for s in sorted_subjects]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (train_idxs, val_idxs) in enumerate(skf.split(sorted_subjects, labels)):
        folds.append({
            "fold": fold_idx + 1,
            "val_subjects": [sorted_subjects[i] for i in val_idxs],
            "train_subjects": [sorted_subjects[i] for i in train_idxs],
        })
    return folds


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic k-fold splits")
    parser.add_argument("--processed_dir", type=str, default="./data/processed",
                        help="Root directory of processed data (default: ./data/processed)")
    parser.add_argument("--task", type=str, default="GNG",
                        choices=["GNG", "1backWM", "VF", "SS"],
                        help="Reference task used to enumerate subjects (default: GNG)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for StratifiedKFold (default: 42)")
    parser.add_argument("--output", type=str, default="./data/splits/kfold_splits.json",
                        help="Output path for the splits JSON")
    args = parser.parse_args()

    subjects = collect_subjects(args.processed_dir, args.task)
    sorted_subjects = list(subjects.keys())
    label_counts = {}
    for v in subjects.values():
        label_counts[v] = label_counts.get(v, 0) + 1

    print(f"Reference task : {args.task}")
    print(f"Total subjects : {len(sorted_subjects)}")
    print(f"Class distribution: {label_counts}")
    print(f"Subjects (sorted): {sorted_subjects}")

    result = {
        "generated_at": date.today().strftime("%Y-%m-%d"),
        "seed": args.seed,
        "reference_task": args.task,
        "total_subjects": len(sorted_subjects),
        "class_distribution": label_counts,
        "subjects": subjects,
    }

    for k in [3, 5, 10]:
        folds = generate_kfold(subjects, k, args.seed)
        result[f"kfold_{k}"] = folds
        print(f"\nkfold_{k} splits:")
        for fold in folds:
            print(f"  Fold {fold['fold']:>2}: val={fold['val_subjects']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
