from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:

    # 1 Load features + target
    X_path = FEATURE_DIR / "magpie_X_test_200.parquet"
    y_path = FEATURE_DIR / "magpie_y_test_200.parquet"

    print("Loading:", X_path.name)
    X = pd.read_parquet(X_path)

    print("Loading:", y_path.name)
    y = pd.read_parquet(y_path)["e_form"].to_numpy()

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 2 Define numeric ML matrix (drop non-numeric columns)
    non_numeric_cols = [c for c in X.columns if X[c].dtype == "object"]
    print("Non-numeric columns:", non_numeric_cols)

    X_num = X.drop(columns=non_numeric_cols)
    print("X_num shape:", X_num.shape)

    # 3 Reproducible split
    test_size = 0.2
    seed = 42

    idx = np.arange(len(X_num))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed
    )

    print("Train size:", len(train_idx))
    print("Test size :", len(test_idx))

    # 4 Save split indices
    np.save(SPLIT_DIR / "train_idx.npy", train_idx)
    np.save(SPLIT_DIR / "test_idx.npy", test_idx)

    meta = {
        "dataset": "matbench_mp_e_form (test_200 subset)",
        "features": "Magpie (matminer ElementProperty preset: magpie)",
        "n_samples": int(len(X_num)),
        "test_size": float(test_size),
        "random_seed": int(seed),
    }
    with open(SPLIT_DIR / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", SPLIT_DIR / "train_idx.npy")
    print("Saved:", SPLIT_DIR / "test_idx.npy")
    print("Saved:", SPLIT_DIR / "split_meta.json")


if __name__ == "__main__":
    main()
