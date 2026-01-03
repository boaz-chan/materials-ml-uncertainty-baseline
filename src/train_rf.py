from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:

    # 1 Load data
    X_path = FEATURE_DIR / "magpie_X_test_200.parquet"
    y_path = FEATURE_DIR / "magpie_y_test_200.parquet"

    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)["e_form"].to_numpy()

    # Drop non-numeric columns (composition string)
    X_num = X.drop(columns=[c for c in X.columns if X[c].dtype == "object"])

    train_idx = np.load(SPLIT_DIR / "train_idx.npy")
    test_idx = np.load(SPLIT_DIR / "test_idx.npy")

    X_train = X_num.iloc[train_idx].to_numpy()
    y_train = y[train_idx]
    X_test = X_num.iloc[test_idx].to_numpy()
    y_test = y[test_idx]

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    # 2 Baseline single RF
    # Note: n_jobs=1 for stability on Windows.
    # Later, once stable, you can increase n_jobs for speed.
    base_rf = RandomForestRegressor(
        n_estimators=500,
        random_state=0,
        n_jobs=1,
    )
    base_rf.fit(X_train, y_train)
    base_pred = base_rf.predict(X_test)

    base_mae = mean_absolute_error(y_test, base_pred)
    base_rmse = rmse(y_test, base_pred)
    base_r2 = r2_score(y_test, base_pred)

    print("\nBaseline RF performance:")
    print(f"  MAE:  {base_mae:.3f} eV")
    print(f"  RMSE: {base_rmse:.3f} eV")
    print(f"  R²:   {base_r2:.3f}")

    # 3 Ensemble for uncertainty (epistemic proxy)
    n_models = 20
    print(f"\nTraining ensemble of {n_models} RF models...")

    all_preds = np.zeros((n_models, len(y_test)), dtype=float)

    for i in range(n_models):
        rf = RandomForestRegressor(
            n_estimators=500,
            random_state=1000 + i,  # different seed each model
            n_jobs=1,
        )
        rf.fit(X_train, y_train)
        all_preds[i] = rf.predict(X_test)

    mean_pred = all_preds.mean(axis=0)
    std_pred = all_preds.std(axis=0)

    ens_mae = mean_absolute_error(y_test, mean_pred)
    ens_rmse = rmse(y_test, mean_pred)
    ens_r2 = r2_score(y_test, mean_pred)

    print("\nEnsemble performance (mean prediction):")
    print(f"  MAE:  {ens_mae:.3f} eV")
    print(f"  RMSE: {ens_rmse:.3f} eV")
    print(f"  R²:   {ens_r2:.3f}")

    print("\nUncertainty (std) summary:")
    print(
        f"  min={std_pred.min():.6f}  "
        f"median={np.median(std_pred):.6f}  "
        f"max={std_pred.max():.6f}"
    )

    # 4 Save predictions for analysis (restart-safe)

    df_out = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred_mean": mean_pred,
            "y_pred_std": std_pred,
            "abs_error": np.abs(mean_pred - y_test),
        }
    )

    out_path = RESULTS_DIR / "predictions_test.csv"
    df_out.to_csv(out_path, index=False)

    print("\nSaved results table:", out_path)


if __name__ == "__main__":
    main()
