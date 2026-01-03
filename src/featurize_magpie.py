from __future__ import annotations

from pathlib import Path
import pandas as pd

from matminer.datasets import load_dataset
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty


# Paths (always relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:

    # 1 Load dataset


print("Loading dataset: matbench_mp_e_form (via matminer)...")
df = load_dataset("matbench_mp_e_form")
print("Rows:", len(df))
print("Columns:", list(df.columns))  # expected: ['structure', 'e_form']


# 2 Derive composition (Magpie needs composition)


print("Deriving composition from structure...")
df["composition"] = df["structure"].apply(lambda s: s.composition)

# Keep only what we need to avoid saving heavy objects (structures)
df = df[["composition", "e_form"]].copy()


# 3 Small test first (stability check)


n_test = 200
print(f"Sampling {n_test} rows (test)...")
df_small = df.sample(n=n_test, random_state=0).reset_index(drop=True)


# 4 Build Magpie featurizer

# Note: ElementProperty.from_preset("magpie") is the standard Magpie descriptor set
print("Building Magpie featurizer...")
magpie = ElementProperty.from_preset("magpie")

# MultipleFeaturizer lets us combine multiple featurizers later (e.g., Magpie + Stoichiometry)
featurizer = MultipleFeaturizer([magpie])

# IMPORTANT FOR WINDOWS STABILITY:
# matminer can use multiprocessing in some contexts; we avoid that and run single-process.
featurizer.set_n_jobs(1)


# 5 Featurize

print(f"Featurizing {n_test} rows...")
X = featurizer.featurize_dataframe(
    df_small[["composition"]].copy(),
    col_id="composition",
    ignore_errors=True,
    return_errors=True,
    pbar=False,
)

# Parquet cannot store pymatgen Composition objects -> convert composition to string
X["composition"] = X["composition"].apply(str)


# 6 Save outputs (restart-safe)

X_path = FEATURE_DIR / "magpie_X_test_200.parquet"
y_path = FEATURE_DIR / "magpie_y_test_200.parquet"

X.to_parquet(X_path, index=False)
df_small[["e_form"]].to_parquet(y_path, index=False)

print("Saved feature matrix:", X_path)
print("Saved target vector :", y_path)
print("X shape:", X.shape)


if __name__ == "__main__":
    main()
