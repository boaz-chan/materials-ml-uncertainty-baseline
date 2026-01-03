## Project 1- Uncertainty-Aware Materials ML Baseline

# Motivation

Most materials ML models report only point predictions, despite being used in
high-stakes decision-making contexts such as materials screening and experimental
prioritization.

In practice, practitioners need not only accurate predictions, but also a
quantitative estimate of model confidence to:
- triage candidates,
- manage risk,
- and avoid costly experimental dead-ends.

This project implements a simple, fully reproducible baseline for uncertainty-aware
materials property prediction using classical ML techniques.

# Problem Definition

Given an inorganic crystal structure, we aim to predict its formation energy
(eV/atom) using composition-based descriptors.

- Input: crystal structure (Materials Project format)
- Representation: Magpie composition features
- Target property: formation energy (eV/atom)
- Dataset: Matbench MP formation energy benchmark

# Methodology

1. Feature extraction using Magpie elemental descriptors
2. Train/test split with fixed random seed for reproducibility
3. Baseline model: Random Forest regression
4. Uncertainty estimation via ensemble variance (20 independent models)
5. Diagnostic analysis of uncertainty vs. prediction error

# Key Results

Baseline Random Forest:
- MAE ≈ 0.32 eV
- RMSE ≈ 0.51 eV
- R² ≈ 0.80

Ensemble mean prediction:
- Slight improvement in MAE and RMSE
- Ensemble standard deviation used as epistemic uncertainty proxy

Uncertainty–error correlation:
- Weak correlation observed (ρ ≈ 0.05)

# What This Project Demonstrates

- Practical experience with materials ML pipelines
- Understanding of uncertainty estimation beyond point predictions
- Awareness of limitations of ensemble-based uncertainty
- Emphasis on reproducibility and diagnostic analysis

# Limitations and Future Work

- Small dataset size limits statistical conclusions
- Ensemble variance captures epistemic uncertainty only
- No calibration against experimental uncertainty

Future extensions:
- Bayesian regression or quantile regression
- Active learning loops
- Integration with DFT-derived datasets
