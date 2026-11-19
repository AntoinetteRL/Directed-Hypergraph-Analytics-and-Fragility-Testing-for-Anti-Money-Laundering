# Directed-Hypergraph Analytics and Fragility Testing for Anti-Money Laundering

## Download the complete code and data package

Download the complete package from the
[latest release](../../releases/latest).

Under **Assets**, select:

**Directed Hypergraph Analytics and Fragility Testing for Anti Money Laundering(1).zip**

The package includes the datasets, processed inputs, and code required to run the experiments.

Download the attached package rather than GitHub's automatically generated **Source code (zip)** or **Source code (tar.gz)**.

Research code and data for the paper by Run Li and Paul Laux.

## Experiments
1. Address classification with original Elliptic++ features (LR, RF, MLP, XGBoost; Table 5).
2. Address classification with 55 original and nine structural features (Table 6).
3. XGBoost risk scoring of unknown addresses (Table 7).
4. Enumeration of illicit-to-licit motifs with one, two and three unknown intermediaries (Table 8).
5. Progressive risk-ranked intermediary removal, path loss and first-failure thresholds (Table 9).

## Installation
Use Python 3.12 in a fresh virtual environment.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
```

The single command runs all experiment stages. Outputs are computed into `results/`; files in `expected/` are used only for comparison, never substituted for model outputs. The final verification step reports any deviation from the values printed in the paper.

Individual stages:
```bash
python run.py --stage prepare
python run.py --stage baseline
python run.py --stage classification
python run.py --stage prediction
python run.py --stage motifs
python run.py --stage fragility
python run.py --stage verify
```

Memory and runtime depend on the host. The full baseline table is approximately 0.6 GB on disk; allow substantial additional RAM for pandas copies and model fitting. Bootstrap intervals use 2,000 resamples.

## Data
The full-data ZIP includes six original actor CSVs in `data/raw/`, the canonical enriched feature table in `data/features/`, and address-aware split records and classification matrices in `data/processed/`.

Original Elliptic++ repository:
https://github.com/git-disl/EllipticPlusPlus

Dataset authors' download folder (linked by the original repository):
https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l?usp=sharing

Required actor files are wallets_features.csv, wallets_classes.csv, AddrAddr_edgelist.csv, AddrTx_edgelist.csv and TxAddr_edgelist.csv. The supplied wallets_features_classes_combined.csv is the input used by the original-feature baseline. File hashes identify the versions included in this distribution.

Large raw data and NPZ files should be distributed as a companion release asset or through a suitable large-file data host; do not assume an ordinary Git commit can hold all files. `DATA_SOURCES.md` records sources and ownership. Verify upstream redistribution terms before publishing bundled raw data.

## Methods and deterministic settings
The 64-feature input contains 55 original features, PageRank, sampled directed betweenness, weak component size, and six directed-hypergraph features. The ordered manifest is in data/processed.

The 70/30 supervised split retains row order with no shuffling. The data-preparation stage preserves address IDs, replaces non-finite values by zero, applies the documented full-known-sample MinMax transform and reconstructs the matrices. The classification stage uses the supplied numerical matrices to preserve their floating-point representation. Labels encode illicit as 1 and licit as 0. Unknown addresses are scored separately.

Random states are fixed at 42 for the distributed training implementation. The fragility bootstrap uses NumPy seed 42. Intermediary score ties are resolved by address order.

## Numerical reproduction scope
The enriched four-model classification metrics, unknown-address probability distribution, motif counts and risk-ranked fragility outputs have been checked using the canonical inputs. Small floating-point differences in probabilities are handled separately from equality of rounded table values.

## File layout
- `run.py`: unified experiment command.
- `src/prepare.py`: preprocessing, scaling, splitting and validation.
- `src/baseline.py`: original-feature classification.
- `src/build_structural_features.py`, `src/hypergraph.py`: structural feature computation.
- `src/fragility.py`: motif enumeration, progressive removal, bootstrap and tests.
- `data/`: raw, enriched and numerical inputs.

## Dataset citation
Youssef Elmougy and Ling Liu (2023). Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics. KDD 2023. https://doi.org/10.1145/3580305.3599803
