# Let Bridges Speak: GLCR for Tripartite Biomedical Association Prediction

[中文说明](README_zh.md)

This repository contains the code and processed benchmark inputs for **GLCR** in the paper:

**Let Bridges Speak: Global-Local Collaborative Reasoning for Tripartite Biomedical Association Prediction**

## Overview

Tripartite biomedical association prediction aims to infer a target relation between two endpoint entity types by leveraging a mediator layer, such as phenotypes, diseases, or other biological intermediates. Typical examples include ncRNA-drug association prediction and drug-target interaction prediction with disease-side mediator evidence.

GLCR addresses this problem with a **global-local collaborative reasoning** framework. The model first builds a global structural anchor for each candidate endpoint pair from the full tripartite graph, then organizes mediator evidence as pair-conditioned bridge tokens, and finally fuses global and local evidence through explicit communication, latent compensation, reliability calibration, and a single adaptive utility gate.

This release provides the implementation and processed inputs used to reproduce the **main benchmark setting** of the paper on four datasets.

## Included Benchmarks

The repository includes four tripartite biomedical benchmarks. In all cases, the model predicts the endpoint relation in the `primary_edge` file and uses the other two relations as mediator-side support edges.

| Dataset key | Endpoint relation | Mediator-side support relations |
| --- | --- | --- |
| `ncRNADrug` | ncRNA-drug | ncRNA-phenotype, drug-phenotype |
| `openTargets` | drug-target | drug-disease, target-disease |
| `CTD` | chemical-gene | chemical-disease, gene-disease |
| `PrimeKG` | drug-target | drug-disease, target-disease |

Each dataset directory contains:

- `all_node_features.csv`: unified node feature table with global node indices and typed numeric descriptors
- one primary endpoint relation file
- two mediator-side support relation files

## Method Summary

GLCR is built around the idea that tripartite prediction should not rely on global topology or local mediator evidence in isolation.

- **Global structural modeling** learns graph-aware node representations and forms a pair-level global anchor for each candidate endpoint pair.
- **Local mediator evidence modeling** converts shared mediator candidates into pair-conditioned bridge tokens and aggregates them into an explicit local reasoning state.
- **Latent residual compensation** recovers bridge patterns that may be incomplete in observed mediator candidates.
- **Reliability calibration** estimates whether available local evidence should be trusted for the current pair.
- **Two-token micro-subgraph communication** summarizes local evidence with an explicit bridge token and a pre-communication local-seed token before global-local interaction.
- **Utility-aware adaptive fusion** uses one shared gate to control both the coarse bridge term and the communication-induced local residual.
- **Training objective** follows the paper's fixed three-term formulation: final classification, bridge calibration regularization, and bridge perturbation. The release code does not expose extra auxiliary losses from older development variants.

## Repository Structure

```text
GLCR_open_source/
├── train_glcr.py
├── paper_configs.py
├── requirements.txt
├── model/
│   ├── GLCR.py
│   └── __init__.py
├── layer/
│   ├── GCN.py
│   └── __init__.py
├── tools/
│   ├── Datasets.py
│   ├── subgraph.py
│   ├── utils.py
│   └── __init__.py
└── datasets/
    ├── ncRNADrug/
    ├── openTargets/
    ├── CTD/
    └── PrimeKG/
```

## Environment

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The implementation was prepared and verified with the following core package versions:

- `torch 2.6.0`
- `torch-geometric 2.7.0`
- `numpy 1.26.4`
- `pandas 3.0.1`
- `scikit-learn 1.8.0`

## Reproducing the Main Experiments

The released training script reproduces the main benchmark protocol in this repository:

- random split evaluation
- balanced `1:1` positive/negative sampling
- `5`-fold cross-validation
- validation split inside each training fold
- automatic threshold selection on the validation split using `MCC`
- one unified release configuration shared by all four datasets

Run a full experiment with:

```bash
python train_glcr.py \
  --dataset openTargets \
  --result_dir results/openTargets_main
```

Available dataset keys are:

- `ncRNADrug`
- `openTargets`
- `CTD`
- `PrimeKG`

You can run a subset of folds if needed:

```bash
python train_glcr.py \
  --dataset CTD \
  --fold_indices 1,2 \
  --result_dir results/CTD_partial
```

## Training and Evaluation Protocol

For each fold, the support graph is constructed from:

- training-fold positives of the target endpoint relation
- all observed mediator-side support edges

Validation and test positives are removed from the support graph before graph encoding and pair-context extraction. The model is trained on balanced endpoint pairs, selected from the observed positives and unobserved candidate negatives. Evaluation reports ranking and decision metrics including:

- `AUC`
- `AUPR`
- `F1`
- `MCC`
- `Accuracy`
- `Precision`
- `Recall`

## Outputs

If `--result_dir` is specified, the script writes:

- `fold_metrics.csv`: per-fold metrics together with mean and standard deviation
- `summary.json`: structured summary of the run, including metadata and aggregate metrics

During training, logs are written to `logs/`, and cached split/context payloads are written to `cache/subgraph_features/`.

## Configuration

The release defaults are defined in [paper_configs.py](paper_configs.py). The released implementation uses a unified configuration for the four included benchmarks and applies the same optimization, sampling, and threshold-selection protocol across datasets. This release tracks the streamlined paper version with a **two-token micro-subgraph summary** and a **single adaptive utility gate**.

## Notes

- This repository corresponds to the GLCR model described in the paper.
- The included dataset directories contain the processed inputs consumed directly by the released training code.
- The current release is focused on the main benchmark reproduction pipeline provided in this repository.

## Citation

If you find this repository useful, please cite:

```text
Let Bridges Speak: Global-Local Collaborative Reasoning for Tripartite Biomedical Association Prediction
```
