# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FREEDOM is a multimodal graph neural network recommendation framework. This implementation extends the original ACM MM'23 paper ("A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation") for tsRNA-disease association prediction, using a two-stage FREEDOM + MLP pipeline.

## Common Commands

### Environment Setup (Pixi)
```bash
pixi install                    # Install dependencies
```

### Training
```bash
cd src
python main.py -m FREEDOM -d <dataset>           # Basic training (e.g., baby, sports, tsRNA-disease2)
python cross_validation.py -d tsRNA-disease2 --k 5 --gpu 0   # 5-fold cross validation
python denovo_cross_validation.py -d tsRNA-disease2 --start_fold 1 --end_fold 92  # Leave-one-disease validation
python case_study.py -d tsRNA-disease2 --disease "Breast Neoplasms" --top_k 10 20  # Case study
```

### MLP Stage (After FREEDOM Training)
```bash
cd src
python train_mlp.py -d tsRNA-disease2 --embeddings ../data/tsRNA-disease2/freedom_embeddings.pt
```

## Architecture

### Two-Stage Pipeline
1. **FREEDOM (Stage 1)**: GNN learns user/item embeddings from interaction graph + multimodal features
2. **MLP Predictor (Stage 2)**: Binary classification using frozen FREEDOM embeddings (3-layer MLP with halving hidden dimensions: `[2*dim -> dim -> dim/2 -> 1]`)

### Model Core Components (models/freedom.py)
- **Dual Graph Structure**: Multimodal item graph (`mm_adj`, KNN-based) + user-item interaction graph (`norm_adj`)
- **Edge Pruning**: Degree-sensitive dropout in `pre_epoch_processing()`
- **Hard Negative Mixup**: Enabled via `use_hard_neg_mixup`, generates hard negatives through interpolation (supports SVD-based and simple linear mixup)

### Configuration Hierarchy
Configurations are loaded by priority (later overrides earlier):
1. `src/configs/overall.yaml` - Global defaults
2. `src/configs/dataset/<dataset>.yaml` - Dataset-specific paths
3. `src/configs/model/FREEDOM.yaml` - Model hyperparameters

The `hyper_parameters` list in config defines parameters to search (set values as arrays for grid search).

### Data Flow
```
RecDataset (utils/dataset.py)
  -> Load .inter file (userID, itemID, x_label columns)
  -> Split by x_label (0=train, 1=val, 2=test) or kfold_split()
  ->
TrainDataLoader/EvalDataLoader (utils/dataloader.py)
  -> Negative sampling (training_neg_sample_num)
  ->
FREEDOM model
  -> pre_epoch_processing(): Edge pruning
  -> calculate_loss(): BPR + regularization
  -> save_embeddings(): Save freedom_embeddings.pt
```

## Key Files

| File | Purpose |
|------|---------|
| `src/main.py` | Program entry point |
| `src/cross_validation.py` | 5-fold cross validation (two-stage pipeline) |
| `src/denovo_cross_validation.py` | Leave-one-disease validation experiments |
| `src/models/freedom.py` | FREEDOM GNN model |
| `src/models/mlp_predictor.py` | Stage 2 MLP classifier |
| `src/common/trainer.py` | Training loop, early stopping |
| `src/utils/binary_evaluator.py` | Binary classification metrics (AUC, AUPR, etc.) |
| `src/utils/dataset.py` | Data loading (supports K-fold) |

## Data Format

Located in `data/<dataset>/`:
- `<dataset>.inter`: Tab-separated (userID, itemID, x_label columns)
- `text_feat.npy`, `image_feat.npy`: Pre-extracted features
- `freedom_embeddings.pt`: Saved embeddings after FREEDOM training
- `mm_adj_freedomdsp_*.pt`: Cached multimodal adjacency matrices (delete after modifying KNN parameters)

## Modifying the Model

1. Add new config parameters to `src/configs/model/FREEDOM.yaml`
2. Add to `hyper_parameters` list for grid search
3. Modify loss function in `calculate_loss()` method
4. Graph computations use sparse tensor operations

## Key Hyperparameters

From `src/configs/model/FREEDOM.yaml`:
- `embedding_size`: Embedding dimension
- `knn_k`: KNN neighbors for multimodal graph
- `dropout`: Edge dropout rate
- `use_hard_neg_mixup`: Enable hard negative augmentation
- `mixup_alpha_range`: Mixup interpolation range
- `use_svd_mixup`: Use SVD-based mixup (vs simple linear)

From `src/configs/overall.yaml`:
- `epochs`, `stopping_step`: Training epochs and early stopping
- `training_neg_sample_num`: Negative samples per positive
- `metrics`: ["AUC", "AUPR", "Precision", "Recall", "F1"]

## MLP Stage Training Parameters

From command line arguments:
- `--mlp_epochs`: MLP training epochs (default: 100)
- `--mlp_lr`: Learning rate (default: 0.001)
- `--mlp_batch_size`: Batch size (default: 256)
- `--mlp_dropout`: Dropout rate (default: 0.3)
- `--mlp_stopping_step`: Early stopping steps (default: 10)
