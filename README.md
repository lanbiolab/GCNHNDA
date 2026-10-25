
## Common Commands

### Environment Setup (Pixi)
```bash
pixi install                    # Install dependencies
```

### Training
```bash
cd src
python cross_validation.py -d tsRNA-disease2 --k 5 --gpu 0   # 5-fold cross validation
python denovo_cross_validation.py -d tsRNA-disease2 --start_fold 1 --end_fold 92  # Leave-one-disease validation
python case_study.py -d tsRNA-disease2 --disease "Breast Neoplasms" --top_k 10 20  # Case study
```

## Architecture


### Model Core Components (models/freedom.py)
- **Dual Graph Structure**: Multimodal item graph (`mm_adj`, KNN-based) + user-item interaction graph (`norm_adj`)
- **Edge Pruning**: Degree-sensitive dropout in `pre_epoch_processing()`
- **Hard Negative Mixup**: Enabled via `use_hard_neg_mixup`, generates hard negatives through interpolation (supports SVD-based and simple linear mixup)

## Key Files

| File | Purpose |
|------|---------|
| `src/main.py` | Program entry point |
| `src/cross_validation.py` | 5-fold cross validation (two-stage pipeline) |
| `src/denovo_cross_validation.py` | Leave-one-disease validation experiments |
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

