# My Learning Preferences
- This is my PhD project and I am learning Python / ML and Bioinformatics
- Always explain WHY you chose an approach, not just what
- After each change, briefly explain the concept used
- Point out patterns, best practices, and things I should know
- If you use a library or function I may not know, explain it
- Warn me about common mistakes related to the changes you make
- If there are alternative approaches, mention them and the tradeoffs


---

# scTRaCT
scTRaCT is a supervised transformer-based framework for single-cell RNA-seq cell type classification, combining log-normalized gene expression with MCA (Multiple Correspondence Analysis) distance features.

## Installation

```bash
conda create -n scTRaCT python=3.10
conda activate scTRaCT
pip install -r requirements.txt   # captum is now included here
pip install -e .                  # editable install from local source
# or: pip install git+<repo-url>
```

## Architecture

- **Two input streams**: count embeddings (log-normalized expression) + distance embeddings (MCA-derived distances)
- **3-layer Transformer** with multi-head self-attention, CLS token, residual connections, LayerNorm
- **FocalLoss** for handling class imbalance
- **AnnData (.h5ad)** as the primary data format with layers: `lognorm`, `distance_matrix`

## Module Responsibilities

| Module | Key Exports | Purpose |
|--------|-------------|---------|
| `preprocessing.py` | `prepare_data()` | Splits AnnData into train/val, computes MCA if needed |
| `mca_utils.py` | `RunMCA()`, `GetDistances()` | MCA algorithm and distance computation |
| `model.py` | `TransformerModel`, `FocalLoss` | Model architecture and loss function |
| `trainer.py` | `train_model()`, `evaluate_model()`, `evaluate_on_query()` | Training and evaluation loops |
| `utils.py` | `compute_metrics()` | Accuracy + macro-F1 metrics |
| `interpretability.py` | `get_gene_attributions()`, `explain_celltype()`, `explain_all_celltypes()`, `plot_gene_attributions()` | XAI: Integrated Gradients + GradientSHAP via `captum` |

## Typical Workflow

```python
from scTRaCT import (prepare_data, TransformerModel, FocalLoss,
                     train_model, evaluate_on_query,
                     explain_celltype, explain_all_celltypes)

# 1. Prepare data
X_tr_c, X_tr_d, y_tr, X_val_c, X_val_d, y_val, label_encoder = prepare_data(adata)

# 2. Build model
model = TransformerModel(input_dim=..., num_classes=...)
criterion = FocalLoss(gamma=2)

# 3. Train
train_model(model, criterion, X_tr_c, X_tr_d, y_tr, X_val_c, X_val_d, y_val)

# 4. Evaluate on query data
acc, f1, preds, query_adata = evaluate_on_query(model_path, adata_query, label_encoder)
query_adata.obs['predicted_celltypes'] = preds  # required for XAI step

# 5. Explain predictions (XAI)
# Single cell type:
df = explain_celltype(model, query_adata, label_encoder, cell_type="CD4 T")

# All predicted cell types at once:
results = explain_all_celltypes(model, query_adata, label_encoder,
                                method="both",   # "IG", "SHAP", or "both"
                                save_dir="attribution_plots/")
# Results also stored in: query_adata.uns['gene_attributions']
```

## Key Dependencies

- `torch>=2.0`
- `scanpy>=1.9`
- `anndata>=0.9`
- `scikit-learn>=1.1`
- `scvelo>=0.2.5` (RNA velocity)
- `captum>=0.6` (interpretability — now in requirements.txt)

---

## Interpretability Module — Extended Reference

### Functions

| Function | Purpose |
|----------|---------|
| `get_gene_attributions(model, adata, label_encoder, target_class, ...)` | **Original function — uses true labels (`obs['cell_type']`).** Returns DataFrame with IG_Score + SHAP_Score for one class. Keep unchanged. |
| `explain_celltype(model, adata, label_encoder, cell_type, ...)` | **New.** Uses predicted labels. Single cell type. Returns DataFrame, stores in `adata.uns`, optionally plots. |
| `explain_all_celltypes(model, adata, label_encoder, ...)` | **New.** Loops over all unique predicted types. Returns dict of DataFrames. Generates bar charts + heatmap. Saves PNGs + CSVs if `save_dir` set. |
| `plot_gene_attributions(results_df, cell_type, method, top_n, ...)` | **New.** Standalone horizontal bar chart from any attribution DataFrame. Returns Figure. |

### Key parameter notes
- `method`: `"IG"` (Integrated Gradients), `"SHAP"` (GradientSHAP), or `"both"` — default is `"both"`
- `num_cells`: cells sampled per predicted type before averaging (default 50; uses all if fewer exist)
- `n_steps`: IG integration steps (default 50; higher = more accurate, slower)
- `predicted_key`: the `adata.obs` column holding model predictions (default `"predicted_celltypes"`)
- `device`: **never passed by user** — auto-detected (`cuda` if available, else `cpu`)

### Output structure
```
adata.uns['gene_attributions']
└── "CD4 T"  → DataFrame: gene | IG_Score | SHAP_Score (sorted descending)
└── "CD8 T"  → DataFrame
└── ...

save_dir/
├── <CellType>_attributions.csv       # full gene list
├── <CellType>_attributions_IG.png    # bar chart, top_n genes
├── <CellType>_attributions_SHAP.png
├── summary_heatmap_IG.png            # all types × top genes
└── summary_heatmap_SHAP.png
```

### Important distinctions
- `get_gene_attributions()` uses `obs['cell_type']` (ground truth) — for training-time analysis
- `explain_celltype()` uses `obs['predicted_celltypes']` (model output) — for justifying predictions on new data
- Attention weights are **not used** for gene attribution: the model has only 2 sequence tokens (CLS + gene embedding), so there is no per-gene attention weight
- GradientSHAP background = 100 randomly sampled cells from the full `adata`; do not reuse the same background tensor across batches of different sizes

---

## Plans & Design Decisions Log

Plans for major features are saved in `plans/` as markdown files.

| Plan file | Feature | Status |
|-----------|---------|--------|
| `plans/explainability_xai_plan.md` | XAI gene attribution (IG + SHAP) | Implemented 2026-03-13 |
