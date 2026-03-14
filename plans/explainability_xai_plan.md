# Plan: Explainable AI Gene Attribution Feature for scTRaCT

**Status: Implemented (2026-03-13)**

---

## Context
After cell type classification with `evaluate_on_query()`, users need to justify the model's predictions — which genes drove each cell type assignment. The goal is to provide per-cell-type gene contribution scores using Integrated Gradients (IG) and GradientSHAP, with publication-ready visualizations and exportable tables for use in research papers comparing both methods.

`interpretability.py` already had `get_gene_attributions()` (IG + GradientSHAP, no plots, single cell type, uses true labels). Three new functions were built on top of it.

---

## Finalized Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| XAI methods | User-selectable: `"IG"`, `"SHAP"`, or `"both"` | Professor requested SHAP; comparing IG vs SHAP in paper |
| Cell selection | **Predicted labels** (`adata.obs['predicted_celltypes']`) | Justify the model's own predictions on query data |
| Summary output | Heatmap + bar charts (per type) + DataFrame/CSV tables | Complete output; user will decide what to trim |
| Device | **Auto-detect GPU** (no `device` param) | Consistent with trainer.py; user never has to think about it |
| num_cells default | **50** | Enough for stable averages; still fast on CPU (~10–20 s per type) |

---

## What Inputs Are Required

All three inputs are already in hand after the standard workflow:

```python
# After training:
X_tr_c, X_tr_d, y_tr, X_val_c, X_val_d, y_val, label_encoder = prepare_data(adata)
train_model(...)

# After prediction:
acc, f1, preds, query_adata = evaluate_on_query(adata, checkpoint_path, label_encoder)
query_adata.obs['predicted_celltypes'] = preds  # ← user adds this one line

# XAI — all inputs already available:
explain_celltype(model, query_adata, label_encoder, cell_type="CD4 T")
```

| Input | Type | Where it comes from |
|-------|------|---------------------|
| `model` | `TransformerModel` | Loaded/trained model object |
| `adata` | `AnnData` | Returned by `evaluate_on_query()`; must have `lognorm` and `distance_matrix` layers, and `obs['predicted_celltypes']` |
| `label_encoder` | `sklearn.LabelEncoder` | Returned by `prepare_data()` |

---

## What `num_cells` Does

IG and SHAP compute a gene importance score **per cell**. We then average scores across multiple cells of the same predicted type to get one stable score per gene. `num_cells` = how many cells of each predicted type to sample before averaging.

- **Low (10–20)**: Fast, but noisier — a few atypical cells can skew the result.
- **High (50–100)**: More stable, more representative of the cell type. Slightly slower.
- **Default: 50** — if a cell type has fewer than 50 predicted cells, all of them are used.

---

## Device Handling (GPU Auto-detect)

`device` is not a user parameter. Internally:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Consistent with `trainer.py`. GPU provides ~5–10× speedup for large datasets.

---

## XAI Methods

- **Integrated Gradients (IG)**: Integrates gradients along a path from a zero baseline to the actual input. ~3–8 s per cell type on CPU (50 steps). Theoretically grounded (completeness axiom).
- **GradientSHAP**: Uses a distribution of random background cells as baselines. ~2–5 s per cell type. More robust due to multiple baselines.
- Attention weights: NOT applicable — sequence length is only 2 tokens (CLS + combined gene embedding). There is no token per gene.
- Gene occlusion: NOT used — O(num_genes) forward passes. Far too slow for scRNA-seq.

---

## New API

### `explain_celltype()` — single cell type
```python
def explain_celltype(model, adata, label_encoder, cell_type,
                     method="both",         # "IG", "SHAP", or "both"
                     num_cells=50,          # cells to sample per type (uses all if fewer exist)
                     n_steps=50,            # IG integration steps
                     top_n=20,              # genes shown in plot
                     lognorm_layer="lognorm",
                     distance_layer="distance_matrix",
                     predicted_key="predicted_celltypes",
                     plot=True,
                     save_path=None):
    # Device: auto-detected internally
    # Returns: pd.DataFrame → columns [gene, IG_Score] or [gene, SHAP_Score] or [gene, IG_Score, SHAP_Score]
    # Side effect: stores result in adata.uns['gene_attributions'][cell_type]
```

### `explain_all_celltypes()` — all predicted cell types automatically
```python
def explain_all_celltypes(model, adata, label_encoder,
                          method="both",
                          num_cells=50,
                          n_steps=50,
                          top_n=20,
                          lognorm_layer="lognorm",
                          distance_layer="distance_matrix",
                          predicted_key="predicted_celltypes",
                          plot=True,
                          save_dir=None):
    # Device: auto-detected internally
    # Returns: dict[cell_type_name → pd.DataFrame]
    # Side effect: populates adata.uns['gene_attributions'] for all types
    #              generates per-type bar charts + summary heatmap
    #              saves CSVs if save_dir set
```

### `plot_gene_attributions()` — standalone plot helper
```python
def plot_gene_attributions(results_df, cell_type, method="IG", top_n=20,
                           figsize=(8, 6), save_path=None):
    # Horizontal bar chart of top_n genes
    # Returns: matplotlib Figure
```

---

## Where Results Are Stored

### In memory — `adata.uns['gene_attributions']`

```python
adata.uns['gene_attributions']
# → {
#     "CD4 T":  DataFrame,
#     "CD8 T":  DataFrame,
#     "B cell": DataFrame,
#     ...
# }
```

Each DataFrame (sorted by IG_Score descending):
```
          gene  IG_Score  SHAP_Score
0          LYZ    0.0842      0.0791
1        S100A8   0.0731      0.0688
2        FCGR3A   0.0612      0.0579
3          CST3   0.0558      0.0521
...
```

### On disk (when `save_dir` is set)

```
attribution_plots/
├── CD4 T_attributions.csv
├── CD8 T_attributions.csv
├── B cell_attributions.csv
├── CD4 T_attributions_IG.png
├── CD8 T_attributions_IG.png
├── B cell_attributions_IG.png
├── CD4 T_attributions_SHAP.png
├── summary_heatmap_IG.png
└── summary_heatmap_SHAP.png
```

---

## Runtime Estimates (CPU)
| Scenario | Estimate |
|----------|----------|
| 1 cell type, IG only, 50 cells, 50 steps | ~5–12 s |
| 1 cell type, IG + SHAP, 50 cells | ~8–18 s |
| 10 cell types, IG + SHAP, 50 cells each | ~2–3 min |
| GPU (CUDA) | ~5–10× faster |

---

## Files Modified

| File | Change |
|------|--------|
| `scTRaCT/interpretability.py` | Added `_to_dense()`, `plot_gene_attributions()`, `explain_celltype()`, `explain_all_celltypes()`. Kept existing `get_gene_attributions()` unchanged. |
| `requirements.txt` | Added `captum>=0.6` |
| `scTRaCT/__init__.py` | Added explicit import of all four interpretability functions |

---

## Verification Checklist
- [ ] Run `explain_celltype()` on BMMC tutorial dataset → bar chart renders, DataFrame has correct columns and gene names
- [ ] Run `explain_all_celltypes()` → heatmap renders with all predicted cell types, CSVs saved
- [ ] Check `adata.uns['gene_attributions']` is populated for each type
- [ ] Confirm graceful warning when a cell type has 0 predicted cells
- [ ] Confirm `captum` appears in `requirements.txt`
- [ ] Confirm existing `get_gene_attributions()` still works unchanged
- [ ] Confirm GPU is used automatically if CUDA is available
