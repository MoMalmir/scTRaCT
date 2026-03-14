import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients, GradientShap

def get_gene_attributions(model, adata, label_encoder, target_class, 
                           lognorm_layer="lognorm", distance_layer="distance_matrix",
                           num_cells=20, n_steps=50, device="cpu"):
    """
    Calculates both Integrated Gradients and Gradient SHAP for a specific cell type.
    """
    model.eval()
    model.to(device)
    
    # 1. Identify Target Cells
    target_idx = list(label_encoder.classes_).index(target_class)
    cell_indices = np.where(adata.obs['cell_type'] == target_class)[0]
    
    # Sample a few cells to keep computation manageable
    if len(cell_indices) > num_cells:
        cell_indices = np.random.choice(cell_indices, num_cells, replace=False)
    
    subset = adata[cell_indices]
    
    # 2. Prepare Inputs
    counts = torch.tensor(subset.layers[lognorm_layer].toarray(), dtype=torch.float32).to(device)
    dist = subset.layers[distance_layer]
    if hasattr(dist, "toarray"): dist = dist.toarray()
    
    # Inverse distance as per your model's preprocessing
    epsilon = 1e-6
    dist = torch.tensor(1 / (dist + epsilon), dtype=torch.float32).to(device)
    
    # 3. Create Baselines (Required for both IG and SHAP)
    # A common baseline is 'zero expression' or the mean of the dataset
    count_baseline = torch.zeros_like(counts).to(device)
    dist_baseline = torch.zeros_like(dist).to(device)
    
    # --- METHOD 1: Integrated Gradients ---
    ig = IntegratedGradients(model)
    attr_ig, _ = ig.attribute(inputs=(counts, dist), 
                               baselines=(count_baseline, dist_baseline),
                               target=target_idx, 
                               n_steps=n_steps, 
                               return_convergence_delta=True)
    
    # --- METHOD 2: Gradient SHAP ---
    # SHAP benefits from a distribution of baselines (randomly sampled cells)
    gs = GradientShap(model)
    # Using the first 100 cells of the original adata as background distribution
    bg_indices = np.random.choice(len(adata), min(100, len(adata)), replace=False)
    bg_counts = torch.tensor(adata[bg_indices].layers[lognorm_layer].toarray(), dtype=torch.float32).to(device)
    bg_dist = torch.tensor(1 / (adata[bg_indices].layers[distance_layer].toarray() + epsilon), dtype=torch.float32).to(device)
    
    attr_shap = gs.attribute(inputs=(counts, dist),
                             baselines=(bg_counts, bg_dist),
                             target=target_idx)

    # 4. Aggregate Results
    # We use the mean of absolute values across the sampled cells
    scores_ig = np.mean(np.abs(attr_ig[0].cpu().detach().numpy()), axis=0)
    scores_shap = np.mean(np.abs(attr_shap[0].cpu().detach().numpy()), axis=0)
    
    results = pd.DataFrame({
        'gene': adata.var_names,
        'IG_Score': scores_ig,
        'SHAP_Score': scores_shap
    }).sort_values(by='IG_Score', ascending=False)

    return results


# ---------------------------------------------------------------------------
# Helper: convert a layer to a dense numpy float32 array
# ---------------------------------------------------------------------------
def _to_dense(layer):
    """Return a (cells × genes) numpy float32 array regardless of sparsity."""
    if hasattr(layer, "toarray"):
        return layer.toarray().astype(np.float32)
    return np.asarray(layer, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. plot_gene_attributions — standalone plotting helper
# ---------------------------------------------------------------------------
def plot_gene_attributions(results_df, cell_type, method="IG", top_n=20,
                           figsize=(8, 6), save_path=None):
    """
    Horizontal bar chart of the top_n genes for a single cell type.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns 'gene' and at least one of 'IG_Score' / 'SHAP_Score'.
    cell_type : str
        Label used in the chart title.
    method : str
        'IG' or 'SHAP' — which score column to plot.
    top_n : int
        Number of genes to display.
    figsize : tuple
        Matplotlib figure size.
    save_path : str or None
        If given, the figure is saved here (PNG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    score_col = "IG_Score" if method == "IG" else "SHAP_Score"
    if score_col not in results_df.columns:
        raise ValueError(f"Column '{score_col}' not found in results_df. "
                         f"Available: {list(results_df.columns)}")

    top = results_df.nlargest(top_n, score_col)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top["gene"][::-1], top[score_col][::-1], color="steelblue")
    ax.set_xlabel(f"Mean |{method} Score|")
    ax.set_title(f"Top {top_n} genes — {cell_type} ({method})")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved bar chart: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# 2. explain_celltype — attribution for a single predicted cell type
# ---------------------------------------------------------------------------
def explain_celltype(model, adata, label_encoder, cell_type,
                     method="both",
                     num_cells=50,
                     n_steps=50,
                     top_n=20,
                     lognorm_layer="lognorm",
                     distance_layer="distance_matrix",
                     predicted_key="predicted_celltypes",
                     plot=True,
                     save_path=None):
    """
    Compute gene attribution scores for cells predicted as `cell_type`.

    Uses Integrated Gradients (IG), GradientSHAP, or both, then averages
    the absolute attribution scores across the sampled cells.  Results are
    stored in ``adata.uns['gene_attributions'][cell_type]``.

    Parameters
    ----------
    model : TransformerModel
        Trained model.
    adata : AnnData
        Query AnnData; must have layers ``lognorm_layer`` and
        ``distance_layer``, and ``obs[predicted_key]`` populated.
    label_encoder : sklearn.LabelEncoder
        Encoder returned by ``prepare_data()``.
    cell_type : str
        The predicted cell-type label to explain.
    method : {'IG', 'SHAP', 'both'}
        Which XAI method(s) to run.
    num_cells : int
        Max cells to sample from this predicted type (uses all if fewer exist).
    n_steps : int
        Number of integration steps for IG (higher → more accurate, slower).
    top_n : int
        Genes shown in the bar chart.
    lognorm_layer : str
        Key in ``adata.layers`` with log-normalised counts.
    distance_layer : str
        Key in ``adata.layers`` with MCA distances.
    predicted_key : str
        Column in ``adata.obs`` that holds the model's predicted labels.
    plot : bool
        Whether to display a bar chart.
    save_path : str or None
        File path (PNG) to save the bar chart.

    Returns
    -------
    pd.DataFrame or None
        Columns: gene + IG_Score and/or SHAP_Score, sorted descending.
        Returns None if no cells with this predicted label are found.
    """
    # --- auto-detect device (consistent with trainer.py) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- find cells predicted as this type ---
    mask = adata.obs[predicted_key] == cell_type
    cell_indices = np.where(mask)[0]

    if len(cell_indices) == 0:
        print(f"[explain_celltype] Warning: no cells predicted as '{cell_type}'. Skipping.")
        return None

    # sample up to num_cells
    if len(cell_indices) > num_cells:
        cell_indices = np.random.choice(cell_indices, num_cells, replace=False)

    subset = adata[cell_indices]

    # --- prepare tensors ---
    epsilon = 1e-6
    counts = torch.tensor(_to_dense(subset.layers[lognorm_layer]),
                          dtype=torch.float32, device=device)
    dist = torch.tensor(1.0 / (_to_dense(subset.layers[distance_layer]) + epsilon),
                        dtype=torch.float32, device=device)

    # zero baselines for IG
    count_baseline = torch.zeros_like(counts)
    dist_baseline  = torch.zeros_like(dist)

    # class index for this cell type
    target_idx = list(label_encoder.classes_).index(cell_type)

    model.eval()
    model.to(device)

    scores_ig   = None
    scores_shap = None

    # --- Integrated Gradients ---
    if method in ("IG", "both"):
        ig = IntegratedGradients(model)
        attr_ig, _ = ig.attribute(
            inputs=(counts, dist),
            baselines=(count_baseline, dist_baseline),
            target=target_idx,
            n_steps=n_steps,
            return_convergence_delta=True,
        )
        # attr_ig is a tuple (counts_attr, dist_attr); we use the counts stream
        scores_ig = np.mean(np.abs(attr_ig[0].cpu().detach().numpy()), axis=0)

    # --- GradientSHAP ---
    if method in ("SHAP", "both"):
        # background = random sample of all cells in adata (not just the subset)
        bg_idx = np.random.choice(len(adata), min(100, len(adata)), replace=False)
        bg_counts = torch.tensor(
            _to_dense(adata[bg_idx].layers[lognorm_layer]),
            dtype=torch.float32, device=device)
        bg_dist = torch.tensor(
            1.0 / (_to_dense(adata[bg_idx].layers[distance_layer]) + epsilon),
            dtype=torch.float32, device=device)

        gs = GradientShap(model)
        attr_shap = gs.attribute(
            inputs=(counts, dist),
            baselines=(bg_counts, bg_dist),
            target=target_idx,
        )
        scores_shap = np.mean(np.abs(attr_shap[0].cpu().detach().numpy()), axis=0)

    # --- build DataFrame ---
    data = {"gene": adata.var_names}
    if scores_ig   is not None: data["IG_Score"]   = scores_ig
    if scores_shap is not None: data["SHAP_Score"] = scores_shap

    sort_col = "IG_Score" if "IG_Score" in data else "SHAP_Score"
    df = pd.DataFrame(data).sort_values(by=sort_col, ascending=False).reset_index(drop=True)

    # --- store in adata.uns ---
    adata.uns.setdefault("gene_attributions", {})[cell_type] = df

    # --- optional plot ---
    if plot:
        plot_method = "IG" if method in ("IG", "both") else "SHAP"
        fig = plot_gene_attributions(df, cell_type, method=plot_method,
                                     top_n=top_n, save_path=save_path)
        plt.show()

    return df


# ---------------------------------------------------------------------------
# 3. explain_all_celltypes — attribution for every predicted cell type
# ---------------------------------------------------------------------------
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
    """
    Compute gene attributions for **every** predicted cell type in ``adata``.

    Calls ``explain_celltype()`` for each unique label in
    ``adata.obs[predicted_key]``, then (optionally) generates:

    * per-type bar charts
    * a summary heatmap (IG scores and/or SHAP scores across all types)
    * CSV tables per cell type

    Parameters
    ----------
    model : TransformerModel
    adata : AnnData
    label_encoder : sklearn.LabelEncoder
    method : {'IG', 'SHAP', 'both'}
    num_cells : int
    n_steps : int
    top_n : int
        Genes shown in bar charts *and* heatmap columns.
    lognorm_layer : str
    distance_layer : str
    predicted_key : str
    plot : bool
        Show bar charts and heatmap interactively.
    save_dir : str or None
        Directory to save all PNGs and CSVs.  Created automatically if
        it does not exist.

    Returns
    -------
    dict[str, pd.DataFrame]
        Maps each predicted cell-type label to its attribution DataFrame.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    cell_types = sorted(adata.obs[predicted_key].dropna().unique())
    print(f"[explain_all_celltypes] Running attributions for {len(cell_types)} cell types "
          f"using method='{method}' …")

    results = {}

    for ct in cell_types:
        print(f"  → {ct}")
        df = explain_celltype(
            model, adata, label_encoder, ct,
            method=method, num_cells=num_cells, n_steps=n_steps, top_n=top_n,
            lognorm_layer=lognorm_layer, distance_layer=distance_layer,
            predicted_key=predicted_key,
            plot=False,          # suppress per-type plots here; we do them below
            save_path=None,
        )
        if df is not None:
            results[ct] = df

    if not results:
        print("[explain_all_celltypes] No results to plot (no cells found for any type).")
        return results

    # ------------------------------------------------------------------ plots
    methods_to_plot = []
    if method in ("IG",   "both"): methods_to_plot.append("IG")
    if method in ("SHAP", "both"): methods_to_plot.append("SHAP")

    for ct, df in results.items():
        for m in methods_to_plot:
            score_col = f"{m}_Score"
            if score_col not in df.columns:
                continue
            bar_path = os.path.join(save_dir, f"{ct}_attributions_{m}.png") if save_dir else None
            fig = plot_gene_attributions(df, ct, method=m, top_n=top_n,
                                         save_path=bar_path)
            if plot:
                plt.show()
            plt.close(fig)

    # ---------------------------------------------------------------- heatmap
    for m in methods_to_plot:
        score_col = f"{m}_Score"

        # gather the union of top-N genes across all types
        top_genes_union = []
        for df in results.values():
            if score_col in df.columns:
                top_genes_union.extend(df.nlargest(top_n, score_col)["gene"].tolist())
        top_genes_union = list(dict.fromkeys(top_genes_union))  # dedup, preserve order

        # build matrix: rows = cell types, columns = top genes
        heatmap_data = pd.DataFrame(index=list(results.keys()),
                                    columns=top_genes_union, dtype=float)
        for ct, df in results.items():
            if score_col not in df.columns:
                continue
            gene_scores = df.set_index("gene")[score_col]
            for gene in top_genes_union:
                heatmap_data.loc[ct, gene] = gene_scores.get(gene, 0.0)

        heatmap_data = heatmap_data.fillna(0.0).astype(float)

        fig, ax = plt.subplots(figsize=(max(10, len(top_genes_union) * 0.45),
                                        max(4, len(results) * 0.55)))
        sns.heatmap(heatmap_data, ax=ax, cmap="YlOrRd", linewidths=0.3,
                    cbar_kws={"label": f"Mean |{m} Score|"})
        ax.set_title(f"Gene Attribution Heatmap — {m}")
        ax.set_xlabel("Gene")
        ax.set_ylabel("Cell Type")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()

        if save_dir:
            heatmap_path = os.path.join(save_dir, f"summary_heatmap_{m}.png")
            fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
            print(f"  Saved heatmap: {heatmap_path}")

        if plot:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------ CSVs
    if save_dir:
        for ct, df in results.items():
            csv_path = os.path.join(save_dir, f"{ct}_attributions.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Saved CSV: {csv_path}")

    print(f"[explain_all_celltypes] Done. Results stored in adata.uns['gene_attributions'].")
    return results
