import torch
import pandas as pd
import numpy as np
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
