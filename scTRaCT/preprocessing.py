import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
from .mca_utils import RunMCA, GetDistances
from scipy.sparse import issparse

def prepare_data(adata, lognorm_layer="lognorm", distance_layer="distance_matrix", 
                 cell_type_key="cell_type", is_train_key="is_train", j=30):
    """
    Prepares train and validation data from an AnnData object.
    Calculates distance matrix if not available.

    Args:
        adata: AnnData object with train/test cells.
        lognorm_layer: Layer name for log-normalized counts.
        distance_layer: Layer name to store/use distance matrix.
        cell_type_key: Key in .obs for cell type labels.
        is_train_key: Key in .obs to distinguish train/test.
        j: Dimensions for MCA (default 30).

    Returns:
        X_train_counts, X_train_dist, y_train, X_val_counts, X_val_dist, y_val, LabelEncoder object
    """

    # If distance matrix not calculated, do it
    if distance_layer not in adata.layers.keys():
        print("Calculating distance matrix using MCA...")
        norm_matrix_df = pd.DataFrame(
            adata.layers[lognorm_layer].toarray(),
            index=adata.obs_names,
            columns=adata.var_names
        )
        mca_result = RunMCA(norm_matrix_df, j=j)
        distance_matrix = GetDistances(
            cellCoordinates=mca_result.cellCoordinates,
            geneCoordinates=mca_result.geneCoordinates
        )
        adata.layers[distance_layer] = distance_matrix.T.copy()

    # Split into train and validation
    train_adata = adata[adata.obs[is_train_key] == 'train'].copy()
    
    # Features
    X_counts = train_adata.layers[lognorm_layer].toarray()

    if issparse(train_adata.layers[distance_layer]):
        X_dist = train_adata.layers[distance_layer].toarray()
    else:
        X_dist = train_adata.layers[distance_layer]

    # Inverse of distance
    epsilon = 1e-6
    X_dist = 1 / (X_dist + epsilon)

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_adata.obs[cell_type_key])
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_counts = torch.tensor(X_counts, dtype=torch.float32)
    X_dist = torch.tensor(X_dist, dtype=torch.float32)

    # Split
    X_train_counts, X_val_counts, X_train_dist, X_val_dist, y_train, y_val = train_test_split(
        X_counts, X_dist, y_train, test_size=0.2, random_state=42
    )

    return X_train_counts, X_train_dist, y_train, X_val_counts, X_val_dist, y_val, le
