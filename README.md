# scTRaCT v0.1.0

**scTRaCT** is a supervised transformer-based deep learning framework that integrates log-normalized gene expression with complementary distance-based features derived from Multiple Correspondence Analysis (MCA). By transforming continuous expression data into a metric that quantifies the association between genes and cells, scTRaCT enriches the input representation, capturing subtle transcriptional differences that are critical for distinguishing closely related cell types. Processed through a Transformer-based architecture with self-attention mechanisms, our model effectively learns long-range dependencies and complex gene interactions

![](Images/scTRaCT_overview.png)

---

##  Key Features

-  Combines log-normalized gene expression with MCA-based distance features.
-  Transformer-based architecture with multi-head self-attention.
-  Handles class imbalance using Focal Loss.
-  Easily customizable embedding dimensions, attention heads, and training parameters.
-  Open-source and easy to use on any preprocessed `.h5ad` dataset.

---

## Installation
Clone the repository and install using pip:

Make sure to create a virtual environment and install required dependencies first:

```bash
conda create -n scTRaCT python=3.10 -y
conda activate scTRaCT
pip install -r requirements.txt
```
And then: 

```bash
pip install git+https://github.com/msmalmir/scTRaCT.git
```
OR

```bash
git clone https://github.com/msmalmir/scTRaCT.git
cd scTRaCT
pip install 
```
---

## Usage
### Step 1: Prepare Input Data
Your AnnData object should contain both train and test cells. Add a `is_train_key` column to `adata.obs` for example `is_train` where `"train"` indicates training cells and other values indicate test/query cells. You should also specify the key for cell type annotations `cell_type_key`(e.g. `cell_type`). 

```python
from scTRaCT import prepare_data
X_train_counts, X_train_dist, y_train, X_val_counts, X_val_dist, y_val, label_encoder = prepare_data(
    adata,
    lognorm_layer="lognorm",
    distance_layer="distance_matrix",
    cell_type_key="cell_type",
    is_train_key="is_train",
    j=30
)
```

### Step 2: Initialize and Train the Model
``` python
from scTRaCT import TransformerModel, FocalLoss, train_model

model = TransformerModel(
    num_genes=X_train_counts.shape[1],
    num_classes=len(label_encoder.classes_),
    num_heads=8,
    dim_feedforward=2048,
    dropout=0.1,
    embedding_dim=1024
)

criterion = FocalLoss(gamma=2.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

train_model(
    model, train_loader, val_loader,
    criterion, optimizer,
    num_epochs=50,
    save_dir="saved_models",
    save_every=10,
    save_name="scTRaCT_checkpoint"
)
```
### Step 3: Evaluate on Test Set
```python
from scTRaCT import predict_query
query_adata, acc, f1, predictions = predict_query(
    adata,
    checkpoint_path="saved_models/scTRaCT_checkpoint_epoch50.pth",
    lognorm_layer="lognorm",
    distance_layer="distance_matrix",
    cell_type_key="cell_type",
    is_train_key="is_train",
    label_encoder=label_encoder,
    prediction_key="predicted_celltypes"
)
```
---

## Tutorial
For a complete usage example pleae refer to **Tutorial** folder in this GitHub repository, you can directly access it from [here.](https://github.com/msmalmir/scTransID/tree/main/Tutorial)

---

## Output
- Saved model checkpoints in the saved_models/ directory

- Accuracy and F1 score for the query/test set

- Updated query_adata object with predicted labels

---

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/msmalmir/scTransID/tree/main/LICENSE.txt) file for more details.

---

## Acknowledgements
scTRaCT builds on top of widely-used libraries like `Scanpy`, `PyTorch`, and `AnnData`.

---

## Contact
For questions, feature requests, or bug reports, please open an issue or contact the repository maintainer at [malmir.edumail@gmail.com].

---