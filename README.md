# Retail Customer Segmentation & Neural Collaborative Filtering Recommender

A machine learning pipeline that segments retail customers using RFM analysis and K-Means clustering, then trains a Neural Collaborative Filtering (NCF) model to generate personalised product recommendations.

---

## Overview

This project is built on the [Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and consists of two main components:

1. **Customer Segmentation** — Customers are segmented into 4 clusters based on their Recency, Frequency, and Monetary (RFM) behaviour using K-Means clustering.
2. **Recommendation System** — A Neural Collaborative Filtering model uses user embeddings, item embeddings, and cluster embeddings to predict the likelihood of a customer purchasing a given product, then surfaces the top-K recommendations.

---

## Project Structure

```
├── data/
│   └── online_retail_II_RAW.csv       # Raw dataset (not tracked in git)
├── data_processed/
│   ├── cleaned_retail_data.csv        # Cleaned data with encoded user/item IDs
│   ├── rfm.csv                        # RFM features per customer
│   └── interactions.csv               # Positive + negative sampled interactions
├── model_weights/
│   └── ncf_model.pth                  # Trained NCF model weights
├── plots/                             # Generated visualisations
├── 01_data_cleaning+rfm.ipynb         # Data cleaning & RFM feature engineering
├── 02_cluster_analysis.ipynb          # K-Means clustering on RFM features
├── 03_neg_samp+interaction_matrix.ipynb  # Negative sampling & interaction matrix
├── 04_ncf_model.ipynb                 # NCF model training & evaluation
├── 05_baseline_logistic_reg.ipynb     # Baseline logistic regression model
├── 06_main.ipynb                      # Generate recommendations for a customer
├── model.py                           # NCF model architecture
├── config.py                          # NUM_USERS, NUM_ITEMS constants
└── requirements.txt
```

---

## Pipeline

Run the notebooks in order:

### `01_data_cleaning+rfm.ipynb`
- Loads the raw dataset (1,067,371 rows)
- Drops missing Customer IDs, non-numeric invoices, and negative quantities/prices
- Engineers RFM features (Recency, Frequency, Monetary value) per customer
- Outputs: `data_processed/cleaned_retail_data.csv`, `data_processed/rfm.csv`

### `02_cluster_analysis.ipynb`
- Scales RFM features and selects optimal K using the elbow method and silhouette scores
- Fits K-Means with **K=4** clusters
- Outputs: cluster assignments appended to `rfm.csv`, cluster plots in `plots/`

### `03_neg_samp+interaction_matrix.ipynb`
- Encodes `Customer ID` and `StockCode` to integer `user_id` / `item_id`
- Builds a positive interaction matrix (purchased = 1)
- Applies **4:1 negative sampling** (items the user has not bought) to balance the dataset
- Outputs: `data_processed/interactions.csv`

### `04_ncf_model.ipynb`
- Splits interactions into 80/20 train/test sets
- Trains the NCF model (see architecture below) with BCE loss and Adam optimiser
- Evaluates using **Hit Rate @ 10** and **NDCG** metrics
- Saves weights to `model_weights/ncf_model.pth`

### `05_baseline_logistic_reg.ipynb`
- Trains a logistic regression baseline using one-hot encoded user/item/cluster features
- Reports classification metrics and ROC-AUC for comparison with the NCF model

### `06_main.ipynb`
- Loads the trained NCF model and interaction data
- Given a `user_id`, masks all items the user has already purchased (in train)
- Returns the **top-10 recommended product descriptions**

---

## Model Architecture

```
NCF(
  (user_embedding):    Embedding(5878, 64)
  (item_embedding):    Embedding(4631, 64)
  (cluster_embedding): Embedding(4, 8)
  (fc1): Linear(136 → 64)
  (fc2): Linear(64  → 32)
  (fc3): Linear(32  → 16)
  (output): Linear(16 → 1)
  (activation): ReLU + Dropout(0.2)
  (output_activation): Sigmoid
)
```

User, item, and cluster embeddings are concatenated (64 + 64 + 8 = 136) and passed through three fully-connected layers with ReLU activations and dropout, before a sigmoid output predicts purchase probability.

