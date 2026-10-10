import pandas as pd
import numpy as np
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# Define the Neural Network with Embeddings
class TabularEmbeddingNet(nn.Module):
    def __init__(self, emb_sizes, num_numerical):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(num_cats, emb_dim) for num_cats, emb_dim in emb_sizes])
        total_emb_dim = sum(e.embedding_dim for e in self.embeddings)
        self.fc = nn.Linear(total_emb_dim + num_numerical, 1)
        
    def forward(self, x_cat, x_num):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1) if x else torch.empty((x_cat.shape[0], 0), device=x_cat.device)
        x = torch.cat([x, x_num], 1)
        return self.fc(x).squeeze(1)
    
    def extract_features(self, x_cat, x_num):
        with torch.no_grad():
            x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
            x = torch.cat(x, 1) if x else torch.empty((x_cat.shape[0], 0), device=x_cat.device)
            return torch.cat([x, x_num], 1).cpu().numpy()

def prepare_data(X_train, X_val):
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    # Numeric preprocessing
    num_imputer = SimpleImputer(strategy='median')
    num_scaler = StandardScaler()
    X_train_num = num_scaler.fit_transform(num_imputer.fit_transform(X_train[num_cols]))
    X_val_num = num_scaler.transform(num_imputer.transform(X_val[num_cols]))
    
    # Categorical preprocessing (Ordinal Encoding for PyTorch Embeddings)
    cat_imputer = SimpleImputer(strategy='constant', fill_value='Missing')
    X_train_cat_imp = cat_imputer.fit_transform(X_train[cat_cols])
    X_val_cat_imp = cat_imputer.transform(X_val[cat_cols])
    
    ordinal_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_cat = ordinal_enc.fit_transform(X_train_cat_imp).astype(np.int64)
    # For validation, replace unknown (-1) with 0 (safe fallback) to prevent embedding out-of-bounds
    X_val_cat = ordinal_enc.transform(X_val_cat_imp).astype(np.int64)
    X_val_cat[X_val_cat < 0] = 0
    
    # Determine embedding sizes: min(50, (num_categories + 1) // 2)
    emb_sizes = []
    for i in range(len(cat_cols)):
        num_cats = len(ordinal_enc.categories_[i])
        emb_dim = min(50, (num_cats + 1) // 2)
        emb_sizes.append((num_cats, max(1, emb_dim)))
        
    return X_train_num, X_val_num, X_train_cat, X_val_cat, emb_sizes

def train_embeddings():
    print("Loading data splits...")
    X_train = pd.read_pickle("data/processed/X_train.pkl")
    y_train = pd.read_pickle("data/processed/y_train.pkl").values
    X_val = pd.read_pickle("data/processed/X_val.pkl")
    y_val = pd.read_pickle("data/processed/y_val.pkl").values
    
    print("Preprocessing data for PyTorch...")
    X_train_num, X_val_num, X_train_cat, X_val_cat, emb_sizes = prepare_data(X_train, X_val)
    
    # Convert to tensors
    t_train_num = torch.tensor(X_train_num, dtype=torch.float32)
    t_train_cat = torch.tensor(X_train_cat, dtype=torch.long)
    t_train_y = torch.tensor(y_train, dtype=torch.float32)
    
    t_val_num = torch.tensor(X_val_num, dtype=torch.float32)
    t_val_cat = torch.tensor(X_val_cat, dtype=torch.long)
    t_val_y = torch.tensor(y_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(t_train_cat, t_train_num, t_train_y)
    val_dataset = TensorDataset(t_val_cat, t_val_num, t_val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
    
    model = TabularEmbeddingNet(emb_sizes, X_train_num.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Training PyTorch Embedding model...")
    start_time = time.time()
    best_loss = float('inf')
    patience = 2
    patience_counter = 0
    
    for epoch in range(10):
        model.train()
        for b_cat, b_num, b_y in train_loader:
            optimizer.zero_grad()
            logits = model(b_cat, b_num)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_cat, b_num, b_y in val_loader:
                logits = model(b_cat, b_num)
                val_loss += criterion(logits, b_y).item() * b_y.size(0)
        val_loss /= len(val_dataset)
        
        print(f"  Epoch {epoch+1} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  Early stopping triggered.")
                break
                
    fit_time = time.time() - start_time
    
    print("\nExtracting embeddings and training Logistic Regression...")
    model.eval()
    X_train_emb = model.extract_features(t_train_cat, t_train_num)
    X_val_emb = model.extract_features(t_val_cat, t_val_num)
    
    # Train scikit-learn Logistic Regression on the extracted features
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X_train_emb, y_train)
    
    y_pred_proba = lr.predict_proba(X_val_emb)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    num_features = X_train_emb.shape[1]
    
    print(f"Embedding Encoder -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Features: {num_features} | Time: {fit_time:.1f}s")
    
    # Load previous results, append, and save
    try:
        with open("reports/encoder_comparison.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []
        
    results.append({
        "encoder": "embedding",
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "num_features": int(num_features),
        "fit_time_seconds": round(fit_time, 2)
    })
    
    with open("reports/encoder_comparison.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nFinal Encoder Comparison:")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    train_embeddings()
