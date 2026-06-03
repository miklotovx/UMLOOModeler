# Note:
# This implementation is illustrative and focuses on making the architectural roles explicit
# for UML extraction and framework instantiation (not on optimizing explanation quality).

# Environment:
# Conda env (python 3.10) with PyTorch (CPU) and Captum (DeepLIFT).
# HiSeqV2 genomic dataset and BRCA_clinical are needed.
# Remember to change the directory path.

# Jupyter Notebook Cell 1 - Imports + Database + Data organization - <<DataSource>>
import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

try:
    from captum.attr import DeepLift
    _HAS_CAPTUM = True
except ImportError:
    DeepLift = None
    _HAS_CAPTUM = False

class GeneticDatabase:
    def __init__(self, expression_path, clinical_path, 
                 top_k_genes=1000, do_standardize=True, verbose=True):
        self.expression_path = expression_path
        self.clinical_path = clinical_path
        self.top_k_genes = top_k_genes
        self.do_standardize = do_standardize
        self.verbose = verbose
        self.X = None
        self.y = None
        self.genes = None
        self._load()

    def _load_expression(self):
        df = pd.read_csv(self.expression_path, sep="\t")
        df = df.set_index(df.columns[0])
        df = df.T
        return df

    def _load_clinical(self):
        df = pd.read_csv(self.clinical_path, sep="\t")
        if "sampleID" in df.columns:
            df = df.set_index("sampleID")
        return df

    def _align_samples(self, expr, clin):
        common = expr.index.intersection(clin.index)
        expr = expr.loc[common]
        clin = clin.loc[common]
        return expr, clin

    def _select_top_genes(self, expr):
        variances = expr.var(axis=0)
        top = variances.nlargest(self.top_k_genes).index
        return expr[top], top

    def _standardize(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def _load(self):
        expr = self._load_expression()
        clin = self._load_clinical()
        expr, clin = self._align_samples(expr, clin)
        expr, genes = self._select_top_genes(expr)
        clin = clin.dropna(subset=["vital_status"])
        common = expr.index.intersection(clin.index)
        expr = expr.loc[common]
        clin = clin.loc[common]
        
        X = expr.values.astype(np.float32)
        y = clin["vital_status"].astype("category").cat.codes.values
        
        if self.do_standardize:
            X = self._standardize(X)
            
        self.X = X
        self.y = y
        self.genes = genes

    def get_tensor_dataset(self):
        X_tensor = torch.tensor(self.X, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        return TensorDataset(X_tensor, y_tensor)

# Jupyter Notebook Cell 2 - Model instantiation and training - <<ModelDefinition>>
class ModelC_Cnn1d:
    def __init__(self, database: GeneticDatabase, hidden_channels=32, epochs=2, lr=1e-3):
        self.database: GeneticDatabase = database
        self.hidden_channels = hidden_channels
        self.epochs = epochs
        self.lr = lr
        
        self.untrained_model = SimpleCnn1d(
            in_channels=1,
            num_classes=len(np.unique(self.database.y)),
            seq_len=self.database.X.shape[1],
            hidden_channels=self.hidden_channels
        )
        self.trained_model: TrainedModelC | None = None

    def build_model(self):
        ds = self.database.get_tensor_dataset()
        n = len(ds)
        train_size = int(0.7 * n)
        test_size = n - train_size
        train_ds, test_ds = random_split(ds, [train_size, test_size])
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        model = self.untrained_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(Xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

        self.trained_model = TrainedModelC(
            model=self.untrained_model,
            train_dataset=train_ds,
            test_dataset=test_ds,
            device=device
        )
        return self.trained_model

# Jupyter Notebook Cell 3 - Custom classifier - <<Classifier>>
class SimpleCnn1d(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, seq_len=1000, hidden_channels=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_channels*2)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(hidden_channels*2, hidden_channels*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_channels*4)
        self.act3 = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_channels*4, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.pool3(self.act3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)
        
# Jupyter Notebook Cell 4 - Trained CNN Model - <<TrainedModel>>
class TrainedModelC:
    def __init__(self, model: SimpleCnn1d, train_dataset, test_dataset, device):
        self.model: SimpleCnn1d = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device

    def predict(self, X):
        self.model.eval()
        
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
            
        if X.ndim == 1:
            X = X.unsqueeze(0).unsqueeze(0)
        elif X.ndim == 2:
            X = X.unsqueeze(0)
            
        X = X.to(self.device)
        
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def get_train_sample(self, idx=0):
        return self.train_dataset[idx]

# Jupyter Notebook Cell 5 - DeepLIFT Explainer + Main - <<DeepExplainer>>
class DeepLiftExplainer:
    def __init__(self, trained_model: TrainedModelC, baseline_type='mean', n_baselines=10):
        self.trained_model = trained_model
        self.model = trained_model.model
        self.device = trained_model.device
        self.n_baselines = n_baselines
        self.baseline_type = baseline_type
        self.baseline = self._prepare_baseline()
        
        if _HAS_CAPTUM:
            self.explainer = DeepLift(self.model)
            self._use_captum = True
        else:
            self._use_captum = False

    def _prepare_baseline(self):
        bg_samples = []
        for i in range(min(self.n_baselines, len(self.trained_model.train_dataset))):
            x, _ = self.trained_model.get_train_sample(i)
            bg_samples.append(x.unsqueeze(0).cpu().numpy())
        
        if not bg_samples: return None
        bas = np.concatenate(bg_samples, axis=0)
        return np.mean(bas, axis=0, keepdims=True).astype(np.float32)

    def explain(self, x):
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
        
        if x_np.ndim == 2:
            x_np = np.expand_dims(x_np, axis=0)
        elif x_np.ndim == 1:
            x_np = np.expand_dims(np.expand_dims(x_np, axis=0), axis=0)

        inp = torch.tensor(x_np, dtype=torch.float32, device=self.device, requires_grad=True)
        bas = torch.tensor(self.baseline, dtype=torch.float32, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(inp)
            target = torch.argmax(logits, dim=1)

        attributions = self.explainer.attribute(inp, baselines=bas, target=target)
        return [attributions.detach().cpu().numpy()]

if __name__ == "__main__":
    #CHANGE DIRECTORY HERE. HiSeqV2 and BRCA_clinical.tsv ARE NEEDED!
    EXPR_PATH = r"C:\Users\YOURUSERNAME\HiSeqV2"
    CLIN_PATH = r"C:\Users\YOURUSERNAME\BRCA_clinical.tsv"
    db = GeneticDatabase(EXPR_PATH, CLIN_PATH)
    modelC = ModelC_Cnn1d(db, hidden_channels=32, epochs=1)
    trained_model = modelC.build_model()
    x_sample, _ = trained_model.get_train_sample(0)
    print(f"Prediction (Labels Prob): {trained_model.predict(x_sample)}")
    
    if _HAS_CAPTUM:
        explainer = DeepLiftExplainer(trained_model)
        shap_vals = explainer.explain(x_sample)
        print(f"DeepLIFT Explanation generated. Shape: {np.array(shap_vals).shape}")