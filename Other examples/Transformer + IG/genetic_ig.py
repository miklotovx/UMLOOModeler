# Note:
# This implementation is illustrative and focuses on making the architectural roles explicit
# for UML extraction and framework instantiation (not on optimizing explanation quality).

# Environment:
# Conda env (python 3.10) with PyTorch (CPU) and Captum (Integrated Gradients).
# HiSeqV2 genomic dataset and BRCA_clinical are needed.
# Remember to change the directory path.

# Jupyter Notebook Cell 1 - Imports + Database + Data organization - <<DataSource>>
import os
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

try:
    from captum.attr import IntegratedGradients
    _HAS_CAPTUM = True
except Exception:
    IntegratedGradients = None
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
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        return TensorDataset(X_tensor, y_tensor)

# Jupyter Notebook Cell 2 - Model instantiation and training - <<ModelDefinition>>
class ModelD_Transformer:
    def __init__(self, database: GeneticDatabase, 
                 embed_dim=64, heads=4, layers=2, 
                 hidden_dim=256, epochs=1, lr=1e-3):
        
        self.database: GeneticDatabase = database
        self.embed_dim = embed_dim
        self.heads = heads
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        
        self.untrained_model = SimpleGenomicTransformer(
            seq_len=self.database.X.shape[1],
            embed_dim=self.embed_dim,
            num_heads=self.heads,
            num_layers=self.layers,
            num_classes=len(np.unique(self.database.y)),
            hidden_dim=self.hidden_dim
        )

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

        trained_model = TrainedModelD(
            model=self.untrained_model,
            train_dataset=train_ds,
            test_dataset=test_ds,
            device=device
        )
        return trained_model

# Jupyter Notebook Cell 3 - Custom classifier - <<Classifier>>
class SimpleGenomicTransformer(nn.Module):
    def __init__(self, seq_len=1000, embed_dim=64, num_heads=4, 
                 num_layers=2, num_classes=2, hidden_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Linear(1, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

# Jupyter Notebook Cell 4 - Trained Transformer Model - <<TrainedModel>>
class TrainedModelD:
    def __init__(self, model: SimpleGenomicTransformer, train_dataset, test_dataset, device):
        self.model: SimpleGenomicTransformer = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device

    def predict(self, X):
        self.model.eval()
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def get_train_sample(self, idx=0):
        return self.train_dataset[idx]

# Jupyter Notebook Cell 5 - IG Explainer + Main - <<IGExplainer>>
class IntegratedGradientsExplainer:
    def __init__(self, trained_model: TrainedModelD, baseline_type='mean', n_baselines=10):
        self.trained_model: TrainedModelD = trained_model
        self.model = trained_model.model
        self.device = trained_model.device
        self.n_baselines = n_baselines
        self.baseline_type = baseline_type
        self.baseline = self._prepare_baseline()

        if _HAS_CAPTUM:
            try:
                self.explainer = IntegratedGradients(self.model)
                self._use_captum = True
            except Exception:
                self._use_captum = False
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
            x_numpy = x.cpu().numpy()
        else:
            x_numpy = np.array(x, dtype=np.float32)

        if x_numpy.ndim == 1:
            x_numpy = np.expand_dims(x_numpy, axis=0)

        if self._use_captum:
            inp = torch.tensor(x_numpy, dtype=torch.float32, device=self.device, requires_grad=True)
            bas = torch.tensor(self.baseline, dtype=torch.float32, device=self.device)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(inp)
                preds = torch.argmax(logits, dim=1)
            attrs = self.explainer.attribute(inp, baselines=bas, target=preds)
            return [attrs.detach().cpu().numpy()]
        return []

if __name__ == "__main__":
    #CHANGE DIRECTORY HERE. HiSeqV2 and BRCA_clinical.tsv ARE NEEDED!
    EXPR_PATH = r"C:\Users\YOURUSERNAME\HiSeqV2"
    CLIN_PATH = r"C:\Users\YOURUSERNAME\BRCA_clinical.tsv"
    db = GeneticDatabase(EXPR_PATH, CLIN_PATH)
    modelD = ModelD_Transformer(db, epochs=1)
    trained = modelD.build_model()
    x_sample, _ = trained.get_train_sample(0)
    print(f"Prediction: {trained.predict(x_sample)}")
    explainer = IntegratedGradientsExplainer(trained)
    vals = explainer.explain(x_sample)
    print(f"IG Explanation generated: {np.array(vals).shape}")
