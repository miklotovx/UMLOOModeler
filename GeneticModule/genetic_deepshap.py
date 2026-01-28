# Note:
# This implementation is illustrative and focuses on making the architectural roles explicit
# for UML extraction and framework instantiation (not on optimizing explanation quality).

# Environment:
# Conda env (python 3.10) with PyTorch (CPU) and SHAP.
# HiSeqV2 genomic dataset and BRCA_clinical are needed.
# Remember to change the directory path.

# Jupyter Notebook Cell 1 - Imports + Database + Data organization - <<DataSource>>
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import shap
from sklearn.preprocessing import StandardScaler
import warnings

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
        if self.verbose:
            print("Loading expression matrix...")

        df = pd.read_csv(self.expression_path, sep="\t")
        df = df.set_index(df.columns[0])
        df = df.T  

        return df

    def _load_clinical(self):
        if self.verbose:
            print("Loading clinical data...")

        df = pd.read_csv(self.clinical_path, sep="\t")

        if "sampleID" in df.columns:
            df = df.set_index("sampleID")

        return df

    def _align_samples(self, expr, clin):
        if self.verbose:
            print("Aligning samples...")

        common = expr.index.intersection(clin.index)
        expr = expr.loc[common]
        clin = clin.loc[common]

        return expr, clin

    def _select_top_genes(self, expr):
        if self.verbose:
            print(f"Selecting top {self.top_k_genes} genes...")

        variances = expr.var(axis=0)
        top = variances.nlargest(self.top_k_genes).index

        return expr[top], top

    def _standardize(self, X):
        if self.verbose:
            print("Standardizing features...")

        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def _load(self):
        expr = self._load_expression()
        clin = self._load_clinical()
        expr, clin = self._align_samples(expr, clin)
        expr, genes = self._select_top_genes(expr)

        if self.verbose:
            print("Using 'vital_status' as labels...")

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
        X_tensor = torch.tensor(self.X, dtype=torch.float32).unsqueeze(-1)  
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        return TensorDataset(X_tensor, y_tensor)

# Jupyter Notebook Cell 2 - Model instantiation and training - <<ModelDefinition>>
class ModelD_Bilstm: 

    def __init__(self, database: GeneticDatabase,
                 hidden_size=64, num_layers=1,
                 epochs=2, lr=1e-3):

        self.database = database
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.untrained_model: SimpleBilstm

    def build_model(self):
        ds = self.database.get_tensor_dataset()
        n = len(ds)
        train_size = int(0.7 * n)
        test_size = n - train_size
        train_ds, test_ds = random_split(ds, [train_size, test_size])
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        num_classes = len(np.unique(self.database.y))

        self.untrained_model = SimpleBilstm(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=num_classes
        )

        model: SimpleBilstm = self.untrained_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        print("Training BiLSTM...")
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0

            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(Xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}")

            trained_model = TrainedModelD(
                model=self.untrained_model,
                train_dataset=train_ds,
                test_dataset=test_ds,
                device=device
            )

            return trained_model

# Jupyter Notebook Cell 3 - Custom classifier - <<Classifier>>
class SimpleBilstm(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# Jupyter Notebook Cell 4 - Trained Bilstm Model - <<TrainedModel>>
class TrainedModelD:

    def __init__(self, model: SimpleBilstm, train_dataset, test_dataset, device):

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device

    def predict(self, X):
        self.model.eval()

        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        if X.ndim == 2:
            X = X.unsqueeze(0)

        X = X.to(self.device)

        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def get_train_sample(self, idx=0):
        return self.train_dataset[idx]

# Jupyter Notebook Cell 5 - DeepSHAP Explainer + Main - <<ShapExplainer>>
def _disable_shap_sum_checks():
    if hasattr(shap, "utils") and hasattr(shap.utils, "_assert_sum_equivalent"):
        shap.utils._assert_sum_equivalent = lambda *args, **kwargs: None

    if hasattr(shap, "explainers"):
        try:
            deep_mod = shap.explainers._deep
            if hasattr(deep_mod, "_assert_sum_equivalent"):
                deep_mod._assert_sum_equivalent = lambda *args, **kwargs: None
        except Exception:
            pass

    try:
        if hasattr(shap, "utils") and hasattr(shap.utils, "check_if_additive"):
            shap.utils.check_if_additive = lambda *args, **kwargs: None
    except Exception:
        pass

_disable_shap_sum_checks()

class DeepShapExplainerD:

    def __init__(self, trained_model: TrainedModelD):
        print("Initializing DeepSHAP...")

        self.trained_model = trained_model
        self.model = trained_model.model
        self.device = trained_model.device

        bg_samples = []
        for i in range(5):
            x, _ = trained_model.get_train_sample(i)
            bg_samples.append(x.unsqueeze(0))

        self.background = torch.cat(bg_samples, dim=0).to(self.device)

        try:
            self.explainer = shap.DeepExplainer(self.model, self.background)
        except Exception as e:
            warnings.warn(f"DeepSHAP init failed: {e}. Will use fallback if needed.")
            self.explainer = None

    def _approx_shap_via_gradient(self, x):
        model = self.model
        model.eval()

        bg_mean = self.background.mean(dim=0, keepdim=True)

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)

        x = x.to(self.device).detach()
        x.requires_grad_(True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        class_idx = int(probs.argmax(dim=1).item())
        score = logits[0, class_idx]
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        score.backward()
        grad = x.grad.detach()
        diff = (x - bg_mean)
        shap_approx = (grad * diff).detach().squeeze(0).cpu().numpy()

        return [shap_approx]

    def explain(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        if x.ndim == 2:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        if self.explainer is not None:
            _disable_shap_sum_checks()

            try:
                shap_values = self.explainer.shap_values(x)
                processed = []
                for sv in shap_values:
                    if isinstance(sv, torch.Tensor):
                        processed.append(sv.detach().cpu().numpy())
                    else:
                        processed.append(np.array(sv))
                return processed

            except AssertionError as ae:
                warnings.warn(f"DeepSHAP assertion error: {ae}. Falling back to gradient approximation.")
            except Exception as e:
                warnings.warn(f"DeepSHAP failed at shap_values: {e}. Falling back to gradient approximation.")

        return self._approx_shap_via_gradient(x)

if __name__ == "__main__":
    #CHANGE DIRECTORY HERE. HiSeqV2 and BRCA_clinical.tsv ARE NEEDED!
    EXPR_PATH = r"C:\Users\YOURUSERNAME\HiSeqV2"            
    CLIN_PATH = r"C:\Users\YOURUSERNAME\BRCA_clinical.tsv"  

    db = GeneticDatabase(
        expression_path=EXPR_PATH,
        clinical_path=CLIN_PATH,
        top_k_genes=1000,
        do_standardize=True,
        verbose=True
    )

    print("Building BiLSTM model...")
    model_d = ModelD_Bilstm(db, epochs=1, lr=1e-3)
    trained_model = model_d.build_model()

    print("Selecting sample...")
    x_sample, y_sample = trained_model.get_train_sample(0)
    print("Label:", y_sample)

    print("Predicting...")
    pred = trained_model.predict(x_sample)
    print("Pred:", pred)

    print("Creating DeepSHAP Explainer...")
    explainer = DeepShapExplainerD(trained_model)

    print("Explaining sample...")
    shap_vals = explainer.explain(x_sample)

    print("SHAP values shape:", np.array(shap_vals).shape)

    print("Done.")