# Note:
# This implementation is illustrative and focuses on making the architectural roles explicit
# for UML extraction and framework instantiation (not on optimizing explanation quality).

# Environment:
# Conda env (python 3.10) with PyTorch (CPU) and SHAP (GradientSHAP approximation).
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
import shap
from sklearn.preprocessing import StandardScaler

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
        X_tensor = torch.tensor(self.X, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        return TensorDataset(X_tensor, y_tensor)

# Jupyter Notebook Cell 2 - Model instantiation and training - <<ModelDefinition>>
class ModelB_Gru:
    def __init__(self, database: GeneticDatabase,
                 hidden_size=64, num_layers=1,
                 epochs=2, lr=1e-3):

        self.database: GeneticDatabase = database
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        
        self.untrained_model = SimpleGru(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=len(np.unique(self.database.y)) if self.database.y is not None else 2
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

        trained_model = TrainedModelB(
            model=self.untrained_model,
            train_dataset=train_ds,
            test_dataset=test_ds,
            device=device
        )

        return trained_model

# Jupyter Notebook Cell 3 - Custom classifier - <<Classifier>>
class SimpleGru(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)            
        out = out[:, -1, :]             
        return self.fc(out)

# Jupyter Notebook Cell 4 - Trained GRU Model - <<TrainedModel>>
class TrainedModelB:
    def __init__(self, model: SimpleGru, train_dataset, test_dataset, device):
        self.model: SimpleGru = model
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

# Jupyter Notebook Cell 5 - GShap Explainer + Main - <<GShapExplainer>>
class GradientShap:
    def __init__(self, trained_model: TrainedModelB, nsamples=50):
        self.trained_model: TrainedModelB = trained_model
        self.model = trained_model.model
        self.device = trained_model.device
        self.nsamples = nsamples

        bg_samples = []
        for i in range(min(10, len(self.trained_model.train_dataset))):
            x, _ = self.trained_model.get_train_sample(i)
            if isinstance(x, torch.Tensor):
                bg_samples.append(x.unsqueeze(0).cpu().numpy())  
            else:
                bg_samples.append(np.expand_dims(x, axis=0))
        if len(bg_samples) == 0:
            self.background = np.zeros((1, 1, 1), dtype=np.float32)
        else:
            self.background = np.concatenate(bg_samples, axis=0)

        def model_predict_numpy(x_numpy):
            with torch.no_grad():
                xt = torch.tensor(x_numpy, dtype=torch.float32).to(self.device)
                logits = self.model(xt)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        try:
            self.explainer = shap.GradientExplainer(model_predict_numpy, self.background)
            self._use_native = True
        except Exception:
            self.explainer = None
            self._use_native = False

    def _approx_shap_via_gradient(self, x_numpy):
        model = self.model
        device = self.device

        if x_numpy.ndim == 2:
            x_numpy = np.expand_dims(x_numpy, axis=0)

        bg_mean = np.mean(self.background, axis=0, keepdims=True).astype(np.float32)
        accum = np.zeros_like(x_numpy, dtype=np.float32)
        B = x_numpy.shape[0]

        for _ in range(self.nsamples):
            alpha = np.random.rand(B, 1, 1).astype(np.float32)
            interp = bg_mean + alpha * (x_numpy - bg_mean)
            noise = np.random.normal(scale=0.01, size=interp.shape).astype(np.float32)
            inp = torch.tensor(interp + noise, dtype=torch.float32, device=device, requires_grad=True)

            logits = model(inp)
            probs = torch.softmax(logits, dim=1)
            class_idx = torch.argmax(probs, dim=1) 
            for i in range(B):
                model.zero_grad()
                if inp.grad is not None:
                    inp.grad.zero_()
                score = logits[i, int(class_idx[i])]
                score.backward(retain_graph=True)
                grad = inp.grad[i].detach().cpu().numpy()  
                diff = (x_numpy[i] - bg_mean[0])
                accum[i] += (grad * diff)

        shap_approx = accum / float(self.nsamples)
        return [shap_approx]

    def explain(self, x):
        if isinstance(x, torch.Tensor):
            x_numpy = x.cpu().numpy()
        elif isinstance(x, np.ndarray):
            x_numpy = x
        else:
            x_numpy = np.array(x, dtype=np.float32)

        if x_numpy.ndim == 2:
            x_numpy = np.expand_dims(x_numpy, axis=0)

        if self._use_native and (self.explainer is not None):
            try:
                shap_values = self.explainer.shap_values(x_numpy)
                processed = []
                for sv in shap_values:
                    processed.append(np.array(sv))
                return processed
            except Exception as e:
                warnings.warn(f"[Explainer] GradientExplainer.shap_values failed: {e}. Using fallback.")
                return self._approx_shap_via_gradient(x_numpy)
        else:
            return self._approx_shap_via_gradient(x_numpy)

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

    modelB = ModelB_Gru(db, epochs=1, lr=1e-3)
    trained_model = modelB.build_model()
    x_sample, y_sample = trained_model.get_train_sample(0)
    pred = trained_model.predict(x_sample)
    print("Pred:", pred)
    explainer = GradientShap(trained_model)
    shap_vals = explainer.explain(x_sample)
    print("GSHAP values shape (approx):", np.array(shap_vals).shape)
