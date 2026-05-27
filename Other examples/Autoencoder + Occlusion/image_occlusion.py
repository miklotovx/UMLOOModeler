# Note:
# This implementation is illustrative and focuses on making the architectural roles explicit
# for UML extraction and framework instantiation (not on optimizing explanation quality).

# Environment:
# Conda env (python 3.10) with PyTorch, Torchvision, scikit-image, NumPy,
# matplotlib, Pillow and JupyterLab.
# Occlusion-based reconstruction explainability using a convolutional Autoencoder.
# BreakHis image dataset is needed.
# Remember to change the BreakHis directory path.

# Jupyter Notebook Cell 1 - Imports + Database - <<DataSource>>
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

class ImageDatabase:
    def __init__(self, dataset_dir, image_size=224, batch_size=16):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        self.dataset = datasets.ImageFolder(
            root=self.dataset_dir,
            transform=self.transform
        )

        self.classes = self.dataset.classes

    def get_dataset(self):
        return self.dataset

    def get_dataloader(self, subset=None, shuffle=True):
        ds = subset if subset else self.dataset
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def get_class_names(self):
        return self.classes

# Jupyter Notebook Cell 2 - Model instantiation and training - <<ModelDefinition>>
class ModelC_Autoencoder:
    def __init__(self, database: ImageDatabase, epochs=2, lr=1e-3, latent_dim=128):
        self.database = database
        self.epochs = epochs
        self.lr = lr
        self.latent_dim = latent_dim
        self.untrained_model: SimpleAutoencoder

    def build_model(self):
        dataset = self.database.get_dataset()
        n = len(dataset)
        train_size = int(0.7 * n)
        test_size = n - train_size
        train_ds, test_ds = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_ds, batch_size=self.database.batch_size, shuffle=True)

        self.untrained_model = SimpleAutoencoder(latent_dim=self.latent_dim)
        model = self.untrained_model
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            model.train()
            running = 0.0
            for imgs, _ in train_loader:
                imgs = imgs.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = loss_fn(outputs, imgs)
                loss.backward()
                optimizer.step()
                running += loss.item()

            print(f"[Epoch {epoch+1}/{self.epochs}] Loss: {running:.4f}")

        print("ModelC_Autoencoder successfully trained.")

        trained_model = TrainedModelC_Autoencoder(
            autoencoder=model,
            train_dataset=train_ds,
            test_dataset=test_ds,
            class_names=self.database.get_class_names(),
            device=device
        )

        return trained_model

# Jupyter Notebook Cell 3 - Custom Autoencoder - <<Classifier>>
class SimpleAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(SimpleAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 14 * 14),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 14, 14)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),   
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Jupyter Notebook Cell 4 - Trained Autoencoder Model - <<TrainedModel>>
class TrainedModelC_Autoencoder:
    def __init__(
        self,
        autoencoder: SimpleAutoencoder,   
        train_dataset,
        test_dataset,
        class_names,
        device
    ):
        self.autoencoder = autoencoder   
        self.model = autoencoder
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.class_names = class_names
        self.device = device
        self.version = "1.0"

        self.explainer = OcclusionExplainerC(self)

    def reconstruct(self, x):
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            if x.max() > 1:
                x = x / 255.
            out = self.model(x)
        return out.cpu()

    def reconstruction_error(self, x):
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            if x.max() > 1:
                x = x / 255.
            recon = self.model(x)
            mse = torch.mean((recon - x) ** 2)
        return mse.item()

    def evaluate_reconstruction(self, subset):
        loader = DataLoader(subset, batch_size=16, shuffle=False)
        total_mse = 0.0
        count = 0

        self.model.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                if x.max() > 1:
                    x = x / 255.
                recon = self.model(x)
                mse = torch.mean((recon - x) ** 2).item()
                total_mse += mse
                count += 1

        return total_mse / count if count > 0 else 0.0

    def get_train_sample(self, idx=0):
        return self.train_dataset[idx]

# Jupyter Notebook Cell 5 - Occlusion Explainer - <<OcclusionExplainer>>
class OcclusionExplainerC:
    def __init__(self, trained_model):
        self.trained_model = trained_model
        print("OcclusionExplainerC created.")

    def explain_instance(self, image_tensor, patch_size=32, stride=32):
        print("Running Occlusion Sensitivity...")
        trained = self.trained_model
        device = trained.device

        img = image_tensor.unsqueeze(0).to(device)
        if img.max() > 1:
            img = img / 255.

        _, _, H, W = img.shape
        heatmap = np.zeros((H, W))
        img_np = img.squeeze(0).permute(1,2,0).cpu().numpy()

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                occluded = img_np.copy()
                occluded[y:y+patch_size, x:x+patch_size] = 0

                occ_tensor = torch.tensor(occluded, dtype=torch.float32)
                occ_tensor = occ_tensor.permute(2,0,1).unsqueeze(0).to(device)

                with torch.no_grad():
                    recon = trained.model(occ_tensor)

                mse = torch.mean((recon - img) ** 2).item()
                heatmap[y:y+patch_size, x:x+patch_size] = mse

        heatmap_norm = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
        return heatmap_norm, img_np

if __name__ == "__main__":
    #CHANGE DIRECTORY HERE. BREAKHIS IS NEEDED!
    DATASET_DIR = r"C:\Users\YOURUSERNAME\BreaKHis_v1\histology_slides\breast"

    print("Loading database...")
    db = ImageDatabase(DATASET_DIR, image_size=224, batch_size=8)

    print("Training model...")
    model_c = ModelC_Autoencoder(db, epochs=2, lr=1e-3, latent_dim=128)
    trained_model = model_c.build_model()

    print("Selecting sample...")
    image_tensor, label = trained_model.get_train_sample(0)
    print("Label:", label, trained_model.class_names[label])

    print("Reconstruction error for sample...")
    err = trained_model.reconstruction_error(image_tensor.unsqueeze(0))
    print("Reconstruction MSE:", err)

    print("Evaluating model reconstruction on training set...")
    mse_avg = trained_model.evaluate_reconstruction(trained_model.train_dataset)
    print("Average reconstruction MSE:", mse_avg)

    print("Creating explainer...")

    heatmap, original_img = trained_model.explainer.explain_instance(
        image_tensor, 
        patch_size=32, 
        stride=32
    )

    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(heatmap, cmap="hot")
    plt.title("Occlusion Heatmap")
    plt.axis("off")
    plt.show()

    print("Done.")