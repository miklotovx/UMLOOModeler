# Note:
# This implementation is illustrative and focuses on making the architectural roles explicit
# for UML extraction and framework instantiation (not on optimizing explanation quality).

# Environment:
# Conda env (python 3.10) with PyTorch, Torchvision, scikit-image,
# NumPy, Matplotlib, Pillow and JupyterLab.
# Vision Transformer pretrained weights provided by Torchvision.
# BreakHis image dataset is needed.
# Remember to change the BreakHis directory path.
# This code can take a while to run. Be patient. In a medium PC, the kernel may crash.

# Jupyter Notebook Cell 1 - Imports + Database - <<DataSource>>
import warnings
warnings.filterwarnings("ignore")
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torchvision
from torchvision import transforms, datasets
from torchvision.models import vit_b_16, ViT_B_16_Weights
import matplotlib.pyplot as plt

VIT_MEAN = [0.485, 0.456, 0.406]
VIT_STD  = [0.229, 0.224, 0.225]

class ImageDatabase:
    def __init__(self, dataset_dir, image_size=224, batch_size=16):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
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
class ModelD_ViT:
    def __init__(self, database: ImageDatabase, epochs=1, lr=1e-4):
        self.database = database
        self.epochs = epochs
        self.lr = lr
        self.untrained_model: vit_b_16
        self.trained_model: TrainedModelD_ViT

    def build_model(self):
        print("Loading dataset...")
        dataset = self.database.get_dataset()
        n = len(dataset)

        train_size = int(0.7 * n)
        test_size = n - train_size
        train_ds, test_ds = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(
            train_ds,
            batch_size=self.database.batch_size,
            shuffle=True
        )

        num_classes = len(self.database.get_class_names())

        print("Initializing ViT-B/16...")
        self.untrained_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        in_features = self.untrained_model.heads.head.in_features
        self.untrained_model.heads.head = nn.Linear(in_features, num_classes)

        model = self.untrained_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        print("Training model...")
        for epoch in range(self.epochs):
            model.train()
            running = 0.0
            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                running += loss.item()

            print(f"[Epoch {epoch+1}/{self.epochs}] Loss: {running:.4f}")

        print("ModelD_ViT successfully trained.")

        self.trained_model = TrainedModelD_ViT(
            model=model,
            train_dataset=train_ds,
            test_dataset=test_ds,
            class_names=self.database.get_class_names(),
            device=device,
            mean=VIT_MEAN,
            std=VIT_STD
        )

        return self.trained_model

# Jupyter Notebook Cell 3 - Trained VIT Model - <<TrainedModel>>
class TrainedModelD_ViT:
    def __init__(self, model, train_dataset, test_dataset, class_names, device, mean, std):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.class_names = class_names
        self.device = device
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.explainer = IntegratedGradientsExplainerD(self)

    def predict(self, x):
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def evaluate_accuracy(self, subset):
        loader = DataLoader(subset, batch_size=16, shuffle=False)
        correct = total = 0

        self.model.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                logits = self.model(imgs)
                preds = logits.argmax(dim=1)
                correct += (preds.cpu() == labels).sum().item()
                total += len(labels)

        return correct / total

    def get_train_sample(self, idx=0):
        return self.train_dataset[idx]

# Jupyter Notebook Cell 4 - IG Explainer + Main - <<IGExplainer>>
class IntegratedGradientsExplainerD:
    def __init__(self, trained_model, steps=30, baseline_type="black"):
        self.trained_model = trained_model
        self.model = trained_model.model
        self.device = trained_model.device
        self.steps = max(1, steps)
        self.baseline_type = baseline_type
        print(f"IntegratedGradientsExplainerD created (steps={self.steps}, baseline={baseline_type}).")

    def _make_baseline(self, image_tensor):
        if self.baseline_type == "black":
            return torch.zeros_like(image_tensor).to(self.device)
        else:
            img = image_tensor.permute(1,2,0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            blurred = cv2.GaussianBlur(img, (31,31), 0)
            t = torch.tensor(blurred / 255.0, dtype=torch.float32).permute(2,0,1)
            return t.to(self.device)

    def _denormalize_for_visualization(self, img_np):
        img = img_np.transpose(2,0,1)
        img = img * self.trained_model.std.reshape(3,1,1) + self.trained_model.mean.reshape(3,1,1)
        img = np.clip(img, 0, 1)
        img = (img.transpose(1,2,0) * 255).astype(np.uint8)
        return img

    def explain(self, image_tensor, target_class=None):
        img = image_tensor.unsqueeze(0).to(self.device)
        baseline = self._make_baseline(image_tensor).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(img)
            probs = torch.softmax(logits, dim=1)
            pred_label = int(probs.argmax(dim=1).item())

        if target_class is None:
            target_class = pred_label

        total_grad = torch.zeros_like(img).to(self.device)

        for i in range(1, self.steps+1):
            alpha = i / self.steps
            interp = baseline + alpha * (img - baseline)
            interp.requires_grad_(True)

            out = self.model(interp)
            score = out[0, target_class]

            self.model.zero_grad()
            score.backward()

            total_grad += interp.grad.detach()

        avg_grad = total_grad / self.steps
        ig_map = torch.sum(torch.abs((img - baseline) * avg_grad), dim=1).squeeze(0)

        ig_map = ig_map.cpu().numpy()
        ig_map = (ig_map - ig_map.min()) / (ig_map.max() + 1e-8)

        heat_uint8 = np.uint8(ig_map * 255)
        heat_uint8 = np.stack([heat_uint8]*3, axis=-1)

        heatmap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        img_np = image_tensor.permute(1,2,0).cpu().numpy()
        img_vis = self._denormalize_for_visualization(img_np)

        overlay = (0.4 * heatmap + 0.6 * img_vis).astype(np.uint8)

        return ig_map, heatmap, overlay

if __name__ == "__main__":
    #CHANGE DIRECTORY HERE. BREAKHIS IS NEEDED!
    DATASET_DIR = r"C:\Users\YOURUSERNAME\BreaKHis_v1\histology_slides\breast"

    print("Loading database...")
    db = ImageDatabase(DATASET_DIR, image_size=224, batch_size=4)

    vit_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
    ])
    db.transform = vit_transform

    db.dataset = datasets.ImageFolder(
        root=db.dataset_dir,
        transform=db.transform
    )

    print("Classes:", db.classes)
    print("Número de classes:", len(db.classes))

    model_d = ModelD_ViT(db, epochs=1, lr=1e-4)
    trained = model_d.build_model()

    print("Selecting sample...")
    image_tensor, label = trained.get_train_sample(0)
    print("Label:", label, trained.class_names[label])

    print("Predicting...")
    probs = trained.predict(image_tensor.unsqueeze(0))
    print("Pred:", probs)

    print("Evaluating accuracy...")
    acc = trained.evaluate_accuracy(trained.train_dataset)
    print("Train accuracy:", acc)

    print("Running Integrated Gradients via composed explainer...")
    igmap, heatmap, overlay = trained.explainer.explain(image_tensor, target_class=label)

    img_np = image_tensor.permute(1,2,0).cpu().numpy()
    img_vis = trained.explainer._denormalize_for_visualization(img_np)

    plt.figure(figsize=(18,6))
    plt.subplot(1,3,1)
    plt.imshow(img_vis)
    plt.title("Original (Denormalized)")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(heatmap)
    plt.title("IG Heatmap")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")
    plt.show()

    print("Done.")