# Note:
# This implementation is illustrative and focuses on making the architectural roles explicit
# for UML extraction and framework instantiation (not on optimizing explanation quality).

# Environment:
# Conda env (python 3.10) with PyTorch, Torchvision, OpenCV, NumPy,
# Matplotlib, scikit-learn, pandas and JupyterLab.
# ResNet18 pretrained weights provided by Torchvision.
# BreakHis image dataset is needed.
# Remember to change the BreakHis directory path.
# This code can take a while to run. Be patient.


# Jupyter Notebook Cell 1 - Imports + Database - <<DataSource>>
import warnings
warnings.filterwarnings("ignore")
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

class ImageDatabase:
    def __init__(self, dataset_dir, image_size=224, batch_size=16):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
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
class ModelB_ResNet18:
    def __init__(self, database: ImageDatabase, epochs=2, lr=1e-4):
        self.database = database
        self.epochs = epochs
        self.lr = lr
        self.untrained_model: resnet18
        self.trained_model: TrainedModelB_ResNet18

    def build_model(self):
        print("Loading dataset...")
        dataset = self.database.get_dataset()
        n = len(dataset)

        train_size = int(0.7 * n)
        test_size = n - train_size
        train_ds, test_ds = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_ds, batch_size=self.database.batch_size, shuffle=True)
        num_classes = len(self.database.get_class_names())

        print("Initializing ResNet18...")
        self.untrained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        in_feats = self.untrained_model.fc.in_features
        self.untrained_model.fc = nn.Linear(in_feats, num_classes)

        model = self.untrained_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        print("Training model...")
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"[Epoch {epoch+1}/{self.epochs}] Loss: {running_loss:.4f}")

        print("ModelB_ResNet18 successfully trained.")

        self.trained_model = TrainedModelB_ResNet18(
            classifier=model,
            train_dataset=train_ds,
            test_dataset=test_ds,
            class_names=self.database.get_class_names(),
            device=device
        )

        return self.trained_model

# Jupyter Notebook Cell 3 - Trained ResNet Model - <<TrainedModel>>
class TrainedModelB_ResNet18:
    def __init__(self, classifier, train_dataset, test_dataset, class_names, device):
        self.model = classifier
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.class_names = class_names
        self.device = device
        self.version = "1.0"
        self.explainer = GradCAMExplainerB(self)

    def predict(self, x):
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            if x.max() > 1:
                x = x / 255.0
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
                if imgs.max() > 1:
                    imgs = imgs / 255.0
                logits = self.model(imgs)
                preds = logits.argmax(dim=1)
                correct += (preds.cpu() == labels).sum().item()
                total += len(labels)

        return correct / total

    def get_train_sample(self, idx=0):
        return self.train_dataset[idx]

# Jupyter Notebook Cell 4 - GradCAM Explainer + Main - <<GradCAMExplainer>>
class GradCAMExplainerB:
    def __init__(self, trained_model):
        self.trained_model = trained_model
        self.model = trained_model.model
        self.device = trained_model.device
        
        self.gradients = None
        self.activations = None

        target_layer = self.model.layer4[-1].conv2

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
        print("GradCAMExplainerB created and hooks registered.")

    def explain(self, image_tensor, class_index=None):
        self.model.eval()
        img = image_tensor.unsqueeze(0).to(self.device)
        if img.max() > 1:
            img = img / 255.0

        logits = self.model(img)
        probs = torch.softmax(logits, dim=1)

        if class_index is None:
            class_index = probs.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = logits[0, class_index]
        class_score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(self.device)

        for c, w in enumerate(weights):
            cam += w * activations[c]

        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam_resized = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[1]))

        if cam_resized.ndim == 3 and cam_resized.shape[2] != 1:
            cam_resized = cam_resized[:, :, 0]

        cam_resized = np.maximum(cam_resized, 0)
        cam_resized = cam_resized / (cam_resized.max() + 1e-8)
        cam_uint8 = np.uint8(cam_resized * 255)

        if cam_uint8.ndim == 2:
            cam_uint8 = np.stack([cam_uint8]*3, axis=-1)

        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        img_np = image_tensor.permute(1,2,0).cpu().numpy()
        
        if img_np.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)

        if img_np.max() <= 1:
            img_np = (img_np * 255).astype(np.uint8)

        overlay = (0.4 * heatmap + 0.6 * img_np).astype(np.uint8)

        return cam_resized, heatmap, overlay

if __name__ == "__main__":
    #CHANGE DIRECTORY HERE. BREAKHIS IS NEEDED!
    DATASET_DIR = r"C:\Users\YOURUSERNAME\BreaKHis_v1\histology_slides\breast"

    print("Loading database...")
    db = ImageDatabase(DATASET_DIR, image_size=224, batch_size=8)
    model_b = ModelB_ResNet18(db, epochs=2, lr=1e-4)
    trained_model = model_b.build_model()

    print("Selecting sample...")
    image_tensor, label = trained_model.get_train_sample(0)
    print("Label:", label, trained_model.class_names[label])

    print("Predicting...")
    probs = trained_model.predict(image_tensor.unsqueeze(0))
    print("Pred:", probs)

    print("Evaluating accuracy...")
    acc = trained_model.evaluate_accuracy(trained_model.train_dataset)
    print("Train accuracy:", acc)

    print("Running Grad-CAM...")
    cam, heatmap, overlay = trained_model.explainer.explain(image_tensor)

    orig_plot = image_tensor.permute(1,2,0).cpu().numpy()
    if orig_plot.min() < 0:
        orig_plot = np.clip(orig_plot * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

    plt.figure(figsize=(18,6))
    plt.subplot(1,3,1)
    plt.imshow(orig_plot)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(overlay)
    plt.title("Grad-CAM Overlay")
    plt.axis("off")
    plt.show()

    print("Done.")