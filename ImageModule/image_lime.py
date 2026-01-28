# Note:
# This implementation is illustrative and focuses on making the architectural roles explicit
# for UML extraction and framework instantiation (not on optimizing explanation quality).

# Environment:
# Conda env (python 3.10) with PyTorch and image processing libraries.
# LIME installed via pip.
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
from lime import lime_image
from skimage.segmentation import slic, mark_boundaries
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
class ModelC_CNN:
    def __init__(self, database: ImageDatabase, epochs=2, lr=1e-3):
        self.database = database
        self.epochs = epochs
        self.lr = lr
        self.untrained_model: SimpleCNN

    def build_model(self):
        dataset = self.database.get_dataset()
        n = len(dataset)
        train_size = int(0.7 * n)
        test_size = n - train_size
        train_ds, test_ds = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_ds, batch_size=self.database.batch_size, shuffle=True)
        num_classes = len(self.database.get_class_names())
        self.untrained_model = SimpleCNN(num_classes=num_classes)
        model = self.untrained_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            model.train()
            running = 0.0
            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = loss_fn(out, labels)
                loss.backward()
                optimizer.step()
                running += loss.item()

            print(f"[Epoch {epoch+1}/{self.epochs}] Loss: {running:.4f}")

        print("ModelC_CNN successfully trained.")

        trained_model = TrainedModelC_CNN(
            classifier=model,
            train_dataset=train_ds,
            test_dataset=test_ds,
            class_names=self.database.get_class_names(),
            device=device
        )

        return trained_model

# Jupyter Notebook Cell 3 - Custom classifier - <<Classifier>>
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Jupyter Notebook Cell 4 - Trained CNN Model - <<TrainedModel>>
class TrainedModelC_CNN:
    def __init__(
        self,
        classifier: SimpleCNN,   
        train_dataset,
        test_dataset,
        class_names,
        device
    ):
        
        self.classifier = classifier   
        self.model = classifier
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.class_names = class_names
        self.device = device
        self.perturbator = PerturbatorC_Image(self)

    def predict(self, x):
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            if x.max() > 1:
                x = x / 255.
            out = self.model(x)
            probs = torch.softmax(out, dim=1)
        return probs.cpu().numpy()

    def evaluate_accuracy(self, subset):
        loader = DataLoader(subset, batch_size=16, shuffle=False)
        correct = total = 0

        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                if x.max() > 1:
                    x = x / 255.
                logits = self.model(x)
                preds = logits.argmax(dim=1)
                correct += (preds.cpu() == y).sum().item()
                total += len(y)

        return correct / total

    def get_train_sample(self, idx=0):
        return self.train_dataset[idx]

# Jupyter Notebook Cell 5 - Lime Perturbator - <<Perturbator>>
class PerturbatorC_Image:
    def __init__(self, trained_model):
        self.trained_model = trained_model
        self.explainer = LimeExplainerC_Image(self)
        print("PerturbatorC_Image created.")

    def perturb(self, image_tensor, num_samples=50, n_segments=50):
        print("Generating perturbations...")

        img = image_tensor.permute(1,2,0).cpu().numpy()
        if img.max() > 1:
            img = img / 255.

        segments = slic(img, n_segments=n_segments, compactness=10)
        num_sp = len(np.unique(segments))

        perturbed = []
        for _ in range(num_samples):
            mask = np.random.choice([0,1], size=num_sp)
            new = img.copy()
            for sp in range(num_sp):
                if mask[sp] == 0:
                    new[segments == sp] = 0.
            perturbed.append(new)

        return perturbed, segments

# Jupyter Notebook Cell 6 - Lime Explainer + Main - <<LimeExplainer>>
class LimeExplainerC_Image:
    def __init__(self, perturbator):
        self.perturbator = perturbator
        print("LimeExplainerC_Image created.")

    def explain_instance(self, image_tensor, num_samples=500):

        trained = self.perturbator.trained_model

        img_np = image_tensor.permute(1,2,0).cpu().numpy()
        if img_np.max() > 1:
            img_np = img_np / 255.

        def predict_fn(images_np):
            t = torch.tensor(images_np, dtype=torch.float32).permute(0,3,1,2)
            t = t.to(trained.device)
            if t.max() > 1:
                t = t / 255.
            with torch.no_grad():
                out = trained.model(t)
                p = torch.softmax(out, dim=1)
            return p.cpu().numpy()

        explainer = lime_image.LimeImageExplainer()

        explanation = explainer.explain_instance(
            img_np,
            classifier_fn=predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=num_samples
        )

        label = explanation.top_labels[0]

        temp, mask = explanation.get_image_and_mask(
            label=label,
            positive_only=True,
            hide_rest=False,
            num_features=5
        )

        highlighted = mark_boundaries(temp, mask)
        return explanation, mask, highlighted

if __name__ == "__main__":
    #CHANGE DIRECTORY HERE. BREAKHIS IS NEEDED!
    DATASET_DIR = r"C:\Users\YOURUSERNAME\BreaKHis_v1\histology_slides\breast"

    print("Loading database...")
    db = ImageDatabase(DATASET_DIR, image_size=224, batch_size=8)

    print("Training model...")
    model_c = ModelC_CNN(db, epochs=2, lr=1e-3)
    trained_model = model_c.build_model()

    print("Selecting sample...")
    image_tensor, label = trained_model.get_train_sample(0)
    print("Label:", label, trained_model.class_names[label])

    print("Predicting...")
    probs = trained_model.predict(image_tensor.unsqueeze(0))
    print("Pred:", probs)

    print("Evaluating accuracy...")
    acc = trained_model.evaluate_accuracy(trained_model.train_dataset)
    print("Train accuracy:", acc)

    print("Creating explainer...")

    print("Running LIME...")
    explanation, mask, highlighted = trained_model.perturbator.explainer.explain_instance(
        image_tensor,
        num_samples=300
    )

    plt.figure(figsize=(7,7))
    plt.imshow(highlighted)
    plt.title("LIME Explanation")
    plt.axis("off")
    plt.show()

    print("Done.")