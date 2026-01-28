# Note:
# This implementation is illustrative and focuses on making the architectural roles explicit
# for UML extraction and framework instantiation (not on optimizing explanation quality).

# Environment:
# Conda env (python 3.10) with scikit-learn, pandas, numpy, matplotlib and JupyterLab.
# SHAP installed via conda-forge.

# Jupyter Notebook Cell 1 - Imports + Database - <<DataSource>>
import warnings
warnings.filterwarnings("ignore")
import shap
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class ClinicalDatabase:
    def __init__(self):
        dataset = load_breast_cancer(as_frame=True)
        self.data = dataset.frame.drop(columns=["target"])
        self.labels = dataset.frame["target"]
        self.feature_names = list(self.data.columns)

    def get_features(self, feature_list):
        return self.data[feature_list]

    def get_labels(self):
        return self.labels

# Jupyter Notebook Cell 2 - Model instantiation and training - <<ModelDefinition>>
class ModelB_RF:
    def __init__(self, database: ClinicalDatabase):
        self.model_type = "RandomForest"
        self.database = database

        X = self.database.get_features(self.database.feature_names)
        y = self.database.get_labels()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model_b = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        model_b.fit(X_train, y_train)

        print("ModelB_RF successfully trained.")

        self.trained_model = TrainedModelB(
            model_b,
            X_train,
            y_train,
            list(X_train.columns)
        )

    def build_model(self):
        return self.trained_model

# Jupyter Notebook Cell 3 - Trained RF Model - <<TrainedModel>>
class TrainedModelB:
    def __init__(self, model_b, X_train, y_train, feature_names):
        self.model = model_b
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.explainer = ShapExplainerB(self)
        print("TrainedModelB instantiated.")

    def predict(self, X):
        return self.model.predict(X)

# Jupyter Notebook Cell 4 - Shap Explainer + Main - <<ShapExplainer>>
class ShapExplainerB:
    def __init__(self, trained_model_b):
        self.trained_model = trained_model_b
        self.type = "TreeExplainer"
        self.shap_values = None
        print("ShapExplainerB instantiated and connected to TrainedModelB.")

    def explain_instance(self, instance):
        
        print("Executing SHAP explanation...")
        explainer = shap.TreeExplainer(self.trained_model.model)
        if not isinstance(instance, pd.DataFrame):
            instance_df = pd.DataFrame([instance], columns=self.trained_model.feature_names)
        else:
            instance_df = instance

        shap_values = explainer(instance_df)
        self.shap_values = shap_values
        print("Explanation completed successfully.")
        if len(shap_values.values.shape) == 3:
            shap_values_single = shap_values[:, :, 1]
        else:
            shap_values_single = shap_values

        shap.plots.waterfall(shap_values_single[0])
        return shap_values

if __name__ == "__main__":
    db = ClinicalDatabase()
    model_b = ModelB_RF(db)
    trained_model_b = model_b.build_model()
    sample_instance_b = trained_model_b.X_train.iloc[0]
    print("Selected instance:")
    print(sample_instance_b)

    print("Instance real target:")
    print(trained_model_b.y_train.iloc[0])

    print("Model prediction for the instance:")
    print(trained_model_b.predict(sample_instance_b.to_frame().T))

    print("Model Accuracy:")
    print(trained_model_b.model.score(trained_model_b.X_train, trained_model_b.y_train))

    trained_model_b.explainer.explain_instance(sample_instance_b)