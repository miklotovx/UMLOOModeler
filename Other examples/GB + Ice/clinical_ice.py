# Note:
# This implementation is illustrative and focuses on making the architectural roles explicit
# for UML extraction and framework instantiation (not on optimizing explanation quality).

# Environment:
# Conda env (python 3.10) with scikit-learn, pandas, numpy, matplotlib and JupyterLab.
# PyCEbox installed via pip for ICE (Individual Conditional Expectation) explanations.

# Jupyter Notebook Cell 1 - Imports + Database - <<DataSource>>
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

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
class ModelC_GB:
    def __init__(self, database: ClinicalDatabase):
        self.model_type = "GradientBoosting"
        self.database = database

        X = self.database.get_features(self.database.feature_names)
        y = self.database.get_labels()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model_c = GradientBoostingClassifier(random_state=42)
        model_c.fit(X_train, y_train)
        print("ModelC_GB successfully trained.")
        
        self.trained_model = TrainedModelC(model_c, X_train, y_train, list(X_train.columns))

    def build_model(self):
        return self.trained_model

# Jupyter Notebook Cell 3 - Trained GB Model - <<TrainedModel>>
class TrainedModelC:
    def __init__(self, model_c, X_train, y_train, feature_names):
        self.model = model_c
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.version = "1.0"
        self.explainer = IceExplainerC(self)
        print("TrainedModelC instantiated.")

    def predict(self, X):
        return self.model.predict(X)

# Jupyter Notebook Cell 4 - Ice Explainer + Main - <<IceExplainer>>
class IceExplainerC:
    def __init__(self, trained_model_c):
        self.trained_model = trained_model_c
        self.type = "ICE"
        print("IceExplainerC instantiated and connected to TrainedModelC.")

    def explain_instance(self, feature_name="mean radius"):
        from pycebox.ice import ice, ice_plot
        import matplotlib.pyplot as plt
        
        model_c = self.trained_model.model
        X_train = self.trained_model.X_train
        print(f"Generating ICE for feature: {feature_name}")
        
        ice_df = ice(
            data=X_train,
            column=feature_name,
            predict=lambda X: model_c.predict_proba(X)[:, 1]
        )
        
        plt.figure(figsize=(8, 5))
        ice_plot(ice_df, c="steelblue")
        plt.title(f"ICE Plot for '{feature_name}'")
        plt.xlabel(feature_name)
        plt.ylabel("Predicted probability (malignant)")
        plt.show()
        return ice_df

if __name__ == "__main__":
    db = ClinicalDatabase()

    model_c = ModelC_GB(db)
    trained_model_c = model_c.build_model()
    sample_instance_c = trained_model_c.X_train.iloc[0]
    print("\nSelected instance:")
    print(sample_instance_c[:3])

    print("Instance real target:", trained_model_c.y_train.iloc[0])

    print("Model prediction for the instance:")
    print(trained_model_c.predict(sample_instance_c.to_frame().T))

    print("Model Accuracy:")
    print(trained_model_c.model.score(trained_model_c.X_train, trained_model_c.y_train))

    trained_model_c.explainer.explain_instance(feature_name="mean radius")