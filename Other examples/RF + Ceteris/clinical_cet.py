# Note:
# This implementation is illustrative and focuses on making the architectural roles explicit
# for UML extraction and framework instantiation (not on optimizing explanation quality).

# Environment:
# Conda env (python 3.10) with scikit-learn, pandas, numpy, matplotlib and JupyterLab.
# DALEX installed via pip for Ceteris Paribus explanations.

# Jupyter Notebook Cell 1 - Imports + Database - <<DataSource>>
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
class ModelD_LR:
    def __init__(self, database: ClinicalDatabase):
        self.model_type = "LogisticRegression"
        self.database = database

        X = self.database.get_features(self.database.feature_names)
        y = self.database.get_labels()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model_d = LogisticRegression(max_iter=2000, solver='lbfgs')
        model_d.fit(X_train, y_train)

        print("ModelD_LR successfully trained.")

        self.trained_model = TrainedModelD(
            model_d,
            X_train,
            y_train,
            list(X_train.columns)
        )

    def build_model(self):
        return self.trained_model

# Jupyter Notebook Cell 3 - Trained LR Model - <<TrainedModel>>
class TrainedModelD:
    def __init__(self, model_d, X_train, y_train, feature_names):
        self.model = model_d
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.version = "1.0"
        self.explainer = CeterisExplainerD(self)
        print("TrainedModelD instantiated.")

    def predict(self, X):
        return self.model.predict(X)

# Jupyter Notebook Cell 4 - Ceteris Explainer + Main - <<CeterisExplainer>>
class CeterisExplainerD:
    def __init__(self, trained_model_d):
        self.trained_model = trained_model_d
        self.type = "CeterisParibus"
        print("CeterisExplainerD instantiated and connected to TrainedModelD.")

    def explain_instance(self, instance, feature_name="mean radius"):
        import dalex as dx
        import matplotlib.pyplot as plt
        
        print("Executing Dalex Ceteris Paribus explanation...")
        model_d = self.trained_model.model
        X_train = self.trained_model.X_train
        y_train = self.trained_model.y_train
        
        explainer = dx.Explainer(model_d, X_train, y_train, label="Logistic Regression (Clinical)", verbose=False)
        
        if not isinstance(instance, pd.DataFrame):
            instance_df = pd.DataFrame([instance], columns=self.trained_model.feature_names)
        else:
            instance_df = instance

        cp_profile = explainer.predict_profile(instance_df, variables=[feature_name])

        cp_profile.plot() 
        plt.title(f"Ceteris Paribus Plot for '{feature_name}'")
        plt.show()
        return cp_profile

if __name__ == "__main__":
    db = ClinicalDatabase()
    model_d = ModelD_LR(db)
    trained_model_d = model_d.build_model()
    sample_instance_d = trained_model_d.X_train.iloc[0]
    
    print("\nSelected instance:")
    print(sample_instance_d[:3]) 

    print("Instance real target:", trained_model_d.y_train.iloc[0])

    print("Model prediction for the instance:")
    print(trained_model_d.predict(sample_instance_d.to_frame().T))

    print("Model Accuracy:")
    print(trained_model_d.model.score(trained_model_d.X_train, trained_model_d.y_train))

    trained_model_d.explainer.explain_instance(sample_instance_d, feature_name="mean radius")