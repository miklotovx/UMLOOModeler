# Note:
# This implementation is illustrative and focuses on making the architectural roles explicit
# for UML extraction and framework instantiation (not on optimizing explanation quality).

# Environment:
# Conda env (python 3.10) with scikit-learn, pandas, numpy, matplotlib and JupyterLab.
# LIME installed via pip. 

# Jupyter Notebook Cell 1 - Imports + Database - <<DataSource>>
import warnings                                                                      
warnings.filterwarnings("ignore")
import pandas as pd    
import numpy as np                                        
from sklearn.datasets import load_breast_cancer               
from sklearn.model_selection import train_test_split         
from sklearn.neural_network import MLPClassifier   
from lime.lime_tabular import LimeTabularExplainer
         
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
class ModelA_MLP:
    def __init__(self, database: ClinicalDatabase):
        self.model_type = "MLP"
        self.database = database
        X = self.database.get_features(self.database.feature_names)
        y = self.database.get_labels()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model_a = MLPClassifier(max_iter=500, random_state=42)
        model_a.fit(X_train, y_train)
        print("ModelA_MLP successfully trained.")
        self.trained_model = TrainedModelA(model_a, X_train, y_train, list(X_train.columns))

    def build_model(self):
        return self.trained_model
        
# Jupyter Notebook Cell 3 - Trained MLP Model - <<TrainedModel>>
class TrainedModelA:                                               
    def __init__(self, model_a, X_train, y_train, feature_names):  
        self.model = model_a                                       
        self.X_train = X_train                                     
        self.y_train = y_train                                     
        self.feature_names = feature_names                         
        self.perturbator = PerturbatorA(self)                       
        print("TrainedModelA instantiated.")

    def predict(self, X):                                          
        return self.model.predict(X)                               

# Jupyter Notebook Cell 4 - Lime Perturbator - <<Perturbator>>
class PerturbatorA:
    def __init__(self, trained_model_a):
        self.trained_model = trained_model_a
        self.explainer = LimeExplainerA(self)
        print("PerturbatorA created and connected to TrainedModelA.")

    def perturb(self, instance, num_samples=50):

        print("Perturbing...")
        perturbed_data = []
        for _ in range(num_samples):
            noise = np.random.normal(0, 0.1, size=instance.shape)
            perturbed_instance = instance + noise
            perturbed_data.append(perturbed_instance)
        
        df_perturbed = pd.DataFrame(perturbed_data, columns=self.trained_model.feature_names)
        print(f"{num_samples} perturbed instances.")
        return df_perturbed

# Jupyter Notebook Cell 5 - Lime Explainer + Main - <<LimeExplainer>>
class LimeExplainerA:
    def __init__(self, perturbator_a):
        self.perturbator = perturbator_a
        print("LimeExplainerA instantiated and connected to PerturbatorA.")

    def explain_instance(self, instance):

        print("Executing LIME explanation.")
        model_a = self.perturbator.trained_model.model
        X_train = self.perturbator.trained_model.X_train
        feature_names = self.perturbator.trained_model.feature_names

        explainer_a = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=['malignant', 'benign'],
            mode="classification"
        )

        instance_array = np.array(instance).reshape(1, -1)
        explanation_a = explainer_a.explain_instance(
            data_row=instance_array[0],
            predict_fn=model_a.predict_proba
        )

        explanation_a.show_in_notebook(show_table=True)
        return explanation_a

if __name__ == "__main__":
    db = ClinicalDatabase()
    model_a = ModelA_MLP(db)                    
    trained_model_a = model_a.build_model()  
    sample_instance_a = trained_model_a.X_train.iloc[0]
    print("Selected instance:")
    print(sample_instance_a)

    print("Instance real target:")
    print(trained_model_a.y_train.iloc[0])

    print("Model prediction for the instance:")
    print(trained_model_a.predict(sample_instance_a.to_frame().T))

    print("Model Accuracy:")
    print(trained_model_a.model.score(trained_model_a.X_train, trained_model_a.y_train))

    print("MLP architecture:")
    print(trained_model_a.model.hidden_layer_sizes)

    trained_model_a.perturbator.explainer.explain_instance(sample_instance_a)