
#Importando as bibliotecas necessárias
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from urllib.parse import urlparse

#Manipulação de dados
import pandas as pd

# Criação do modelo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Métricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Ignorar avisos de atualização, etc
import warnings
warnings.filterwarnings("ignore")


import logging
import sys

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Initiate MLflow client
client = MlflowClient()

# Get parsed experiment name
experiment_name = 'Diabetes_Classification'

# Create MLflow experiment
try:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = client.get_experiment_by_name(experiment_name)
except:
    experiment = client.get_experiment_by_name(experiment_name)

mlflow.set_experiment(experiment_name)

# Print experiment details
print(f"Name: {experiment_name}")
print(f"Experiment_id: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")
print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
print(f"Tracking uri: {mlflow.get_tracking_uri()}")

tags = {
        "Projeto": "Experiments MLOps",
        "team": "Data Science - Qexpert",
        "dataset": "Diabetes"
       }

def metricas(y_test, y_predict):
    acuracia = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)
    return acuracia, precision, recall, f1


def modelo1():
    #Criação do modelo
    max_depth = 15
    balanced = 1
    n_estimators = 300
    balanced = "balanced" if balanced == 1 else None
    clf = RandomForestClassifier(random_state=42, n_estimators=n_estimators, class_weight=balanced, max_depth=max_depth)
    clf.fit(x_train, y_train)
    #balanced = "none" if balanced == None else "balanced"
    hp = {
        "max_depth": max_depth,
        "balanced": balanced,
        "n_estimators": n_estimators
    }
    return hp, clf

def modelo2():
    #Criação do modelo
    max_depth = 10
    balanced = 1
    n_estimators = 150
    balanced = "balanced" if balanced == 1 else None
    clf = RandomForestClassifier(random_state=42, n_estimators=n_estimators, class_weight=balanced, max_depth=max_depth)
    clf.fit(x_train, y_train)
    #balanced = "none" if balanced == None else "balanced"
    hp = {
        "max_depth": max_depth,
        "balanced": balanced,
        "n_estimators": n_estimators
    }
    return hp, clf


def modelo3():
    #Criação do modelo
    max_depth = 10
    balanced = 0
    n_estimators = 500
    balanced = "balanced" if balanced == 1 else None
    clf = RandomForestClassifier(random_state=42, n_estimators=n_estimators, class_weight=balanced, max_depth=max_depth)
    clf.fit(x_train, y_train)
    #balanced = "none" if balanced == None else "balanced"
    hp = {
        "max_depth": max_depth,
        "balanced": balanced,
        "n_estimators": n_estimators
    }
    return hp, clf

def previsao(x_test, modelo):
    y_pred = modelo.predict(x_test)
    return y_pred

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    df = pd.read_csv("data/processed/train.csv")

    #TO DO: Cross Validation
    train, test = train_test_split(df, random_state=42)
    x_train = train.drop(columns=["Outcome"])
    x_test = test.drop(columns=["Outcome"])
    y_train = train[["Outcome"]]
    y_test = test[["Outcome"]]
    
    with mlflow.start_run(run_name='RandomForest_V1'):
        warnings.filterwarnings("ignore")
        mlflow.set_tags(tags)
        hp, clf = modelo1()
        y_pred = previsao(x_test, clf)
        #Métricas
        acuracia, precision, recall, f1 = metricas(y_test, y_pred)
        print("Acurácia: {}\nPrecision: {}\nRecall: {}\nF1-Score: {}".
             format(acuracia, precision, recall, f1))
        #Registro dos hiper-parametros
        for attr, value in hp.items():
            mlflow.log_param(attr, value)
        #Registro das métricas
        mlflow.log_metric("Acuracia", acuracia)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1-Score", f1)
        #Registro do modelo
        mlflow.sklearn.log_model(clf, "model")
        mlflow.log_artifact(local_path='./train.py', artifact_path='code')
        model_uri = mlflow.get_artifact_uri("model")
        print(f'Model saved in {model_uri}')
        # Get IDs of current experiment run
        exp_id = experiment.experiment_id
        run_id = mlflow.active_run().info.run_id

    with mlflow.start_run(run_name='RandomForest_V2'):
        warnings.filterwarnings("ignore")
        mlflow.set_tags(tags)
        hp, clf = modelo2()
        y_pred = previsao(x_test, clf)
        #Métricas
        acuracia, precision, recall, f1 = metricas(y_test, y_pred)
        print("Acurácia: {}\nPrecision: {}\nRecall: {}\nF1-Score: {}".
             format(acuracia, precision, recall, f1))
        #Registro dos hiper-parametros
        for attr, value in hp.items():
            mlflow.log_param(attr, value)
        #Registro das métricas
        mlflow.log_metric("Acuracia", acuracia)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1-Score", f1)
        #Registro do modelo
        mlflow.sklearn.log_model(clf, "model")
        mlflow.log_artifact(local_path='./train.py', artifact_path='code')
        model_uri = mlflow.get_artifact_uri("model")
        print(f'Model saved in {model_uri}')
        # Get IDs of current experiment run
        exp_id = experiment.experiment_id
        run_id = mlflow.active_run().info.run_id

    with mlflow.start_run(run_name='RandomForest_V3'):
        warnings.filterwarnings("ignore")
        mlflow.set_tags(tags)
        hp, clf = modelo3()
        y_pred = previsao(x_test, clf)
        #Métricas
        acuracia, precision, recall, f1 = metricas(y_test, y_pred)
        print("Acurácia: {}\nPrecision: {}\nRecall: {}\nF1-Score: {}".
             format(acuracia, precision, recall, f1))
        #Registro dos hiper-parametros
        for attr, value in hp.items():
            mlflow.log_param(attr, value)
        #Registro das métricas
        mlflow.log_metric("Acuracia", acuracia)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1-Score", f1)
        #Registro do modelo
        mlflow.sklearn.log_model(clf, "model")
        mlflow.log_artifact(local_path='./train.py', artifact_path='code')
        model_uri = mlflow.get_artifact_uri("model")
        print(f'Model saved in {model_uri}')
        # Get IDs of current experiment run
        exp_id = experiment.experiment_id
        run_id = mlflow.active_run().info.run_id