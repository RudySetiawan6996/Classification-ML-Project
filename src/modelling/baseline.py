from sklearn.dummy import DummyClassifier
from src.utils.helper import dump_joblib
from sklearn.metrics import *
import pandas as pd


def modeling_baseline(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    dummy_clf=DummyClassifier(strategy='most_frequent')

    dummy_clf.fit(X_train, y_train)
    
    dump_joblib(dummy_clf, params["model_dump_path"] + "baseline_model.pkl")
    
    return dummy_clf


def predict_baseline(model, X_valid, y_valid):
    y_pred_dummy = model.predict(X_valid)
    
    print(f"accuracy: {accuracy_score(y_valid, y_pred_dummy)}")
    print(f"recall: {recall_score(y_valid, y_pred_dummy)}")
