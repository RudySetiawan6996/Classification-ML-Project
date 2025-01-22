from sklearn.ensemble import RandomForestClassifier
from src.utils.helper import dump_joblib
from sklearn.metrics import *
import pandas as pd


def modeling_linreg(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    rf_clf = RandomForestClassifier()

    rf_clf.fit(X_train, y_train)
    
    dump_joblib(rf_clf, params["model_dump_path"] + "vanilla_rf_model.pkl")
    
    return rf_clf


def predict_baseline(model, X_valid, y_valid):
    y_pred_dummy = model.predict(X_valid)
    
    print(f"Accuracy: {accuracy_score(y_valid, y_valid)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")