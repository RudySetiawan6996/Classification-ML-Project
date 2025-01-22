from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from src.utils.helper import dump_joblib
from sklearn.metrics import *
import pandas as pd


def modeling_multiple(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    dt_baseline = RandomForestClassifier()
    logistic_baseline = LogisticRegression()
    SVM_baseline = SVC()
    
    dt_baseline.fit(X_train, y_train)
    logistic_baseline.fit(X_train, y_train)
    SVM_baseline.fit(X_train, y_train)
    
    dump_joblib(dt_baseline, params["model_dump_path"] + "dt_baseline.pkl")
    dump_joblib(logistic_baseline, params["model_dump_path"] + "logistic_baseline.pkl")
    dump_joblib(SVM_baseline, params["model_dump_path"] + "SVM_baseline.pkl")

    return dt_baseline, logistic_baseline, SVM_baseline