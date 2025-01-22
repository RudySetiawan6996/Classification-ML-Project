from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import numpy as np
import pandas as pd
from src.utils.helper import load_joblib, dump_joblib


def hyperparam_process(model_path: str, X_train: pd.DataFrame, y_train: pd.Series):
    model = load_joblib(path = model_path)
    
    random_grid = {
        'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]
    }
    
    k_folds = KFold(n_splits = 5)
    
    best_rf_random=RandomizedSearchCV(estimator=RandomForestClassifier,
                                  param_distributions=random_grid,
                                  cv=k_folds,
                                  verbose=3,scoring='recall')
    
    best_rf_random.fit(X_train, y_train)
    
    return best_rf_random.best_params_


def best_model_train(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    best_model =RandomForestClassifier(n_estimators=200,
                                    max_features='sqrt',
                                    max_depth=60)
    
    best_model.fit(X_train, y_train)
    
    dump_joblib(best_model, params["model_dump_path"] + "best_model.pkl")
    
    return best_model


def predict_best_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")