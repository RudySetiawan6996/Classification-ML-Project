import joblib
import pandas as pd


def test_features_shape():
    # arrange
    X_train = joblib.load("data/processed/X_train_final.pkl")
    X_valid = joblib.load("data/processed/X_valid_final.pkl")
    X_test = joblib.load("data/processed/X_test_final.pkl")
    
    # act 
    N_COLS_THRESH = 19
    
    # assert
    assert X_train.shape[1] == N_COLS_THRESH, "Input Train columns not match"
    assert X_valid.shape[1] == N_COLS_THRESH, "Input Validation columns not match"
    assert X_test.shape[1] == N_COLS_THRESH, "Input Test columns not match"


def test_null_values():
    # arrange
    X_train = joblib.load("data/processed/X_train_final.pkl")
    y_train = joblib.load("data/processed/y_train_final.pkl")
    
    X_valid = joblib.load("data/processed/X_valid_final.pkl")
    y_valid = joblib.load("data/processed/y_valid_final.pkl")
    
    X_test = joblib.load("data/processed/X_test_final.pkl")
    y_test = joblib.load("data/processed/y_test_final.pkl")
    
    # concat the data
    df_train = pd.concat([X_train, y_train], axis = 1)
    df_valid = pd.concat([X_valid, y_valid], axis = 1)
    df_test = pd.concat([X_test, y_test], axis = 1)
    
    # assert
    assert df_train.isna().values.any() == False, "There's a missing values in train data"
    assert df_valid.isna().values.any() == False, "There's a missing values in valid data"
    assert df_test.isna().values.any() == False, "There's a missing values in test data"
