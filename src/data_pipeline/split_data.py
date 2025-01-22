import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

def split_data(data: pd.DataFrame, params: dict) -> None:
    # set params
    data_dump_raw = params["dataset_dump_path"]["raw"]
    data_dump_interim = params["dataset_dump_path"]["interim"]

    target_col = params["target_col"]

    # set target col
    y = data[target_col]

    X = data.drop(columns = target_col, axis = 1)

    # validation
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # save the X, y to pkl
    joblib.dump(X, data_dump_raw + "X.pkl")
    joblib.dump(y, data_dump_raw + "y.pkl")

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        # stratify = y,
                                                        test_size = 0.2,
                                                        random_state = 42)

    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test,
                                                        # stratify = y_test,
                                                        test_size = 0.2,
                                                        random_state = 42)

    # Validasi
    print('X_train shape :', X_train.shape)
    print('y_train shape :', y_train.shape)
    print('X_valid shape  :', X_valid.shape)
    print('y_valid shape  :', y_valid.shape)
    print('X_test shape  :', X_test.shape)
    print('y_test shape  :', y_test.shape)

    # dump
    joblib.dump(X_train, data_dump_interim + "X_train.pkl")
    joblib.dump(y_train, data_dump_interim + "y_train.pkl")
    joblib.dump(X_valid, data_dump_interim + "X_valid.pkl")
    joblib.dump(y_valid, data_dump_interim + "y_valid.pkl")
    joblib.dump(X_test, data_dump_interim + "X_test.pkl")
    joblib.dump(y_test, data_dump_interim + "y_test.pkl")
    
