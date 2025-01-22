import yaml
import pandas as pd
import joblib


def load_params(param_dir: str) -> dict:
    with open(param_dir, "r") as file:
        params = yaml.safe_load(file)
        
    return params


def dump_joblib(data, path: str) -> None:
    joblib.dump(data, path)
    

def load_joblib(path: str):
    return joblib.load(path)


def read_data(filename: str, params: dict) -> pd.DataFrame:
    data = pd.read_csv(filename)
    
    print(f"Data shape: {data.shape}")
    
    dump_path = params["dataset_dump_path"]["raw"] + "raw_data.pkl"
    joblib.dump(data, dump_path)
    
    return data


def split_num_cat(data: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    # get cat data
    data_cat = data[params["object_columns"]].copy()
    
    # get num data
    data_num = data[params["feature_num_columns"]].copy()
    
    return data_cat, data_num


def concat_data(data_cat: pd.DataFrame, data_num: pd.DataFrame) -> pd.DataFrame:
    final_data = pd.concat([data_cat, data_num], axis = 1)
    
    return final_data
