import yaml
import pandas as pd
import joblib
from sqlalchemy import create_engine


def init_engine():

    cred = {
            'host': "ep-autumn-bar-a1ubq400.ap-southeast-1.aws.neon.tech",
            'user': "siswa_bfp",
            'pass': "bfp_aksel_keren",
            'db': "credit_risk_db",
            'port': 5432
        }

    uri = f"postgresql://{cred['user']}:{cred['pass']}@{cred['host']}:{cred['port']}/{cred['db']}?sslmode=require"

    conn = create_engine(uri)

    return conn


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
