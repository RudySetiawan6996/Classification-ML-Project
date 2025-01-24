import pandas as pd
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
import joblib
import yaml
from src.utils.helper import load_params, load_joblib

params = load_params(param_dir = "config/params.yaml")

def ohe_encoders(data: pd.DataFrame, col_ohe: list):
    ohe = {}

    for col in col_ohe:
        ohe[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe[col].fit(data[[col]])

    # Save the dictionary of encoders
    joblib.dump(ohe, params["dataset_dump_path"]["processed"] + "/ohe_model.pkl")
    return ohe
    
    

def preprocess_ohe(data:pd.DataFrame,params:dict,ohe):
    col=params['ohe_columns']
    encoded=pd.DataFrame(index=data.index)
    for i in col:
        ohe_feat=ohe[i].fit_transform(data[[i]])
        ohe_cols=ohe[i].categories_[0]
        ohe_df=pd.DataFrame(ohe_feat,columns=ohe_cols,index=data.index)
        encoded=pd.concat([encoded,ohe_df],axis=1)
        data=data.drop(columns=i)
    data=pd.concat([data,encoded],axis=1)
    
    
    return data