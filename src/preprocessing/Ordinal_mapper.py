import pandas as pd
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
import joblib
import yaml
from src.utils.helper import load_params, load_joblib

params = load_params(param_dir = "config/params.yaml")

def save_encoders(data: pd.DataFrame, col_ohe: list, col_ordinal: list):
    encoders = {}

    # Initialize and fit OneHotEncoders for all columns
    for col in col_ohe:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(data[[col]])
        encoders[f'ohe_{col}'] = ohe

    # Initialize and fit OrdinalEncoders for all columns
    for col in col_ordinal:
        values = [sorted(list(data[col].unique()))]
        ordinal = OrdinalEncoder(categories=values)
        ordinal.fit(data[[col]])
        encoders[f'ordinal_{col}'] = ordinal
    
    encoder={'ohe':ohe,'ordinal':ordinal}
    joblib.dump(encoder, params["dataset_dump_path"]["processed"] + "ohe_ordinal_model.pkl")
    
    return ohe,ordinal

def preprocess_ordinal(data:pd.DataFrame,params:dict):
    col=params['label_ordinal_columns']
    encoded=pd.DataFrame(index=data.index)
    for i in col:
        ordinal_feat=ordinal.fit_transform(data[[i]])
        ordinal_df=pd.DataFrame(ordinal_feat,columns=[i],index=data.index)
        ordinal_df.rename(columns={i:f'{i}_label'},inplace=True)
        encoded=pd.concat([encoded,ordinal_df],axis=1)
        data=data.drop(columns=i)
    data=pd.concat([data,encoded],axis=1)
    
    
    return data