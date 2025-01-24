import pandas as pd
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
import joblib
import yaml
from src.utils.helper import load_params, load_joblib

params = load_params(param_dir = "config/params.yaml")

def ordinal_encoders(data: pd.DataFrame, col_ordinal: list):
    ordinal = {}

    for col in col_ordinal:
        values = [sorted(list(data[col].unique()))]
        ordinal[col] = OrdinalEncoder(categories=values)
        ordinal[col].fit(data[[col]])

    # Save the dictionary of encoders
    joblib.dump(ordinal, params["dataset_dump_path"]["processed"] + "/ordinal_model.pkl")
    return ordinal
    
    

def preprocess_ordinal(data:pd.DataFrame,params:dict,ordinal):
    col=params['label_ordinal_columns']
    encoded=pd.DataFrame(index=data.index)
    for i in col:
        ordinal_feat=ordinal[i].fit_transform(data[[i]])
        ordinal_df=pd.DataFrame(ordinal_feat,columns=[i],index=data.index)
        ordinal_df.rename(columns={i:f'{i}_label'},inplace=True)
        encoded=pd.concat([encoded,ordinal_df],axis=1)
        data=data.drop(columns=i)
    data=pd.concat([data,encoded],axis=1)
    
    
    return data