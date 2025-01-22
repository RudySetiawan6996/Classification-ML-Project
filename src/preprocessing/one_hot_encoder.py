import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess_ohe(data:pd.DataFrame,params:dict):
    col=params['ohe_columns']
    encoded=pd.DataFrame(index=data.index)
    for i in col:
        ohe=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
        ohe.fit(data[[i]])
        ohe_feat=ohe.fit_transform(data[[i]])
        ohe_cols=ohe.categories_[0]
        ohe_df=pd.DataFrame(ohe_feat,columns=ohe_cols,index=data.index)
        encoded=pd.concat([encoded,ohe_df],axis=1)
        data=data.drop(columns=i)
    data=pd.concat([data,encoded],axis=1)
    
    
    return data