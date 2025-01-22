from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

def preprocess_ordinal(data:pd.DataFrame,params:dict):
    col=params['label_ordinal_columns']
    encoded=pd.DataFrame(index=data.index)
    for i in col:
        values = [sorted(list(data[i].unique()))]
        ordinal=OrdinalEncoder(categories=values)
        ordinal_feat=ordinal.fit_transform(data[[i]])
        ordinal_df=pd.DataFrame(ordinal_feat,columns=[i],index=data.index)
        ordinal_df.rename(columns={i:f'{i}_label'},inplace=True)
        encoded=pd.concat([encoded,ordinal_df],axis=1)
        data=data.drop(columns=i)
    data=pd.concat([data,encoded],axis=1)
    
    
    return data