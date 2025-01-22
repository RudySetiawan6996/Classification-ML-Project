import pandas as pd

def custom_label_encoder(data:pd.DataFrame,params):
    col=params['label_mapper_columns']
    MAPPER_VALUE = {
        "N": 0,
        "Y": 1
    }

    for i in col:
        data[i] = data[i].replace(MAPPER_VALUE)

    return data