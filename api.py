from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.utils.helper import load_joblib, load_params
from src.data_pipeline.data_defense import data_defense_checker
from src.preprocessing.preprocess import preprocess_process


# init models and params
params = load_params(param_dir = "config/params.yaml")
best_model = load_joblib(path = params["model_dump_path"] + "random_forest_best_model.pkl")

# create FastAPI object
app = FastAPI()

# init base model to define the data type
class APIData(BaseModel):
    person_age:int
    person_income :int
    person_home_ownership : str
    person_emp_length : int
    loan_intent :str
    loan_grade :str
    loan_amnt : int
    loan_int_rate : int  
    loan_percent_income : int
    cb_person_default_on_file :str
    cb_person_cred_hist_length : int

# for root dir website, do this process
@app.get("/")
def root():
    return {
        "msg": "Hello",
        "status": "success"
    }


# service for predict ML model based on input data from user
@app.post("/predict")
def predict(data: APIData):
    # Convert input data to DataFrame
    df_data = pd.DataFrame([data.dict()])
 
    # Validate using data checker
    try:
        data_defense_checker(input_data=df_data, params=params)
    except AssertionError as ae:
        return {
            "res": [],
            "error_msg": str(ae),
            "status_code": 400
        }
        
    # If valid, preprocess the data
    df_data = preprocess_process(data=df_data, params=params)
    
    # Predict the input data
    y_pred = best_model.predict(df_data)
    
    if y_pred[0] is None:
        return {
            "res": "Failed API",
            "NLP_user_prediction": None,
            "status_code": 500,
            "error_msg": "Prediction returned None."
        }
        
    return {
        "res": "Found API",
        "NLP_user_prediction": y_pred[0],
        "status_code": 200,
        "error_msg": ""
    }
