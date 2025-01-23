from sklearn.metrics import accuracy_score
from src.utils.helper import load_joblib


def test_model_performance():
    model = load_joblib("models/random_forest_best_model.pkl")
    
    X_test = load_joblib("data/processed/X_test_final.pkl")
    y_test = load_joblib("data/processed/y_test_final.pkl")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    THRESHOLD_ACCURACY = 0.85
    
    assert accuracy > THRESHOLD_ACCURACY, f"Accuracy result is too small: {accuracy}"
