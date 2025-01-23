from src.utils.helper import load_joblib


def test_uppercase():
    assert "espresso on the rock".upper() == "ESPRESSO ON THE ROCK", "Text not match!"
    

def test_reversed():
    assert list(reversed([1, 2, 3, 4])) == [4, 3, 2, 1]


def test_features_shape():
    # arrange
    X_train = load_joblib("data/processed/X_train_final.pkl")
    X_valid = load_joblib("data/processed/X_valid_final.pkl")
    X_test = load_joblib("data/processed/X_test_final.pkl")
    
    # act
    N_COLS_THRESH = 19
    
    # assert
    assert X_train.shape[1] == N_COLS_THRESH, "Input Train columns not match"
    assert X_valid.shape[1] == N_COLS_THRESH, "Input Validation columns not match"
    assert X_test.shape[1] == N_COLS_THRESH, "Input Test columns not match"
