import pandas as pd


def data_defense_checker(input_data: pd.DataFrame, params: dict) -> None:
    try:
        print("===== Start Data Defense Checker =====")
        # check data types
        assert input_data[params["features"]].select_dtypes("object").columns.to_list() == params["object_columns"], "an error occurs in object columns"
        assert input_data[params["features"]].select_dtypes(include=['int', 'float']).columns.to_list() == params["numeric_columns"], "an error occurs in integer columns"

        # check values
        assert set(input_data.person_home_ownership).issubset(set(params["value_person_home_ownership"])), "an error occurs on person_home_ownership column"
        assert set(input_data.loan_intent).issubset(set(params["value_loan_intent"])), "an error occurs on loan_intent column"
        assert set(input_data.cb_person_default_on_file).issubset(set(params["value_cb_person_default_on_file"])), "an error occurs on cb_person_default_on_file column"


    except Exception:
        raise Exception("Failed Data Defense Checker")

    finally:
        print("===== Finish Data Defense Checker =====")
