from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_data():
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="MedHouseVal")
    return X, y
