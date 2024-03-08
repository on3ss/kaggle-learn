"""Module for making basic predictions using Decision Tree Regressor."""

from typing import Union
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def predict(data: pd.DataFrame, max_leaf_nodes: Union[int, None] = None):
    """
    Trains a Decision Tree Regressor model on
    housing data and prints predictions for the first 5 houses.

    Parameters:
        data: DataFrame containing housing data with columns
        'Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', and 'Price'.

    Returns:
        None
    """
    features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
    x = data[features]
    y = data.Price

    train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
    model.fit(train_x, train_y)
    print("Predicted prices:")
    predictions = model.predict(val_x)
    return mean_absolute_error(val_y, predictions)


def run_predictions(data: pd.DataFrame):
    """
    Run predictions for different value of max leaf nodes
    """
    for leaf_nodes in [None, 5, 50, 500, 5000]:
        print(
            f"Leaf Node: {leaf_nodes} | Mean Absolute Error: {predict(data, leaf_nodes)}"
        )
