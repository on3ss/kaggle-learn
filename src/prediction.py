"""Module for making basic predictions using Decision Tree Regressor."""

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def calculate_mae(model, data: pd.DataFrame):
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
    model.fit(train_x, train_y)
    predictions = model.predict(val_x)
    return mean_absolute_error(val_y, predictions)
