import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def calculate_mae(model, data: pd.DataFrame) -> float:
    """
    Trains a model on housing data and calculates Mean Absolute Error (MAE).

    Parameters:
        model: Regression model to train.
        data: DataFrame containing housing data with columns:
              'Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', and 'Price'.

    Returns:
        float: Mean Absolute Error (MAE) of the model.
    """
    features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
    X = data[features]
    y = data["Price"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    return mae
