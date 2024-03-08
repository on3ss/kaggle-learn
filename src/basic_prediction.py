import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from utils import file_util


def predict():
    melbourne_data_file_path = file_util.file_path("datasets", "melb_data.csv")
    melbourne_data = pd.read_csv(melbourne_data_file_path)

    features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]

    x = melbourne_data[features]
    y = melbourne_data.Price

    model = DecisionTreeRegressor(random_state=1)
    model.fit(x, y)

    print("Making predictions for the following 5 houses:")
    print(x.head())
    print("The predictions are")
    print(model.predict(x.head()))
