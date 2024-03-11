import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from utils.file_util import file_path


def predict(model, train_x, val_x, train_y):
    model.fit(train_x, train_y)
    return model.predict(val_x)


def main():
    datafile_path = file_path("datasets", "melb_data.csv")
    dataset = pd.read_csv(datafile_path)
    features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
    x = dataset[features]
    y = dataset.Price

    train_x, val_x, train_y, val_y = train_test_split(
        x, y, train_size=0.8, test_size=0.2, random_state=0, shuffle=True
    )

    fittings = [None, 5, 50, 500, 5000]

    print("DecisionTreeRegressor")
    for max_leaf_nodes in fittings:
        predictions = predict(
            DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes),
            train_x,
            val_x,
            train_y,
        )
        mae = mean_absolute_error(val_y, predictions)
        print(f"Max Leaf Nodes: {max_leaf_nodes} \t MAE: {mae}")

    print("\n")

    print("RandomForestRegressor")
    predictions = predict(
        RandomForestRegressor(random_state=1), train_x, val_x, train_y
    )
    mae = mean_absolute_error(val_y, predictions)
    print(f"MAE: {mae}")


if __name__ == "__main__":
    main()
