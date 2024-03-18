import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import prediction
from utils import file_util


def main():
    """
    Main function to perform housing price prediction.

    Reads the Melbourne housing dataset, iterates over different values of max_leaf_nodes for DecisionTreeRegressor,
    and fits both DecisionTreeRegressor and RandomForestRegressor models to predict housing prices.
    Prints mean absolute error (MAE) for each model.
    """
    melbourne_data = pd.read_csv(file_util.file_path("datasets", "melb_data.csv"))
    fittings = [None, 5, 50, 500, 5000]

    print("DecisionTreeRegressor")
    for max_leaf_nodes in fittings:
        dt_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
        mae = prediction.calculate_mae(dt_model, melbourne_data)
        print(f"Max Leaf Nodes: {max_leaf_nodes} \t MAE: {mae}")

    print("\nRandomForestRegressor")
    rf_model = RandomForestRegressor(random_state=1)
    mae = prediction.calculate_mae(rf_model, melbourne_data)
    print(f"MAE: {mae}")


if __name__ == "__main__":
    main()
