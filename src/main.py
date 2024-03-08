"""
Predict housing prices using decision tree and random forest regressors.
"""

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
    melbourne_data_file_path = file_util.file_path("datasets", "melb_data.csv")
    melbourne_data = pd.read_csv(melbourne_data_file_path)

    max_leaf_nodes_list = [None, 5, 50, 500, 5000]

    print("DecisionTreeRegressor")
    for max_leaf_nodes in max_leaf_nodes_list:
        mae = prediction.calculate_mae(
            DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes), melbourne_data
        )
        print(f"Max Leaf Nodes: {max_leaf_nodes} \t MAE: {mae}")

    print("\n")

    print("RandomForestRegressor")
    mae = prediction.calculate_mae(RandomForestRegressor(), melbourne_data)
    print(f"MAE: {mae}")


if __name__ == "__main__":
    main()
