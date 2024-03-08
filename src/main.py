import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import basic_prediction
from utils import file_util


def main():
    melbourne_data_file_path = file_util.file_path("datasets", "melb_data.csv")
    melbourne_data = pd.read_csv(melbourne_data_file_path)

    max_leaf_nodes_list = [None, 5, 50, 500, 5000]

    print("DecisionTreeRegressor")
    for max_leaf_nodes in max_leaf_nodes_list:
        mae = basic_prediction.predict(
            DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes), melbourne_data
        )
        print(f"Max Leaf Nodes: {max_leaf_nodes} \t MAE: {mae}")

    print("\n")

    print("RandomForestRegressor")
    mae = basic_prediction.predict(RandomForestRegressor(), melbourne_data)
    print(f"MAE: {mae}")


if __name__ == "__main__":
    main()
