import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
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

    train_x_plus = train_x.copy()
    val_x_plus = val_x.copy()

    cols_with_missing = [col for col in train_x.columns if train_x[col].isnull().any()]
    for col in cols_with_missing:
        train_x_plus[col + "_was_missing"] = train_x_plus[col].isnull()
        val_x_plus[col + "_was_missing"] = val_x_plus[col].isnull()

    my_imputer = SimpleImputer()
    imputed_train_x = pd.DataFrame(my_imputer.fit_transform(train_x_plus))
    imputed_val_x = pd.DataFrame(my_imputer.fit_transform(val_x_plus))

    imputed_train_x.columns = train_x_plus.columns
    imputed_val_x.columns = val_x_plus.columns

    fittings = [None, 5, 50, 500, 5000]

    print("DecisionTreeRegressor")
    for max_leaf_nodes in fittings:
        predictions = predict(
            DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes),
            imputed_train_x,
            imputed_val_x,
            train_y,
        )
        mae = mean_absolute_error(val_y, predictions)
        print(f"Max Leaf Nodes: {max_leaf_nodes} \t MAE: {mae}")

    print("\n")

    print("RandomForestRegressor")
    predictions = predict(
        RandomForestRegressor(random_state=1), imputed_train_x, imputed_val_x, train_y
    )
    mae = mean_absolute_error(val_y, predictions)
    print(f"MAE: {mae}")


if __name__ == "__main__":
    main()
