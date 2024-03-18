import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.impute import SimpleImputer
from utils.file_util import file_path


def predict(
    model: sklearn.base.BaseEstimator,
    train_x: pd.DataFrame,
    val_x: pd.DataFrame,
    train_y: pd.Series,
) -> pd.Series:
    """
    Predict the house prices using the given model and data.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The machine learning model to use for prediction.
    train_x : pandas.DataFrame
        The training data features.
    val_x : pandas.DataFrame
        The validation data features.
    train_y : pandas.Series
        The training data target.

    Returns
    -------
    pandas.Series
        The predicted house prices.

    """
    model.fit(train_x, train_y)
    return model.predict(val_x)


def preprocess_data(
    train_x: pd.DataFrame, val_x: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify and impute missing values in the given dataframes.

    Parameters
    ----------
    train_x : pd.DataFrame
        The training data features.
    val_x : pd.DataFrame
        The validation data features.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The preprocessed training and validation dataframes.
    """
    # Identify columns with missing values
    cols_with_missing = train_x.columns[train_x.isnull().any()]

    # Create indicator columns for missing values
    for col in cols_with_missing:
        train_x[col + "_was_missing"] = train_x[col].isnull()
        val_x[col + "_was_missing"] = val_x[col].isnull()

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer()
    imputed_train_x = pd.DataFrame(imputer.fit_transform(train_x))
    imputed_val_x = pd.DataFrame(imputer.transform(val_x))

    # Restore column names
    imputed_train_x.columns = train_x.columns
    imputed_val_x.columns = val_x.columns

    return imputed_train_x, imputed_val_x


def main():
    """
    Main function to run the data analysis.
    """
    # Read data from file
    datafile_path = file_path("datasets", "melb_data.csv")
    dataset = pd.read_csv(datafile_path)

    # Define features and target variable
    features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
    x = dataset[features]
    y = dataset.Price

    # Split data into training and validation sets
    train_x, val_x, train_y, val_y = train_test_split(
        x, y, train_size=0.8, test_size=0.2, random_state=0, shuffle=True
    )

    # Preprocess data
    train_x, val_x = preprocess_data(train_x, val_x)

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

    print("\nRandomForestRegressor")
    predictions = predict(
        RandomForestRegressor(random_state=1), train_x, val_x, train_y
    )
    mae = mean_absolute_error(val_y, predictions)
    print(f"MAE: {mae}")


if __name__ == "__main__":
    main()
