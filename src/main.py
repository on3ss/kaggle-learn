import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from utils.file_util import file_path


def predict(
    model: sklearn.base.BaseEstimator,
    train_x: pd.DataFrame,
    val_x: pd.DataFrame,
    train_y: pd.Series,
) -> pd.Series:
    model.fit(train_x, train_y)
    return model.predict(val_x)


def main():
    datafile_path = file_path("datasets", "melb_data.csv")
    dataset = pd.read_csv(datafile_path)

    x = pd.DataFrame(dataset).drop(["Price", "Address", "Date"], axis=1).copy()
    y = dataset.Price

    train_x, val_x, train_y, val_y = train_test_split(
        x, y, train_size=0.8, test_size=0.2, random_state=0, shuffle=True
    )

    numerical_transformer = SimpleImputer(strategy="constant")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_columns = x.select_dtypes(include=["int", "float"]).columns.to_list()
    categorical_columns = x.select_dtypes(include=["object"]).columns.to_list()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_transformer,
                numerical_columns,
            ),
            (
                "cat",
                categorical_transformer,
                categorical_columns,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=0)),
        ]
    )
    predictions = predict(pipeline, train_x, val_x, train_y)
    mae = mean_absolute_error(val_y, predictions)
    print(f"MAE: {mae}")


if __name__ == "__main__":
    main()
