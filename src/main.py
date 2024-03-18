import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from utils.file_util import file_path


def main():
    datafile_path = file_path("datasets", "melb_data.csv")
    dataset = pd.read_csv(datafile_path)

    x = pd.DataFrame(dataset).drop(["Price", "Address", "Date"], axis=1).copy()
    y = dataset.Price

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

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=0)),
        ]
    )
    scores = -cross_val_score(
        model_pipeline, x, y, cv=5, scoring="neg_mean_absolute_error"
    )
    print(f"MAE Scores: {scores}")
    print(f"Mean Scores: {scores.mean()}")


if __name__ == "__main__":
    main()
