import pandas as pd
import basic_prediction
from utils import file_util


def main():
    melbourne_data_file_path = file_util.file_path("datasets", "melb_data.csv")
    melbourne_data = pd.read_csv(melbourne_data_file_path)

    basic_prediction.run_predictions(melbourne_data)


if __name__ == "__main__":
    main()
