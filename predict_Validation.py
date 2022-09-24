import statistics
import sys
import os
import pickle

import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_squared_log_error

from config import TRAINED_MODELS_PATH, DATA_FILE_TEST, DATA_FILE_TRAIN, RESULTS


LINE_SEP_LOWER_THEN_2017_07_16 = 2945646
daysForcast = 35
daysPredict = 100
forecasts = []
predictions = []
errors = []
Train = pd.read_csv(DATA_FILE_TRAIN, parse_dates=["date"])
Train = Train.set_index("date").to_period("D")
test_df = Train[0:LINE_SEP_LOWER_THEN_2017_07_16 - 1]
validation_df = Train[LINE_SEP_LOWER_THEN_2017_07_16::] #Use this for validation

def forecast_model(res, days):
    pred_uc = res.get_forecast(steps=1 * days)
    return pred_uc.predicted_mean[4:]


if __name__ == '__main__':

    how_many = int(sys.argv[1])

    print(f'how many samples: {how_many}')

    models_counter = 0

    for trained_model in os.listdir(TRAINED_MODELS_PATH):
        model_path = TRAINED_MODELS_PATH / trained_model

        print(f'starting to predict model: {trained_model}')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            trained_model_data = model['train_data']
            model_detail_list = model['model_detail_list']
        models_counter += 1
        if models_counter > how_many:
            break

        forecast = pd.DataFrame(forecast_model(trained_model_data, daysForcast))
        is_store_nbr = validation_df['store_nbr'] == model_detail_list[0]
        Filtered_by_store_df = validation_df[is_store_nbr].copy()
        Filtered_by_family_df = Filtered_by_store_df.loc[Filtered_by_store_df['family'] == model_detail_list[1]]
        error = mean_squared_log_error(Filtered_by_family_df['sales'], forecast.abs(), squared=True)
        errors.append(error)

print(f'The mean error rate is: {statistics.mean(errors)}')
