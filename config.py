import os

from pathlib import Path

DATA_FILE_TRAIN = Path.cwd().parent / 'data' / 'train.csv'
DATA_FILE_TEST = Path.cwd().parent / 'data' / 'test.csv'
MODEL_DATA = Path.cwd().parent / 'model_data'
TRAINED_MODELS_PATH = Path.cwd().parent / 'models'
DATA_SETS_PATH = MODEL_DATA / 'data_sets_s7.data'
SARIMAX_MODELS_PATH = MODEL_DATA / 'srimax_models_s7.data'
RESULTS = Path.cwd().parent / 'results_s3'
DATA_FILE_TRANSACTIONS = Path.cwd().parent / 'data' / 'transactions.csv'

if not Path(TRAINED_MODELS_PATH).is_dir():
    os.mkdir(TRAINED_MODELS_PATH)

if not Path(MODEL_DATA).is_dir():
    os.mkdir(MODEL_DATA)

if not Path(RESULTS).is_dir():
    os.mkdir(RESULTS)



