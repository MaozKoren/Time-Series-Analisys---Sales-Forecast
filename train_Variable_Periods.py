import multiprocessing
import pickle
import sys
import pandas as pd
import statsmodels.api as sm
import logging

from pathlib import Path
from utils import timeit
from config import TRAINED_MODELS_PATH, DATA_FILE_TRAIN, DATA_SETS_PATH, SARIMAX_MODELS_PATH

LINE_SEP_LOWER_THEN_2017_07_16 = 2945646

logger = logging.getLogger(__name__)

def find_best_period(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    freq_spec = pd.concat([pd.Series(freqencies), pd.Series(spectrum)], axis=1).sort_values(by=1, ascending=False, ignore_index=True)
    highestfreq = freq_spec[0][0]
    if highestfreq == 0:
        return 3
    else:
        BestPeriod = 365 / highestfreq
        #print(BestPeriod)
        return round(BestPeriod)

def fit(model):
    counter.value += 1
    current_counter = counter.get()

    try:
        data = model['srima_model'].fit(disp=False)
        model_path = TRAINED_MODELS_PATH / f'model_{current_counter}'
        with open(model_path, 'wb') as f:
            pickle.dump({'train_data': data, 'model_detail_list': model['model_detail_list']}, f)
    except Exception as ex:
        logger.error(model['model_detail_list'], exc_info=ex)

    logger.info(f'finish {current_counter} / {len(models_to_train)}')


@timeit
def train_model():
    with multiprocessing.Pool(processes=CPU_COUNT) as mp:
        mp.map(fit, [model for model in models_to_train])


if __name__ == '__main__':

    with_mp = sys.argv[1]
    start_from = int(sys.argv[2])
    how_many = int(sys.argv[3])
    with_cached_model_metadata = sys.argv[4]
    CPU_COUNT = multiprocessing.cpu_count()
    model_trained = 0
    datasets = []
    models_SARIMAX = []

    logger.info(f'with multiprocessing: {with_mp}')
    logger.info(f'cores: {CPU_COUNT}')
    logger.info(f'starting from sample: {start_from}')
    logger.info(f'how many samples: {how_many - start_from}')
    logger.info(f'with cached model metadata: {with_cached_model_metadata}')

    # Copy of dataset
    Train = pd.read_csv(DATA_FILE_TRAIN, parse_dates=["date"])
    Train = Train.set_index("date").to_period("D")
    stores = Train.store_nbr.unique()
    families = Train.family.unique()

    if with_cached_model_metadata == 'True':
        logger.info('using cached data sets')
        with open(DATA_SETS_PATH, 'rb') as f:
            datasets = pickle.load(f)

        logger.info('using cached srimax')
        with open(SARIMAX_MODELS_PATH, 'rb') as f:
            models_SARIMAX = pickle.load(f)

    else:
        logger.info('building model from data')

        for store in stores:
            is_store_nbr = Train['store_nbr'] == store
            Filtered_df = Train[is_store_nbr].copy()
            for family in families:
                single_test = Filtered_df.loc[Filtered_df['family'] == family]
                X = single_test.copy()
                X["dayofyear"] = X.index.dayofyear
                X["year"] = X.index.year
                BestPeriod = find_best_period(X.sales)
                #print(f'training with Period: {BestPeriod}')
                models_SARIMAX.append({'srima_model': sm.tsa.statespace.SARIMAX(X.sales, trend="c", order=(1, 1, 1),
                                                                                seasonal_order=(1, 0, 1, BestPeriod)),
                                       'model_detail_list': (store, family)})
                model_trained += 1
                datasets.append(X.sales)

        logger.info('finished to build model for data sets')
        with open(DATA_SETS_PATH, 'wb') as f:
            pickle.dump(datasets, f)

        logger.info('finished to build model for srimax')
        with open(SARIMAX_MODELS_PATH, 'wb') as f:
            pickle.dump(models_SARIMAX, f)

        logger.info('finished to build model from data')

    models_to_train = models_SARIMAX[start_from:how_many]

    logger.info(f'starting to train {len(models_to_train)} models')

    if with_mp == 'True':
        manager = multiprocessing.Manager()
        counter = manager.Value('i', start_from)
        train_model()
    else:
        start_from_counter = start_from
        counter = 0
        for model in models_to_train:
            model_path = TRAINED_MODELS_PATH / f'model_{start_from_counter}'

            if not Path(model_path).is_file():
                data = model['srima_model'].fit(disp=False)
                with open(model_path, 'wb') as f:
                    pickle.dump({'train_data': data, 'model_detail_list': model['model_detail_list']}, f)
                start_from_counter += 1
                counter += 1
                logger.info(f'finish {counter} / {len(models_to_train)}')

    logger.info(f'finished training')
