import time
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                              '%m-%d-%Y %H:%M:%S')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)


def timeit(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        try:
            ret = f(*args, **kwargs)
        except Exception as ex:
            print(ex)

        time2 = time.time()
        print('{:s} function took {:.3f} seconds'.format(f.__name__, (time2 - time1)))
        return ret

    return wrap


def find_sep_by_date(path):
    d = '2017-07-16'
    c = 0
    import csv
    with open(path) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for l in csv_reader:
            if l[1] == d:
                print(c)
                break
            c += 1
