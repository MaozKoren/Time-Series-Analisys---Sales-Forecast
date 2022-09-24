import os
import pandas as pd
import csv
from pathlib import Path

from config import RESULTS

id_list = []
value_list = []
for result in os.listdir(RESULTS):
    with open(Path(RESULTS) / result) as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            id_list.append(line[0])
            value_list.append(line[1].rstrip())

results_df = pd.DataFrame({'id': id_list, 'value_list': value_list}).sort_values('id')
file_path = 'results_new.csv'
print(f'writing results to: {file_path}')
results_df.to_csv(file_path, sep=',', index=False, header=False)
