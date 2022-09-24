import csv

import pandas as pd

from config import DATA_FILE_TRAIN
from config import DATA_FILE_TEST
Test = pd.read_csv(DATA_FILE_TRAIN, parse_dates=["date"])
Test = Test.set_index("date").to_period("D")

i = 3000888
c = 0
all_results = []

with open('results.csv') as f:
    csv_reader = csv.reader(f)
    for line in csv_reader:
        all_results.append(int(line[0]))

#print(len(arr))
data = {}

with open(DATA_FILE_TEST) as f:
    csv_reader = csv.reader(f)
    next(csv_reader)
    for line in csv_reader:
        data[int(line[0])] = {'date': line[1], 'num': line[2], 'family': line[3]}

arr_id = []
arr_date = []
arr_store_nbr = []
arr_family = []

for key in data:
    c += 1
    if key not in all_results:
        i += 1
        arr_id.append(key)
print(c)
average_value_list = []
for item in arr_id:
    details = data[item]
    is_store_nbr = Test['store_nbr'] == int(details['num'])
    Filtered_df = Test[is_store_nbr].copy()
    single_test = Filtered_df.loc[Filtered_df['family'] == details['family']]
    avg = single_test['sales'].mean()
    average_value_list.append(avg)

missing_results_df = pd.DataFrame({'id': arr_id, 'value_list': average_value_list}).sort_values('id')
file_path = 'missing_results.csv'
print(f'writing results to: {file_path}')
missing_results_df.to_csv(file_path, sep=',', index=False, header=False)