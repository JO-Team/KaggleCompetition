import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb

#load files
train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])
id_test = test.id

# fill data
clean_result = pd.read_csv("../input/myfill.csv")

for j in range(len(train)):
    if np.isnan(train.ix[j, 'life_sq']):
        temp_1 = clean_result[clean_result['sub_area'] == train.ix[j, 'sub_area']]
        train.ix[j, 'life_sq'] = round(temp_1['mean_life'].values)

for k in range(len(train)):
    if np.isnan(train.ix[k, 'max_floor']):
        temp_2 = clean_result[clean_result['sub_area'] == train.ix[k, 'sub_area']]
        train.ix[k, 'max_floor'] = round(temp_2['mode_max'].values)

for k in range(len(train)):
    if np.isnan(train.ix[k, 'material']):
        temp_2 = clean_result[clean_result['sub_area'] == train.ix[k, 'sub_area']]
        train.ix[k, 'material'] = round(temp_2['mode_material'].values)

for k in range(len(train)):
    if np.isnan(train.ix[k, 'build_year']):
        temp_2 = clean_result[clean_result['sub_area'] == train.ix[k, 'sub_area']]
        train.ix[k, 'build_year'] = round(temp_2['mode_build'].values)

for k in range(len(train)):
    if np.isnan(train.ix[k, 'num_room']):
        temp_2 = clean_result[clean_result['sub_area'] == train.ix[k, 'sub_area']]
        train.ix[k, 'num_room'] = round(temp_2['mode_num'].values)

for k in range(len(train)):
    if np.isnan(train.ix[k, 'kitch_sq']):
        temp_2 = clean_result[clean_result['sub_area'] == train.ix[k, 'sub_area']]
        train.ix[k, 'kitch_sq'] = round(temp_2['mean_kitch'].values)

for k in range(len(train)):
    if np.isnan(train.ix[k, 'state']):
        temp_2 = clean_result[clean_result['sub_area'] == train.ix[k, 'sub_area']]
        train.ix[k, 'state'] = round(temp_2['mode_state'].values)



train.to_csv('../input/train_fillmodemean.csv', index=False)

for j in range(len(test)):
    if np.isnan(test.ix[j, 'life_sq']):
        temp_1 = clean_result[clean_result['sub_area'] == test.ix[j, 'sub_area']]
        test.ix[j, 'life_sq'] = round(temp_1['mean_life'].values)

for k in range(len(test)):
    if np.isnan(test.ix[k, 'max_floor']):
        temp_2 = clean_result[clean_result['sub_area'] == test.ix[k, 'sub_area']]
        test.ix[k, 'max_floor'] = round(temp_2['mode_max'].values)

for k in range(len(test)):
    if np.isnan(test.ix[k, 'material']):
        temp_2 = clean_result[clean_result['sub_area'] == test.ix[k, 'sub_area']]
        test.ix[k, 'material'] = round(temp_2['mode_material'].values)

for k in range(len(test)):
    if np.isnan(test.ix[k, 'build_year']):
        temp_2 = clean_result[clean_result['sub_area'] == test.ix[k, 'sub_area']]
        test.ix[k, 'build_year'] = round(temp_2['mode_build'].values)

for k in range(len(test)):
    if np.isnan(test.ix[k, 'num_room']):
        temp_2 = clean_result[clean_result['sub_area'] == test.ix[k, 'sub_area']]
        test.ix[k, 'num_room'] = round(temp_2['mode_num'].values)

for k in range(len(test)):
    if np.isnan(test.ix[k, 'kitch_sq']):
        temp_2 = clean_result[clean_result['sub_area'] == test.ix[k, 'sub_area']]
        test.ix[k, 'kitch_sq'] = round(temp_2['mean_kitch'].values)

for k in range(len(test)):
    if np.isnan(test.ix[k, 'state']):
        temp_2 = clean_result[clean_result['sub_area'] == test.ix[k, 'sub_area']]
        test.ix[k, 'state'] = round(temp_2['mode_state'].values)


test.to_csv('../input/test_fillmodemean.csv', index=False)

