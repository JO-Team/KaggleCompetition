import gc
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import sklearn
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold


path = '../input/'


def time_features(df):
    # Make some new features with click_time column
    df['datetime'] = pd.to_datetime(df['click_time'])
    df['dow'] = df['datetime'].dt.dayofweek
    df["doy"] = df["datetime"].dt.dayofyear
    # df["dteom"]    = df["datetime"].dt.daysinmonth - df["datetime"].dt.day
    df.drop(['click_time', 'datetime'], axis=1, inplace=True)
    return df


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'auc', sklearn.metrics.auc(labels, preds)


train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}

# Read the last lines because they are more impacting in training than the starting lines
train = pd.read_csv(path + "train.csv", skiprows=range(1, 123903891), nrows=61000000, usecols=train_columns,
                    dtype=dtypes)
test = pd.read_csv(path + "test.csv", usecols=test_columns, dtype=dtypes)

# Drop the IP and the columns from target
y = train['is_attributed']
train.drop(['is_attributed'], axis=1, inplace=True)

# Drop IP and ID from test rows
sub = pd.DataFrame()
sub['click_id'] = test['click_id'].astype('int')
test.drop(['click_id'], axis=1, inplace=True)
gc.collect()

nrow_train = train.shape[0]
merge = pd.concat([train, test])

del train, test
gc.collect()

# Count the number of clicks by ip
ip_count = merge.groupby(['ip'])['channel'].count().reset_index()
ip_count.columns = ['ip', 'clicks_by_ip']
merge = pd.merge(merge, ip_count, on='ip', how='left', sort=False)
merge['clicks_by_ip'] = merge['clicks_by_ip'].astype('uint16')
merge.drop('ip', axis=1, inplace=True)

train = merge[:nrow_train]
test = merge[nrow_train:]

del test, merge
gc.collect()

train = time_features(train)
gc.collect()

# Set the params(this params from Pranav kernel) for xgboost model
params = {'eta': 0.6,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,
          'max_depth': 0,
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'min_child_weight': 0,
          'alpha': 4,
          'objective': 'binary:logistic',
          'scale_pos_weight': 9,
          'eval_metric': 'auc',
          'nthread': 8,
          'random_state': 99,
          'silent': True}

params = {
    'eta': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'max_leaves': [1200, 1300, 1400],
    'max_depth': [3, 4, 5, 6, 7],
    '': [],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'colsample_bytree': [0.6, 0.8, 1.0],

    'subsample': [0.7, 0.75, 0.8]
}


num_round = 200


# Get 10% of train dataset to use as validation
x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, random_state=99)
dtrain = xgb.DMatrix(x1, y1)
dvalid = xgb.DMatrix(x2, y2)
del x1, y1, x2, y2
gc.collect()
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
num_round = 200
model = xgb.train(params, dtrain, num_round, watchlist, maximize=True, early_stopping_rounds=25, verbose_eval=5)
del dvalid


del dtrain
gc.collect()


# Load the test for predict
test = pd.read_csv(path + "test.csv", usecols=test_columns, dtype=dtypes)
test = pd.merge(test, ip_count, on='ip', how='left', sort=False)
del ip_count
gc.collect()

test['clicks_by_ip'] = test['clicks_by_ip'].astype('uint16')
test = time_features(test)
test.drop(['click_id', 'ip'], axis=1, inplace=True)
dtest = xgb.DMatrix(test)
del test
gc.collect()

submission_file_name = 'submission_xgboost_best_param_' + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")) + '.csv'

# Save the predictions
sub['is_attributed'] = model.predict(dtest, ntree_limit=model.best_ntree_limit)
sub.to_csv(submission_file_name, index=False)
