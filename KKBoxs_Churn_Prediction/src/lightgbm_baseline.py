import gc

import lightgbm as lgb
import pandas as pd
import sklearn
from sklearn.model_selection import ShuffleSplit


def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', sklearn.metrics.log_loss(labels, preds)


gc.enable()

transactions = pd.read_csv('../input/processed_transaction_all.csv')

members_v1 = pd.read_csv('../input/members.csv')
members_v2 = pd.read_csv('../input/members_v2.csv')
members = members_v1.append(members_v2, ignore_index=True)

user_log = pd.read_csv('../input/processed_user_log_all.csv')

train_v1 = pd.read_csv('../input/train.csv')
train_v2 = pd.read_csv('../input/train_v2.csv')
train = train_v1.append(train_v2, ignore_index=True)

test = pd.read_csv('../input/sample_submission_v2.csv')

# Merge Data

train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')

train = pd.merge(train, user_log, how='left', on='msno')
test = pd.merge(test, user_log, how='left', on='msno')

train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno')

# Drop duplicates first
test = test.drop_duplicates('msno')

print(test.head(10))

gender = {'male': 1, 'female': 2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)

train = train.fillna(0)
test = test.fillna(0)

# Delete date for now
train = train.drop(['transaction_date', 'membership_expire_date', 'expiration_date', 'registration_init_time'], axis=1)
test = test.drop(['transaction_date', 'membership_expire_date', 'expiration_date', 'registration_init_time'], axis=1)
# Delete date for now

cols = [c for c in train.columns if c not in ['is_churn', 'msno']]

bst = None

for train_indices, val_indices in ShuffleSplit(n_splits=1, test_size=0.1, train_size=0.4).split(train):
    train_data = lgb.Dataset(train[cols].loc[train_indices, :],
                             label=train.loc[train_indices, 'is_churn'])
    val_data = lgb.Dataset(train[cols].loc[val_indices, :],
                           label=train.loc[val_indices, 'is_churn'])

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.002,  # small learn rate, large number of iterations
        'verbose': 0,
        'num_leaves': 108,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 128,
        'max_depth': 10,
    }

    bst = lgb.train(params, train_data, 100, valid_sets=[val_data])

predictions = bst.predict(test[cols])
test['is_churn'] = predictions
test.drop(cols, axis=1, inplace=True)
test.to_csv('submission_lightgbm_combine_data_baseline.csv', index=False)
