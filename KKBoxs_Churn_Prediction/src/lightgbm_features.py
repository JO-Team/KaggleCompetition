import gc

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt


gc.enable()

transactions = pd.read_csv('../input/processed_transaction_all.csv')

members_v1 = pd.read_csv('../input/members.csv')
members_v2 = pd.read_csv('../input/members_v2.csv')
members = members_v1.append(members_v2, ignore_index=True)

user_log_train = pd.read_csv('../input/processed_features_user_log_feb.csv')
user_log_test = pd.read_csv('../input/processed_features_user_log_mar.csv')
user_log_all = pd.read_csv('../input/processed_user_log_all.csv')

train_v1 = pd.read_csv('../input/train.csv')
train_v2 = pd.read_csv('../input/train_v2.csv')
train = train_v1.append(train_v2, ignore_index=True)

test = pd.read_csv('../input/sample_submission_v2.csv')

# Merge Data

train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')

train = pd.merge(train, user_log_train, how='left', on='msno')
test = pd.merge(test, user_log_test, how='left', on='msno')

train = pd.merge(train, user_log_all, how='left', on='msno')
test = pd.merge(test, user_log_all, how='left', on='msno')

train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno')

del transactions, members_v1, members_v2, members, user_log_train, user_log_test, user_log_all, train_v1, train_v2
gc.collect()

# Drop duplicates first
test = test.drop_duplicates('msno')

gender = {'male': 1, 'female': 2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)

train = train.fillna(0)
test = test.fillna(0)

# Delete date for now
train = train.drop(['transaction_date', 'membership_expire_date', 'expiration_date', 'registration_init_time'], axis=1)
test = test.drop(['transaction_date', 'membership_expire_date', 'expiration_date', 'registration_init_time'], axis=1)
# Delete date for now

feature_list = [
    # raw data
    'msno', 'payment_method_id', 'payment_plan_days', 'plan_list_price', 'actual_amount_paid', 'is_auto_renew',
    'is_cancel', 'city', 'bd', 'gender', 'registered_via', 'is_churn',
    # advanced features
    # user_log
    'log_day', 'total_25_sum', 'total_50_sum', 'total_75_sum', 'total_985_sum', 'total_100_sum', 'total_unq_sum',
    'total_secs_sum',
    'total_sum', 'total_25ratio', 'total_100ratio', 'persong_play', 'persong_time', 'daily_play', 'daily_listentime',
    'one_week_sum', 'two_week_sum', 'one_week_secs_sum', 'two_week_secs_sum', 'week_secs_sum_ratio', 'week_sum_ratio',
    'one_semimonth_sum', 'two_semimonth_sum', 'one_semimonth_secs_sum', 'two_semimonth_secs_sum',
    'semimonth_secs_sum_ratio', 'semimonth_sum_ratio',
    # transactions
    'discount', 'amt_per_day', 'is_discount', 'membership_days',
    'transaction_date_year', 'transaction_date_month', 'transaction_date_day',
    'membership_expire_date_year', 'membership_expire_date_month', 'membership_expire_date_day'
    # members
]

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
        'max_depth': 7,
    }

    bst = lgb.train(params, train_data, 2500, valid_sets=[val_data], early_stopping_rounds=50)

predictions = bst.predict(test[cols])
test['is_churn'] = predictions
test.drop(cols, axis=1, inplace=True)
test.to_csv('submission_lightgbm_features_all_eta_0.002_round_2500_Dec_13.csv', index=False)

print('Plot feature importances...')
ax = lgb.plot_importance(bst)
# plt.show()
plt.savefig('lightgbm_feaeture_importance_')
