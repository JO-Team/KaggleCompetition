import gc

import pandas as pd
import sklearn
import xgboost as xgb


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

'''
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bytree=1.0, gamma=1, learning_rate=0.002,
              max_delta_step=0, max_depth=6, min_child_weight=5, missing=None,
              n_estimators=600, n_jobs=1, nthread=1, objective='binary:logistic',
              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
              seed=None, silent=True, subsample=0.75)
'''
params = {
    'base_score': 0.5,
    'eta': 0.002,  # use 0.002
    'max_depth': 6,
    'booster': 'gbtree',
    'colsample_bylevel': 1,
    'colsample_bytree': 1.0,
    'gamma': 1,
    'max_child_weight': 5,
    'n_estimators': 600,
    'reg_alpha': '0',
    'reg_lambda': '1',
    'scale_pos_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 2017,
    'silent': True
}
x1, x2, y1, y2 = sklearn.model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.3,
                                                          random_state=2017)
watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
cv_output = xgb.cv(params, xgb.DMatrix(x1, y1), num_boost_round=2500, early_stopping_rounds=20, verbose_eval=50,
                   show_stdv=False)
model = xgb.train(params, xgb.DMatrix(x1, y1), 2500, watchlist, feval=xgb_score, maximize=False, verbose_eval=50,
                  early_stopping_rounds=50)  # use 1500

pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)

test['is_churn'] = pred.clip(0.0000001, 0.999999)
print(len(test))
test[['msno', 'is_churn']].to_csv('submission_xgboost_baseline_best_parameter_eta_0.002_round_2500.csv', index=False)
