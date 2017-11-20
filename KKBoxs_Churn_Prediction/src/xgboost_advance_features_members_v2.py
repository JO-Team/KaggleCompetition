import numpy as np
import pandas as pd
from subprocess import check_output
import sys
import gc
import collections
import xgboost as xgb
import sklearn


gc.enable()
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train_v2.csv')
test = pd.read_csv('../input/sample_submission_v2.csv')
members = pd.read_csv('../input/members_v2.csv')

transactions_features = ['msno', 'payment_method_id', 'payment_plan_days', 'plan_list_price', 'actual_amount_paid',
                         'is_auto_renew', 'transaction_date', 'membership_expire_date', 'is_cancel']
trans_features = [feature for feature in transactions_features[1:] if feature != '_']


def make_transactions_features():
    print('loading...')
    infos = {}
    with open('../input/transactions_v2.csv') as fd:
        count = 0
        fd.readline()
        for line in fd:
            pos = line.find(',')
            msid = line[:pos]
            splits = line[pos + 1:-1].split(',')
            info = [int(value) for value in splits[:]]
            info = [value for index, value in enumerate(info) if transactions_features[index + 1] != '_']
            if msid not in infos:
                infos[msid] = [[value] for value in info]
                infos[msid].insert(0, 1)
            else:
                infos[msid][0] += 1
                for index in range(1, 7):
                    infos[msid][index].append(info[index - 1])
            count += 1
            if count % 100000 == 0:
                print('processed: %d' % count)
    print('done: %d' % count)

    df_transactions = pd.DataFrame()
    df_transactions['msno'] = infos.keys()
    df_transactions['trans_count'] = [infos[key][0] for key in infos.keys()]
    for index, feature in enumerate(trans_features):
        df_transactions[feature] = [collections.Counter(infos[key][index + 1]).most_common()[0][0] for key in
                                    infos.keys()]

    return df_transactions


userlog_features = ['msno', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs']


def make_userlog_features():
    print('loading...')
    infos = {}
    with open('../input/user_logs_v2.csv') as fd:
        count = 0
        fd.readline()
        for line in fd:
            pos = line.find(',')
            msid = line[:pos]
            # _, num_25, num_50, num_75, num_985, num_100, num_unq, total_secs = [int(float(value)) for value in line[pos + 1:-1].split(',')]
            splits = line[pos + 1:-1].split(',')
            info = [int(value) for value in splits[:-1]]
            info.append(int(float(splits[-1])))
            # if len(info) != 8:
            #    print('not expect line: %s'%line[:-1])
            #    continue
            if msid not in infos:
                info[0] = 1
                infos[msid] = info
            else:
                infos[msid][0] += 1
                for index in range(1, 8):
                    infos[msid][index] += info[index]
            count += 1
            if count % 100000 == 0:
                print('processed: %d' % count)
    print('done: %d' % count)

    df_userlog = pd.DataFrame()
    df_userlog['msno'] = infos.keys()
    df_userlog['date_count'] = [infos[key][0] for key in infos.keys()]
    for index, feature in enumerate(userlog_features[1:]):
        if feature == 'total_secs':
            df_userlog[feature] = [infos[key][index] / 3600 for key in infos.keys()]
        else:
            df_userlog[feature] = [infos[key][index] for key in infos.keys()]

    return df_userlog


transactions = make_transactions_features()
user_logs = make_userlog_features()

# Add 4 Features
transactions['discount'] = transactions['plan_list_price']
transactions['is_discount'] = transactions.discount.apply(lambda x: 1 if x > 0 else 0)
transactions['amt_per_day'] = transactions['actual_amount_paid'] / transactions['payment_plan_days']

date_cols = ['transaction_date', 'membership_expire_date']
for col in date_cols:
    transactions[col] = pd.to_datetime(transactions[col], format='%Y%m%d')

transactions['membership_duration'] = transactions.membership_expire_date - transactions.transaction_date
transactions['membership_duration'] = transactions['membership_duration'] / np.timedelta64(1, 'D')
transactions['membership_duration'] = transactions['membership_duration'].astype(int)


# date_cols = ['registration_init_time', 'expiration_date']
date_cols = ['registration_init_time']

for col in date_cols:
    members[col] = pd.to_datetime(members[col], format='%Y%m%d')

train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')

train = pd.merge(train, user_logs, how='left', on='msno')
test = pd.merge(test, user_logs, how='left', on='msno')

train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno')

# Add 4 Features After combine
# train['reg_mem_duration'] = train['registration_duration'] - train['membership_duration']
# train['autorenew_&_not_cancel'] = ((train.is_auto_renew == 1) == (train.is_cancel == 0)).astype(np.int8)
# train['notAutorenew_&_cancel'] = ((train.is_auto_renew == 0) == (train.is_cancel == 1)).astype(np.int8)
# train['long_time_user'] = (((train['registration_duration'] / 365).astype(int)) > 1).astype(int)

# test['reg_mem_duration'] = test['registration_duration'] - test['membership_duration']
# test['autorenew_&_not_cancel'] = ((test.is_auto_renew == 1) == (test.is_cancel == 0)).astype(np.int8)
# test['notAutorenew_&_cancel'] = ((test.is_auto_renew == 0) == (test.is_cancel == 1)).astype(np.int8)
# test['long_time_user'] = (((test['registration_duration'] / 365).astype(int)) > 1).astype(int)

# Add Advanced Features
train['cost_per_second'] = train['actual_amount_paid'] / train['total_secs']
test['cost_per_second'] = test['actual_amount_paid'] / test['total_secs']


# Drop datetime features
train = train.drop(['transaction_date', 'membership_expire_date', 'registration_init_time'], 1)
test = test.drop(['transaction_date', 'membership_expire_date', 'registration_init_time'], 1)

# Deal with gender
gender = {'male': 1, 'female': 2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)

train = train.fillna(0)
test = test.fillna(0)

cols = [c for c in train.columns if c not in ['is_churn', 'msno']]


def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', sklearn.metrics.log_loss(labels, preds)


fold = 1
for i in range(fold):
    params = {
        'eta': 0.02,  # use 0.002
        'max_depth': 7,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = sklearn.model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.3,
                                                              random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 1500, watchlist, feval=xgb_score, maximize=False, verbose_eval=50,
                      early_stopping_rounds=50)  # use 1500
    if i != 0:
        pred += model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
    else:
        pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
pred /= fold

test['is_churn'] = pred.clip(0.0000001, 0.999999)
test[['msno', 'is_churn']].to_csv('submission_xgboost_advance_features_v1.csv', index=False)
