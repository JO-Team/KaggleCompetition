import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import xgboost as xgb

transactions = pd.read_csv('../input/transactions_v2.csv')  # reading the transaction file
members = pd.read_csv('../input/members_v2.csv')
user_log = pd.read_csv('../input/user_logs_v2.csv')
train = pd.read_csv('../input/train_v2.csv')
test = pd.read_csv('../input/sample_submission_v2.csv')

# h=change the type of these series
transactions['payment_method_id'] = transactions['payment_method_id'].astype('int8')
transactions['payment_plan_days'] = transactions['payment_plan_days'].astype('int16')
transactions['plan_list_price'] = transactions['plan_list_price'].astype('int16')
transactions['actual_amount_paid'] = transactions['actual_amount_paid'].astype('int16')
transactions['is_auto_renew'] = transactions['is_auto_renew'].astype('int8')  # changing the type to boolean
transactions['is_cancel'] = transactions['is_cancel'].astype('int8')  # changing the type to boolean
transactions['membership_expire_date'] = pd.to_datetime(transactions['membership_expire_date'].astype(str),
                                                        infer_datetime_format=True, exact=False)
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'].astype(str),
                                                  infer_datetime_format=True, exact=False)
# converting the series to string and then to datetime format for easy manipulation of dates

members['city'] = members['city'].astype('int8')
members['bd'] = members['bd'].astype('int16')
members['bd'] = members['bd'].astype('int8')
members['registration_init_time'] = pd.to_datetime(members['registration_init_time'].astype(str),
                                                   infer_datetime_format=True, exact=False)
train['is_churn'] = train['is_churn'].astype('int8')

user_log['date'] = pd.to_datetime(user_log['date'].astype(str),
                                  infer_datetime_format=True, exact=False)

members_trans = members.merge(transactions, how='inner', on='msno')
data = members_trans.merge(train, how='inner', on='msno')


def assign_gender(item):
    if (item == 'male') | (item == 'female'):
        return 1
    else:
        return 0


data['gender'] = data['gender'].apply(assign_gender)

# Feature discount
data['discount'] = data['plan_list_price'] - data['actual_amount_paid']

newdf = data.join(
    pd.get_dummies(data['payment_method_id']))  # creating a new columns for paymenth method id dummyvariable

payment_method_id = {}
for i in data['payment_method_id'].unique():
    payment_method_id.update(
        {i: 'payment_method_id{}'.format(i)})  # create a dictionary to automatic renaming of columns

newdf = newdf.rename(columns=payment_method_id)  # renaming the new columns
del newdf['payment_method_id']  # deleting the extra columns

# print('Data Column after payment_method_id')
# print(newdf.columns)

newdf = newdf.join(pd.get_dummies(newdf['gender']))  # creating a new columns for paymenth method id dummyvariable

gender = {}
gender.update({True: 'gender_provided'})  # create a dictionary to automatic renaming of columns
gender.update({False: 'gender_not_provided'})
newdf = newdf.rename(columns=gender)  # renaming the new columns
del newdf['gender']  # deleting the extra columns

newdf = newdf.join(
    pd.get_dummies(newdf['registered_via']))  # creating a new columns for paymenth method id dummyvariable

registered_via = {}
for i in data['registered_via'].unique():
    registered_via.update({i: 'registered_via{}'.format(i)})  # create a dictionary to automatic renaming of columns

newdf = newdf.rename(columns=registered_via)  # renaming the new columns
del newdf['registered_via']  # deleting the extra columns

newdf = newdf.join(pd.get_dummies(newdf['city']))  # creating a new columns for paymenth method id dummyvariable

city = {}
for i in data['city'].unique():
    city.update({i: 'city{}'.format(i)})  # create a dictionary to automatic renaming of columns

newdf = newdf.rename(columns=city)  # renaming the new columns
del newdf['city']  # deleting the extra columns


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


bd_mean = np.mean(newdf['bd'])
newdf[(newdf['bd'] < 0) | (newdf['bd'] > 100)].loc[:, 'bd'] = bd_mean  # filling the odd aged people with value

newdf['count_of_recharge'] = 1

print('The data column is:')
print(newdf.columns)

newdf_grouped = newdf.groupby('msno').agg({'bd': np.mean, 'registration_init_time': min,
                                           'payment_plan_days': np.mean, 'plan_list_price': np.mean,
                                           'count_of_recharge': 'sum', 'actual_amount_paid': np.mean,
                                           'is_auto_renew': np.mean, 'transaction_date': min,
                                           'membership_expire_date': max,
                                           'is_cancel': np.mean, 'is_churn': min, 'discount': 'sum',
                                           # 'payment_method_id2': np.mean,
                                           # 2 #'payment_method_id3': sum,
                                           # 'payment_method_id4': np.sum,
                                           # 'payment_method_id5': np.sum,
                                           # 2 #'payment_method_id6': np.sum,
                                           'payment_method_id8': np.sum,
                                           'payment_method_id10': np.sum,
                                           'payment_method_id11': np.sum, 'payment_method_id12': np.sum,
                                           # 2 #'payment_method_id13': np.sum,
                                           'payment_method_id14': np.sum, 'payment_method_id15': np.sum,
                                           'payment_method_id16': np.sum,
                                           'payment_method_id17': np.sum, 'payment_method_id18': np.sum,
                                           'payment_method_id19': np.sum,
                                           'payment_method_id20': np.sum, 'payment_method_id21': np.sum,
                                           'payment_method_id22': np.sum,
                                           'payment_method_id23': np.sum,
                                           # 'payment_method_id24': np.sum,
                                           # 'payment_method_id25': np.sum,
                                           'payment_method_id26': np.sum, 'payment_method_id27': np.sum,
                                           'payment_method_id28': np.sum,
                                           'payment_method_id29': np.sum, 'payment_method_id30': np.sum,
                                           'payment_method_id31': np.sum,
                                           'payment_method_id32': np.sum, 'payment_method_id33': np.sum,
                                           'payment_method_id34': np.sum,
                                           'payment_method_id35': np.sum, 'payment_method_id36': np.sum,
                                           'payment_method_id37': np.sum,
                                           'payment_method_id38': np.sum, 'payment_method_id39': np.sum,
                                           'payment_method_id40': np.sum,
                                           'payment_method_id41': np.sum, 'gender_not_provided': np.mean,
                                           'gender_provided': np.mean,
                                           'registered_via3': np.mean, 'registered_via4': np.mean,
                                           'registered_via7': np.mean,
                                           'registered_via9': np.mean, 'registered_via13': np.mean, 'city1': np.mean,
                                           'city3': np.mean,
                                           'city4': np.mean, 'city5': np.mean, 'city6': np.mean, 'city7': np.mean,
                                           'city8': np.mean,
                                           'city9': np.mean, 'city10': np.mean, 'city11': np.mean, 'city12': np.mean,
                                           'city13': np.mean,
                                           'city14': np.mean, 'city15': np.mean, 'city16': np.mean, 'city17': np.mean,
                                           'city18': np.mean,
                                           'city19': np.mean, 'city20': np.mean, 'city21': np.mean, 'city22': np.mean})

newdf_grouped[newdf_grouped.columns[-28:]] = newdf_grouped[newdf_grouped.columns[-28:]].applymap(
    lambda x: 1 if x > 0 else 0).apply(lambda x: x.astype('int8'))  # converting 0/1 for city

newdf_grouped[newdf_grouped.columns[12:-28]] = newdf_grouped[newdf_grouped.columns[12:-28]].apply(
    lambda x: x.astype('int8'))

newdf_grouped['discount'] = newdf_grouped['discount'].astype('int16')

newdf_grouped[newdf_grouped.columns[2:5]] = newdf_grouped[newdf_grouped.columns[2:5]].apply(
    lambda x: round(x).astype('int16'))

newdf_grouped['days_to_buy_membership'] = newdf_grouped['transaction_date'] - newdf_grouped['registration_init_time']

newdf_grouped['days_to_buy_membership'] = (newdf_grouped['days_to_buy_membership'] / np.timedelta64(1, 'D')).astype(int)

print('The data column is:')
print(newdf_grouped.dtypes)

# Use XGBoost

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
test[['msno', 'is_churn']].to_csv('submission_xgboost_combine_data_baseline_members_v2.csv', index=False)
