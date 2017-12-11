import gc

import pandas as pd
import sklearn
import xgboost as xgb


def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', sklearn.metrics.log_loss(labels, preds)


gc.enable()

train = pd.read_csv('../input/train_final.csv')
test = pd.read_csv('../input/test_final.csv')

# Delete date for now
train = train.drop(['transaction_date', 'membership_expire_date', 'expiration_date', 'registration_init_time'], axis=1)
test = test.drop(['transaction_date', 'membership_expire_date', 'expiration_date', 'registration_init_time'], axis=1)
# Delete date for now

feature_list = [
    #raw data
    'msno','payment_method_id','payment_plan_days','plan_list_price','actual_amount_paid','is_auto_renew',
    'is_cancel','city','bd','gender','registered_via','is_churn',
    #advanced features
    #user_log
    'log_day','total_25_sum','total_50_sum','total_75_sum','total_985_sum','total_100_sum','total_unq_sum','total_secs_sum',
    'total_sum','total_25ratio','total_100ratio','persong_play','persong_time','daily_play','daily_listentime',
    'one_week_sum','two_week_sum','one_week_secs_sum','two_week_secs_sum','week_secs_sum_ratio','week_sum_ratio',
    'one_semimonth_sum','two_semimonth_sum','one_semimonth_secs_sum','two_semimonth_secs_sum','semimonth_secs_sum_ratio','semimonth_sum_ratio',
    #transactions
    'discount','amt_per_day','is_discount','membership_days',
    'transaction_date_year','transaction_date_month','transaction_date_day',
    'membership_expire_date_year','membership_expire_date_month','membership_expire_date_day'
    #members
]

cols = [c for c in train.columns if c not in ['is_churn', 'msno']]

params = {
    'base_score': 0.5,
    'eta': 0.002,
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
                  early_stopping_rounds=50)

pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)

test['is_churn'] = pred.clip(0.0000001, 0.999999)
print(len(test))
test[['msno', 'is_churn']].to_csv('submission_xgboost_features_eta_0.002_round_2500_Dec_9.csv', index=False)
