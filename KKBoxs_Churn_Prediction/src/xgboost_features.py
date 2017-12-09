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
model = xgb.train(params, xgb.DMatrix(x1, y1), 2000, watchlist, feval=xgb_score, maximize=False, verbose_eval=50,
                  early_stopping_rounds=50)

pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)

test['is_churn'] = pred.clip(0.0000001, 0.999999)
print(len(test))
test[['msno', 'is_churn']].to_csv('submission_xgboost_features_eta_0.002_round_2000_Dec_8.csv', index=False)
