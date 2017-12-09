import gc

import pandas as pd
import sklearn
import xgboost as xgb
import numpy as np


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

###############Feature engineering####################
#划分近一个月的数据为数据集28天
train = train[(train['date']<20170301)&(train['date']>20170131)]
test = test[(test['date']<20170329)&(test['date']>20170228)]

#用户一个月的活跃角度
#一个月的登陆天数
train_log_day = train.groupby(['msno']).date.agg({'log_day':'count'})
test_log_day = test.groupby(['msno']).date.agg({'log_day':'count'})
train = pd.merge(train,train_log_day,on=['msno'],how='left')
test = pd.merge(test,test_log_day,on=['msno'],how='left')

#一个月的听歌汇总
train_total_25_sum = train.groupby(['msno']).num_25.agg({'total_25_sum':np.sum})
train_total_50_sum = train.groupby(['msno']).num_50.agg({'total_50_sum':np.sum})
train_total_75_sum = train.groupby(['msno']).num_75.agg({'total_75_sum':np.sum})
train_total_985_sum = train.groupby(['msno']).num_985.agg({'total_985_sum':np.sum})
train_total_100_sum = train.groupby(['msno']).num_100.agg({'total_100_sum':np.sum})
train_total_unq_sum = train.groupby(['msno']).num_unq.agg({'total_unq_sum':np.sum})
train_total_secs_sum = train.groupby(['msno']).total_secs.agg({'total_secs_sum':np.sum})
train = pd.merge(train,train_total_25_sum,on=['msno'],how='left')
train = pd.merge(train,train_total_50_sum,on=['msno'],how='left')
train = pd.merge(train,train_total_75_sum,on=['msno'],how='left')
train = pd.merge(train,train_total_985_sum,on=['msno'],how='left')
train = pd.merge(train,train_total_100_sum,on=['msno'],how='left')
train['total_sum'] = train['total_25_sum']+train['total_50_sum']+train['total_75_sum']+train['total_985_sum']+train['total_100_sum']
#一个月的听歌习惯
train['total_25ratio'] = train['total_25_sum'] / train['total_sum']
train['total_100ratio'] = train['total_100_sum'] / train['total_sum']
#听歌是循环播放还是试听,每首歌播放次数
train['persong_play'] = train['total_sum'] / train['total_unq_sum']
#听歌每首歌平均播放时间
train['persong_time'] = train['total_secs_sum'] / train['total_sum']
#平均每天听歌数量
train['daily_play'] = train['total_sum'] / train['log_day']
#平均每天听歌时间
train['daily_listentime'] = train['total_secs_sum'] / train['log_day']

test_total_25_sum = test.groupby(['msno']).num_25.agg({'total_25_sum':np.sum})
test_total_50_sum = test.groupby(['msno']).num_50.agg({'total_50_sum':np.sum})
test_total_75_sum = test.groupby(['msno']).num_75.agg({'total_75_sum':np.sum})
test_total_985_sum = test.groupby(['msno']).num_985.agg({'total_985_sum':np.sum})
test_total_100_sum = test.groupby(['msno']).num_100.agg({'total_100_sum':np.sum})
test_total_unq_sum = test.groupby(['msno']).num_unq.agg({'total_unq_sum':np.sum})
test_total_secs_sum = test.groupby(['msno']).total_secs.agg({'total_secs_sum':np.sum})
test = pd.merge(test,test_total_25_sum,on=['msno'],how='left')
test = pd.merge(test,test_total_50_sum,on=['msno'],how='left')
test = pd.merge(test,test_total_75_sum,on=['msno'],how='left')
test = pd.merge(test,test_total_985_sum,on=['msno'],how='left')
test = pd.merge(test,test_total_100_sum,on=['msno'],how='left')
test['total_sum'] = test['total_25_sum']+test['total_50_sum']+test['total_75_sum']+test['total_985_sum']+test['total_100_sum']
#一个月的听歌习惯
test['total_25ratio'] = test['total_25_sum'] / test['total_sum']
test['total_100ratio'] = test['total_100_sum'] / test['total_sum']
#听歌是循环播放还是试听,每首歌播放次数
test['persong_play'] = test['total_sum'] / test['total_unq_sum']
#听歌每首歌平均播放时间
test['persong_time'] = test['total_secs_sum'] / test['total_sum']
#平均每天听歌数量
test['daily_play'] = test['total_sum'] / test['log_day']
#平均每天听歌时间
test['daily_listentime'] = test['total_secs_sum'] / test['log_day']




#用户每天的活跃角度
#1.每天听歌总数
train['daily_total_num'] = train['num_25']+train['num_50']+train['num_75']+train['num_985']+train['num_100']
test['daily_total_num'] = test['num_25']+test['num_50']+test['num_75']+test['num_985']+test['num_100']

#2.每天听歌是循环播放还是试听,每首歌歌播放次数
train['daily_permusic_time'] = train['daily_total_num'] / train['num_unq']
test['daily_permusic_time'] = test['daily_total_num'] / test['num_unq']

#3.用户每天的听音乐习惯，是海量试听，还是细细聆听，每种长度占比
train['daily_25ratio'] = train['num_25'] / train['daily_total_num']
train['daily_50ratio'] = train['num_50'] / train['daily_total_num']
train['daily_75ratio'] = train['num_75'] / train['daily_total_num']
train['daily_985ratio'] = train['num_985'] / train['daily_total_num']
train['daily_100ratio'] = train['num_100'] / train['daily_total_num']

test['daily_25ratio'] = test['num_25'] / test['daily_total_num']
test['daily_50ratio'] = test['num_50'] / test['daily_total_num']
test['daily_75ratio'] = test['num_75'] / test['daily_total_num']
test['daily_985ratio'] = test['num_985'] / test['daily_total_num']
test['daily_100ratio'] = test['num_100'] / test['daily_total_num']

#4.每天平均每首歌的时间
train['daily_listentime'] = train['daily_total_num'] / train['total_secs']
test['daily_listentime'] = test['daily_total_num'] / test['total_secs']

#用户一段时间的活跃角度
#1. 总的听歌天数
total_day = train.groupby(['msno']).date.agg({'total_day':'count'})

#1.总的听歌数
total_num = train.groupby(['msno']).daily_total_num.agg({'total_num':np.sum})




#用户的消费角度



# Delete date for now
train = train.drop(['transaction_date', 'membership_expire_date', 'expiration_date', 'registration_init_time'], axis=1)
test = test.drop(['transaction_date', 'membership_expire_date', 'expiration_date', 'registration_init_time'], axis=1)
# Delete date for now

cols = [c for c in train.columns if c not in ['is_churn', 'msno']]

fold = 1
for i in range(fold):
    params = {
        'eta': 0.002,  # use 0.002
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
print(len(test))
test[['msno', 'is_churn']].to_csv('submission_xgboost_baseline_eta_0.002_round_1500.csv', index=False)

