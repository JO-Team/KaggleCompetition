import gc

import numpy as np
import pandas as pd

gc.enable()

# Read Data
user_log_v1 = pd.read_csv('../input/user_logs.csv')
user_log_v2 = pd.read_csv('../input/user_logs_v2.csv')

user_log = user_log_v1.append(user_log_v2)

transactions = pd.read_csv('../input/processed_transaction_all.csv')

members_v1 = pd.read_csv('../input/members.csv')
members_v2 = pd.read_csv('../input/members_v2.csv')
members = members_v1.append(members_v2, ignore_index=True)

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
# 划分近一个月的数据为数据集28天
train = train[(train['date'] < 20170301) & (train['date'] > 20170131)]
test = test[(test['date'] < 20170329) & (test['date'] > 20170228)]

# 用户一个月的活跃角度
# 一个月的登陆天数
train_log_day = train.groupby(['msno']).date.agg({'log_day': 'count'})
test_log_day = test.groupby(['msno']).date.agg({'log_day': 'count'})
train = pd.merge(train, train_log_day, on=['msno'], how='left')
test = pd.merge(test, test_log_day, on=['msno'], how='left')

# 一个月的听歌汇总
train_total_25_sum = train.groupby(['msno']).num_25.agg({'total_25_sum': np.sum})
train_total_50_sum = train.groupby(['msno']).num_50.agg({'total_50_sum': np.sum})
train_total_75_sum = train.groupby(['msno']).num_75.agg({'total_75_sum': np.sum})
train_total_985_sum = train.groupby(['msno']).num_985.agg({'total_985_sum': np.sum})
train_total_100_sum = train.groupby(['msno']).num_100.agg({'total_100_sum': np.sum})
train_total_unq_sum = train.groupby(['msno']).num_unq.agg({'total_unq_sum': np.sum})
train_total_secs_sum = train.groupby(['msno']).total_secs.agg({'total_secs_sum': np.sum})
train = pd.merge(train, train_total_25_sum, on=['msno'], how='left')
train = pd.merge(train, train_total_50_sum, on=['msno'], how='left')
train = pd.merge(train, train_total_75_sum, on=['msno'], how='left')
train = pd.merge(train, train_total_985_sum, on=['msno'], how='left')
train = pd.merge(train, train_total_100_sum, on=['msno'], how='left')
train = pd.merge(train, train_total_unq_sum, on=['msno'], how='left')
train = pd.merge(train, train_total_secs_sum, on=['msno'], how='left')
train['total_sum'] = train['total_25_sum'] + train['total_50_sum'] + train['total_75_sum'] + train['total_985_sum'] + \
                     train['total_100_sum']
# 一个月的听歌习惯
train['total_25ratio'] = train['total_25_sum'] / train['total_sum']
train['total_100ratio'] = train['total_100_sum'] / train['total_sum']
# 听歌是循环播放还是试听,每首歌播放次数
train['persong_play'] = train['total_sum'] / train['total_unq_sum']
# 听歌每首歌平均播放时间
train['persong_time'] = train['total_secs_sum'] / train['total_sum']
# 平均每天听歌数量
train['daily_play'] = train['total_sum'] / train['log_day']
# 平均每天听歌时间
train['daily_listentime'] = train['total_secs_sum'] / train['log_day']

test_total_25_sum = test.groupby(['msno']).num_25.agg({'total_25_sum': np.sum})
test_total_50_sum = test.groupby(['msno']).num_50.agg({'total_50_sum': np.sum})
test_total_75_sum = test.groupby(['msno']).num_75.agg({'total_75_sum': np.sum})
test_total_985_sum = test.groupby(['msno']).num_985.agg({'total_985_sum': np.sum})
test_total_100_sum = test.groupby(['msno']).num_100.agg({'total_100_sum': np.sum})
test_total_unq_sum = test.groupby(['msno']).num_unq.agg({'total_unq_sum': np.sum})
test_total_secs_sum = test.groupby(['msno']).total_secs.agg({'total_secs_sum': np.sum})
test = pd.merge(test, test_total_25_sum, on=['msno'], how='left')
test = pd.merge(test, test_total_50_sum, on=['msno'], how='left')
test = pd.merge(test, test_total_75_sum, on=['msno'], how='left')
test = pd.merge(test, test_total_985_sum, on=['msno'], how='left')
test = pd.merge(test, test_total_100_sum, on=['msno'], how='left')
test = pd.merge(test, test_total_unq_sum, on=['msno'], how='left')
test = pd.merge(test, test_total_secs_sum, on=['msno'], how='left')
test['total_sum'] = test['total_25_sum'] + test['total_50_sum'] + test['total_75_sum'] + test['total_985_sum'] + test[
    'total_100_sum']
# 一个月的听歌习惯
test['total_25ratio'] = test['total_25_sum'] / test['total_sum']
test['total_100ratio'] = test['total_100_sum'] / test['total_sum']
# 听歌是循环播放还是试听,每首歌播放次数
test['persong_play'] = test['total_sum'] / test['total_unq_sum']
# 听歌每首歌平均播放时间
test['persong_time'] = test['total_secs_sum'] / test['total_sum']
# 平均每天听歌数量
test['daily_play'] = test['total_sum'] / test['log_day']
# 平均每天听歌时间
test['daily_listentime'] = test['total_secs_sum'] / test['log_day']

# Test
print(train.columns)
print(test.columns)
print(len(test))

# Output the file
train.to_csv("../input/train_final.csv", index=False)
test.to_csv("../input/test_final.csv", index=False)
