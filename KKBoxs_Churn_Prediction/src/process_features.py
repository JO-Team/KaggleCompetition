# *coding=utf-8*
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
del train_log_day, test_log_day
gc.collect()

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
del train_total_25_sum, train_total_50_sum, train_total_75_sum, train_total_985_sum, train_total_100_sum, train_total_unq_sum, train_total_secs_sum
gc.collect()
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

# train数据两个礼拜的变化
train_one_week = train[(train['date'] < 20170220) & (train['date'] > 20170212)]
train_two_week = train[(train['date'] < 20170227) & (train['date'] > 20170219)]

train_one_week_total_25_sum = train_one_week.groupby(['msno']).num_25.agg({'total_25_sum': np.sum})
train_one_week_total_50_sum = train_one_week.groupby(['msno']).num_50.agg({'total_50_sum': np.sum})
train_one_week_total_75_sum = train_one_week.groupby(['msno']).num_75.agg({'total_75_sum': np.sum})
train_one_week_total_985_sum = train_one_week.groupby(['msno']).num_985.agg({'total_985_sum': np.sum})
train_one_week_total_100_sum = train_one_week.groupby(['msno']).num_100.agg({'total_100_sum': np.sum})
train_one_week_total_secs_sum = train_one_week.groupby(['msno']).total_secs.agg({'one_week_secs_sum': np.sum})
train_one_week = pd.merge(train_one_week, train_one_week_total_25_sum, on=['msno'], how='left')
train_one_week = pd.merge(train_one_week, train_one_week_total_50_sum, on=['msno'], how='left')
train_one_week = pd.merge(train_one_week, train_one_week_total_75_sum, on=['msno'], how='left')
train_one_week = pd.merge(train_one_week, train_one_week_total_985_sum, on=['msno'], how='left')
train_one_week = pd.merge(train_one_week, train_one_week_total_100_sum, on=['msno'], how='left')
train_one_week = pd.merge(train_one_week, train_one_week_total_secs_sum, on=['msno'], how='left')
del train_one_week_total_25_sum, train_one_week_total_50_sum, train_one_week_total_75_sum, train_one_week_total_985_sum, train_one_week_total_100_sum, train_one_week_total_secs_sum
gc.collect()
train_one_week['one_week_sum'] = train_one_week['total_25_sum'] + train_one_week['total_50_sum'] + train_one_week[
    'total_75_sum'] + train['total_985_sum'] + train['total_100_sum']

train_two_week_total_25_sum = train_two_week.groupby(['msno']).num_25.agg({'total_25_sum': np.sum})
train_two_week_total_50_sum = train_two_week.groupby(['msno']).num_50.agg({'total_50_sum': np.sum})
train_two_week_total_75_sum = train_two_week.groupby(['msno']).num_75.agg({'total_75_sum': np.sum})
train_two_week_total_985_sum = train_two_week.groupby(['msno']).num_985.agg({'total_985_sum': np.sum})
train_two_week_total_100_sum = train_two_week.groupby(['msno']).num_100.agg({'total_100_sum': np.sum})
train_two_week_total_secs_sum = train_two_week.groupby(['msno']).total_secs.agg({'two_week_secs_sum': np.sum})
train_two_week = pd.merge(train_two_week, train_two_week_total_25_sum, on=['msno'], how='left')
train_two_week = pd.merge(train_two_week, train_two_week_total_50_sum, on=['msno'], how='left')
train_two_week = pd.merge(train_two_week, train_two_week_total_75_sum, on=['msno'], how='left')
train_two_week = pd.merge(train_two_week, train_two_week_total_985_sum, on=['msno'], how='left')
train_two_week = pd.merge(train_two_week, train_two_week_total_100_sum, on=['msno'], how='left')
train_two_week = pd.merge(train_two_week, train_two_week_total_secs_sum, on=['msno'], how='left')
del train_two_week_total_25_sum, train_two_week_total_50_sum, train_two_week_total_75_sum, train_two_week_total_985_sum, train_two_week_total_100_sum, train_two_week_total_secs_sum
gc.collect()
train_two_week['two_week_sum'] = train_one_week['total_25_sum'] + train_one_week['total_50_sum'] + train_one_week[
    'total_75_sum'] \
                                 + train_one_week['total_985_sum'] + train_one_week['total_100_sum']

train = pd.merge(train, train_one_week[['msno', 'one_week_secs_sum', 'one_week_sum']], on=['msno'], how='left')
train = pd.merge(train, train_two_week[['msno', 'two_week_secs_sum', 'two_week_sum']], on=['msno'], how='left')
del train_one_week, train_two_week
gc.collect()
# 第四周听歌时间与第三周比较
train['week_secs_sum_ratio'] = train['two_week_secs_sum'] / train['one_week_secs_sum']
# 第四周听歌数与第三周比较
train['week_sum_ratio'] = train['two_week_sum'] / train['one_week_sum']

# train数据两个半月的变化
train_one_semimonth = train[(train['date'] < 20170215) & (train['date'] > 20170131)]
train_two_semimonth = train[(train['date'] < 20170301) & (train['date'] > 20170214)]

train_one_semimonth_total_25_sum = train_one_semimonth.groupby(['msno']).num_25.agg({'total_25_sum': np.sum})
train_one_semimonth_total_50_sum = train_one_semimonth.groupby(['msno']).num_50.agg({'total_50_sum': np.sum})
train_one_semimonth_total_75_sum = train_one_semimonth.groupby(['msno']).num_75.agg({'total_75_sum': np.sum})
train_one_semimonth_total_985_sum = train_one_semimonth.groupby(['msno']).num_985.agg({'total_985_sum': np.sum})
train_one_semimonth_total_100_sum = train_one_semimonth.groupby(['msno']).num_100.agg({'total_100_sum': np.sum})
train_one_semimonth_total_secs_sum = train_one_semimonth.groupby(['msno']).total_secs.agg(
    {'one_semimonth_secs_sum': np.sum})
train_one_semimonth = pd.merge(train_one_semimonth, train_one_semimonth_total_25_sum, on=['msno'], how='left')
train_one_semimonth = pd.merge(train_one_semimonth, train_one_semimonth_total_50_sum, on=['msno'], how='left')
train_one_semimonth = pd.merge(train_one_semimonth, train_one_semimonth_total_75_sum, on=['msno'], how='left')
train_one_semimonth = pd.merge(train_one_semimonth, train_one_semimonth_total_985_sum, on=['msno'], how='left')
train_one_semimonth = pd.merge(train_one_semimonth, train_one_semimonth_total_100_sum, on=['msno'], how='left')
train_one_semimonth = pd.merge(train_one_semimonth, train_one_semimonth_total_secs_sum, on=['msno'], how='left')
del train_one_semimonth_total_25_sum, train_one_semimonth_total_50_sum, train_one_semimonth_total_75_sum, train_one_semimonth_total_985_sum, train_one_semimonth_total_100_sum, train_one_semimonth_total_secs_sum
gc.collect()
train_one_semimonth['one_semimonth_sum'] = train_one_semimonth['total_25_sum'] + train_one_semimonth['total_50_sum'] \
                                           + train_one_semimonth['total_75_sum'] + train_one_semimonth[
                                               'total_985_sum'] + train_one_semimonth['total_100_sum']

train_two_semimonth_total_25_sum = train_two_semimonth.groupby(['msno']).num_25.agg({'total_25_sum': np.sum})
train_two_semimonth_total_50_sum = train_two_semimonth.groupby(['msno']).num_50.agg({'total_50_sum': np.sum})
train_two_semimonth_total_75_sum = train_two_semimonth.groupby(['msno']).num_75.agg({'total_75_sum': np.sum})
train_two_semimonth_total_985_sum = train_two_semimonth.groupby(['msno']).num_985.agg({'total_985_sum': np.sum})
train_two_semimonth_total_100_sum = train_two_semimonth.groupby(['msno']).num_100.agg({'total_100_sum': np.sum})
train_two_semimonth_total_secs_sum = train_two_semimonth.groupby(['msno']).total_secs.agg(
    {'two_semimonth_secs_sum': np.sum})
train_two_semimonth = pd.merge(train_two_semimonth, train_two_semimonth_total_25_sum, on=['msno'], how='left')
train_two_semimonth = pd.merge(train_two_semimonth, train_two_semimonth_total_50_sum, on=['msno'], how='left')
train_two_semimonth = pd.merge(train_two_semimonth, train_two_semimonth_total_75_sum, on=['msno'], how='left')
train_two_semimonth = pd.merge(train_two_semimonth, train_two_semimonth_total_985_sum, on=['msno'], how='left')
train_two_semimonth = pd.merge(train_two_semimonth, train_two_semimonth_total_100_sum, on=['msno'], how='left')
train_two_semimonth = pd.merge(train_two_semimonth, train_two_semimonth_total_secs_sum, on=['msno'], how='left')
del train_two_semimonth_total_25_sum, train_two_semimonth_total_50_sum, train_two_semimonth_total_75_sum, train_two_semimonth_total_985_sum, train_two_semimonth_total_100_sum, train_two_semimonth_total_secs_sum
gc.collect()
train_two_semimonth['two_semimonth_sum'] = train_two_semimonth['total_25_sum'] + train_two_semimonth['total_50_sum'] \
                                           + train_two_semimonth['total_75_sum'] + train_two_semimonth[
                                               'total_985_sum'] + train_two_semimonth['total_100_sum']

train = pd.merge(train, train_one_semimonth[['msno', 'one_semimonth_secs_sum', 'one_semimonth_sum']], on=['msno'],
                 how='left')
train = pd.merge(train, train_two_semimonth[['msno', 'two_semimonth_secs_sum', 'two_semimonth_sum']], on=['msno'],
                 how='left')
del train_one_semimonth, train_two_semimonth
gc.collect()
# 第二个半月听歌时间与第一个半月比较
train['semimonth_secs_sum_ratio'] = train['two_semimonth_secs_sum'] / train['one_semimonth_secs_sum']
# 第二个半月听歌数与第一个半月比较
train['semimonth_sum_ratio'] = train['two_semimonth_sum'] / train['one_semimonth_sum']

print(train.columns)
print(len(train))

# Output the file
train.to_csv("../input/train_final.csv", index=False)

del train
gc.collect()
##############################对3月份test数据的处理

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
del test_total_25_sum, test_total_50_sum, test_total_75_sum, test_total_985_sum, test_total_100_sum, test_total_unq_sum, test_total_secs_sum
gc.collect()
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

# test数据两个礼拜的变化
test_one_week = test[(test['date'] < 20170320) & (test['date'] > 20170312)]
test_two_week = test[(test['date'] < 20170327) & (test['date'] > 20170319)]

test_one_week_total_25_sum = test_one_week.groupby(['msno']).num_25.agg({'total_25_sum': np.sum})
test_one_week_total_50_sum = test_one_week.groupby(['msno']).num_50.agg({'total_50_sum': np.sum})
test_one_week_total_75_sum = test_one_week.groupby(['msno']).num_75.agg({'total_75_sum': np.sum})
test_one_week_total_985_sum = test_one_week.groupby(['msno']).num_985.agg({'total_985_sum': np.sum})
test_one_week_total_100_sum = test_one_week.groupby(['msno']).num_100.agg({'total_100_sum': np.sum})
test_one_week_total_secs_sum = test_one_week.groupby(['msno']).total_secs.agg({'one_week_secs_sum': np.sum})
test_one_week = pd.merge(test_one_week, test_one_week_total_25_sum, on=['msno'], how='left')
test_one_week = pd.merge(test_one_week, test_one_week_total_50_sum, on=['msno'], how='left')
test_one_week = pd.merge(test_one_week, test_one_week_total_75_sum, on=['msno'], how='left')
test_one_week = pd.merge(test_one_week, test_one_week_total_985_sum, on=['msno'], how='left')
test_one_week = pd.merge(test_one_week, test_one_week_total_100_sum, on=['msno'], how='left')
test_one_week = pd.merge(test_one_week, test_one_week_total_secs_sum, on=['msno'], how='left')
del test_one_week_total_25_sum, test_one_week_total_50_sum, test_one_week_total_75_sum, test_one_week_total_985_sum, test_one_week_total_100_sum, test_one_week_total_secs_sum
gc.collect()
test_one_week['one_week_sum'] = test_one_week['total_25_sum'] + test_one_week['total_50_sum'] + test_one_week[
    'total_75_sum'] + test['total_985_sum'] + test['total_100_sum']

test_two_week_total_25_sum = test_two_week.groupby(['msno']).num_25.agg({'total_25_sum': np.sum})
test_two_week_total_50_sum = test_two_week.groupby(['msno']).num_50.agg({'total_50_sum': np.sum})
test_two_week_total_75_sum = test_two_week.groupby(['msno']).num_75.agg({'total_75_sum': np.sum})
test_two_week_total_985_sum = test_two_week.groupby(['msno']).num_985.agg({'total_985_sum': np.sum})
test_two_week_total_100_sum = test_two_week.groupby(['msno']).num_100.agg({'total_100_sum': np.sum})
test_two_week_total_secs_sum = test_two_week.groupby(['msno']).total_secs.agg({'two_week_secs_sum': np.sum})
test_two_week = pd.merge(test_two_week, test_two_week_total_25_sum, on=['msno'], how='left')
test_two_week = pd.merge(test_two_week, test_two_week_total_50_sum, on=['msno'], how='left')
test_two_week = pd.merge(test_two_week, test_two_week_total_75_sum, on=['msno'], how='left')
test_two_week = pd.merge(test_two_week, test_two_week_total_985_sum, on=['msno'], how='left')
test_two_week = pd.merge(test_two_week, test_two_week_total_100_sum, on=['msno'], how='left')
test_two_week = pd.merge(test_two_week, test_two_week_total_secs_sum, on=['msno'], how='left')
del test_two_week_total_25_sum, test_two_week_total_50_sum, test_two_week_total_75_sum, test_two_week_total_985_sum, test_two_week_total_100_sum, test_two_week_total_secs_sum
gc.collect()
test_two_week['two_week_sum'] = test_one_week['total_25_sum'] + test_one_week['total_50_sum'] + test_one_week[
    'total_75_sum'] + test_one_week['total_985_sum'] + test_one_week['total_100_sum']

test = pd.merge(test, test_one_week[['msno', 'one_week_secs_sum', 'one_week_sum']], on=['msno'], how='left')
test = pd.merge(test, test_two_week[['msno', 'two_week_secs_sum', 'two_week_sum']], on=['msno'], how='left')
del test_one_week, test_two_week
gc.collect()
# 第四周听歌时间与第三周比较
test['week_secs_sum_ratio'] = test['two_week_secs_sum'] / test['one_week_secs_sum']
# 第四周听歌数与第三周比较
test['week_sum_ratio'] = test['two_week_sum'] / test['one_week_sum']

# test数据两个半月的变化
test_one_semimonth = test[(test['date'] < 20170315) & (test['date'] > 20170228)]
test_two_semimonth = test[(test['date'] < 20170329) & (test['date'] > 20170314)]

test_one_semimonth_total_25_sum = test_one_semimonth.groupby(['msno']).num_25.agg({'total_25_sum': np.sum})
test_one_semimonth_total_50_sum = test_one_semimonth.groupby(['msno']).num_50.agg({'total_50_sum': np.sum})
test_one_semimonth_total_75_sum = test_one_semimonth.groupby(['msno']).num_75.agg({'total_75_sum': np.sum})
test_one_semimonth_total_985_sum = test_one_semimonth.groupby(['msno']).num_985.agg({'total_985_sum': np.sum})
test_one_semimonth_total_100_sum = test_one_semimonth.groupby(['msno']).num_100.agg({'total_100_sum': np.sum})
test_one_semimonth_total_secs_sum = test_one_semimonth.groupby(['msno']).total_secs.agg(
    {'one_semimonth_secs_sum': np.sum})
test_one_semimonth = pd.merge(test_one_semimonth, test_one_semimonth_total_25_sum, on=['msno'], how='left')
test_one_semimonth = pd.merge(test_one_semimonth, test_one_semimonth_total_50_sum, on=['msno'], how='left')
test_one_semimonth = pd.merge(test_one_semimonth, test_one_semimonth_total_75_sum, on=['msno'], how='left')
test_one_semimonth = pd.merge(test_one_semimonth, test_one_semimonth_total_985_sum, on=['msno'], how='left')
test_one_semimonth = pd.merge(test_one_semimonth, test_one_semimonth_total_100_sum, on=['msno'], how='left')
test_one_semimonth = pd.merge(test_one_semimonth, test_one_semimonth_total_secs_sum, on=['msno'], how='left')
del test_one_semimonth_total_25_sum, test_one_semimonth_total_50_sum, test_one_semimonth_total_75_sum, test_one_semimonth_total_985_sum, test_one_semimonth_total_100_sum, test_one_semimonth_total_secs_sum
gc.collect()
test_one_semimonth['one_semimonth_sum'] = test_one_semimonth['total_25_sum'] + test_one_semimonth['total_50_sum'] \
                                          + test_one_semimonth['total_75_sum'] + test_one_semimonth['total_985_sum'] + \
                                          test_one_semimonth['total_100_sum']

test_two_semimonth_total_25_sum = test_two_semimonth.groupby(['msno']).num_25.agg({'total_25_sum': np.sum})
test_two_semimonth_total_50_sum = test_two_semimonth.groupby(['msno']).num_50.agg({'total_50_sum': np.sum})
test_two_semimonth_total_75_sum = test_two_semimonth.groupby(['msno']).num_75.agg({'total_75_sum': np.sum})
test_two_semimonth_total_985_sum = test_two_semimonth.groupby(['msno']).num_985.agg({'total_985_sum': np.sum})
test_two_semimonth_total_100_sum = test_two_semimonth.groupby(['msno']).num_100.agg({'total_100_sum': np.sum})
test_two_semimonth_total_secs_sum = test_two_semimonth.groupby(['msno']).total_secs.agg(
    {'two_semimonth_secs_sum': np.sum})
test_two_semimonth = pd.merge(test_two_semimonth, test_two_semimonth_total_25_sum, on=['msno'], how='left')
test_two_semimonth = pd.merge(test_two_semimonth, test_two_semimonth_total_50_sum, on=['msno'], how='left')
test_two_semimonth = pd.merge(test_two_semimonth, test_two_semimonth_total_75_sum, on=['msno'], how='left')
test_two_semimonth = pd.merge(test_two_semimonth, test_two_semimonth_total_985_sum, on=['msno'], how='left')
test_two_semimonth = pd.merge(test_two_semimonth, test_two_semimonth_total_100_sum, on=['msno'], how='left')
test_two_semimonth = pd.merge(test_two_semimonth, test_two_semimonth_total_secs_sum, on=['msno'], how='left')
del test_two_semimonth_total_25_sum, test_two_semimonth_total_50_sum, test_two_semimonth_total_75_sum, test_two_semimonth_total_985_sum, test_two_semimonth_total_100_sum, test_two_semimonth_total_secs_sum
gc.collect()
test_two_semimonth['two_semimonth_sum'] = test_two_semimonth['total_25_sum'] + test_two_semimonth['total_50_sum'] \
                                          + test_two_semimonth['total_75_sum'] + test_two_semimonth['total_985_sum'] + \
                                          test_two_semimonth['total_100_sum']

test = pd.merge(test, test_one_semimonth[['msno', 'one_semimonth_secs_sum', 'one_semimonth_sum']], on=['msno'],
                how='left')
test = pd.merge(test, test_two_semimonth[['msno', 'two_semimonth_secs_sum', 'two_semimonth_sum']], on=['msno'],
                how='left')
del test_one_semimonth, test_two_semimonth
gc.collect()
# 第二个半月听歌时间与第一个半月比较
test['semimonth_secs_sum_ratio'] = test['two_semimonth_secs_sum'] / test['one_semimonth_secs_sum']
# 第二个半月听歌数与第一个半月比较
test['semimonth_sum_ratio'] = test['two_semimonth_sum'] / test['one_semimonth_sum']

# Test
print(test.columns)
print(len(test))

test.to_csv("../input/test_final.csv", index=False)
