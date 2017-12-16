import gc

import numpy as np
import pandas as pd


def calculate_user_log_features(train):
    """
    Calculate the user log features.
    :param train:
    :return:
    """
    train['total_monthly_sum'] = train['total_25_sum_monthly'] + train['total_50_sum_monthly'] + train[
        'total_75_sum_monthly'] + train['total_985_sum_monthly'] + train['total_100_sum_monthly']

    # Monthly Habit for listening to music
    train['total_25_ratio'] = train['total_25_sum_monthly'] / train['total_monthly_sum']
    train['total_100_ratio'] = train['total_100_sum_monthly'] / train['total_monthly_sum']

    # 听歌是循环播放还是试听,每首歌播放次数
    train['persong_play'] = train['total_monthly_sum'] / train['total_unq_sum_monthly']

    # 听歌每首歌平均播放时间
    train['persong_time'] = train['total_secs_sum_monthly'] / train['total_monthly_sum']

    # 平均每天听歌数量
    train['daily_play'] = train['total_monthly_sum'] / train['log_day_monthly']

    # 平均每天听歌时间
    train['daily_listentime'] = train['total_secs_sum_monthly'] / train['log_day_monthly']

    train.replace(np.inf, 0, inplace=True)
    train = train.fillna(0)

    return train


train = pd.read_csv('../input/processed_user_log_mid_all.csv')

train = calculate_user_log_features(train)

print(len(train))

train.to_csv('../input/processed_features_user_log_all_time.csv', index=False)
