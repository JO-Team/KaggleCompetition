import gc
import time
import warnings

import numpy as np
import pandas as pd


def process_user_log(df):
    """
    Only do simple sum. mean operation.
    :param df: chunk dataframe from very large file.
    :return: processed dataframe
    """

    # Divided DataFrame by date
    # train = train[(train['date'] < 20170301) & (train['date'] > 20170131)]

    # Stage 1: One Month Total Data
    grouped_object = df.groupby('msno', sort=False).reset_index()  # not sorting results in a minor speedup
    func = {'date': ['min', 'max', 'count'],
            'num_25': ['sum'], 'num_50': ['sum'],
            'num_75': ['sum'], 'num_985': ['sum'],
            'num_100': ['sum'], 'num_unq': ['sum'], 'total_secs': ['sum']}
    one_moneth = grouped_object.agg(func)
    print(one_moneth.columns)

    # Stage 2: Week Total Data
    # Divided DataFrame by Two Week
    one_week = df[(df['date'] < 20170220) & (df['date'] > 20170212)]
    two_week = df[(df['date'] < 20170227) & (df['date'] > 20170219)]

    grouped_object = one_week.groupby('msno', sort=False).reset_index()
    one_week = grouped_object.agg(func)
    print(one_week.columns)

    grouped_object = two_week.groupby('msno', sort=False).reset_index()
    two_week = grouped_object.agg(func)
    print(two_week.columns)

    # Stage 1: Semimonth Total Data
    one_semimonth = df[(df['date'] < 20170215) & (df['date'] > 20170131)]
    two_semimonth = df[(df['date'] < 20170301) & (df['date'] > 20170214)]

    grouped_object = one_semimonth.groupby('msno', sort=False).reset_index()
    one_semimonth = grouped_object.agg(func)
    print(one_semimonth.columns)

    grouped_object = two_semimonth.groupby('msno', sort=False).reset_index()
    two_semimonth = grouped_object.agg(func)
    print(two_semimonth.columns)

    return df


def process_user_log_together(df):
    pass

gc.enable()

# size = 1e6
size = 1000
reader = pd.read_csv('../input/user_log_feb.csv', chunksize=size, nrows=10000)
start_time = time.time()
for i in range(17):
    user_log_chunk = next(reader)
    if i == 0:
        train_final = process_user_log(user_log_chunk)
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    else:
        train_final = train_final.append(process_user_log(user_log_chunk))
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    del user_log_chunk

print(len(train_final))
print(train_final.columns)

train_final = process_user_log_together(train_final)
train_final.columns = train_final.columns.get_level_values(0)

print(train_final.head(10))

# train_final.to_csv("../input/processed_user_log_feb.csv", index=False)

print('Done')
