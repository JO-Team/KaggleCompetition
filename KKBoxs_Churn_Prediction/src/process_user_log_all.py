import gc
import time
import warnings

import numpy as np
import pandas as pd


def process_user_log(train, istest):
    pass

def process_user_log_all():
    pass

# Deal with first part
def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn

gc.enable()

size = 4e6  # 1 million
reader = pd.read_csv('../input/user_logs.csv', chunksize=size, nrows=4e7)
start_time = time.time()
for i in range(10):
    user_log_chunk = next(reader)
    if i == 0:
        train_final = process_user_log(user_log_chunk, 0)
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    else:
        train_final = train_final.append(process_user_log(user_log_chunk, 0))
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    del user_log_chunk

print(len(train_final))
print(train_final.columns)

train_final = process_user_log(train_final, 0)
train_final.columns = train_final.columns.get_level_values(0)

train_final.to_csv("../input/processed_features_train_final_v1.csv", index=False)

del train_final
gc.collect()

reader = pd.read_csv('../input/user_logs.csv', chunksize=size, nrows=4e7)
start_time = time.time()
for i in range(10):
    user_log_chunk = next(reader)
    if i == 0:
        test_final = process_user_log(user_log_chunk, 1)
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    else:
        test_final = test_final.append(process_user_log(user_log_chunk, 1))
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    del user_log_chunk

print(len(test_final))
print(test_final.columns)

test_final = process_user_log(test_final, 1)
test_final.columns = test_final.columns.get_level_values(0)

test_final.to_csv("../input/processed_features_test_final_v1.csv", index=False)

del test_final
gc.collect()

size = 1e6
reader = pd.read_csv('../input/user_logs_v2.csv', chunksize=size)
start_time = time.time()
for i in range(18):
    user_log_chunk = next(reader)
    if i == 0:
        train_final = process_user_log(user_log_chunk, 0)
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    else:
        train_final = train_final.append(process_user_log(user_log_chunk, 0))
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    del user_log_chunk

print(len(train_final))
print(train_final.columns)

train_final = process_user_log(train_final, 0)
train_final.columns = train_final.columns.get_level_values(0)

train_final.to_csv("../input/processed_features_train_final_v2.csv", index=False)

del train_final
gc.collect()

reader = pd.read_csv('../input/user_logs_v2.csv', chunksize=size)
start_time = time.time()
for i in range(18):
    user_log_chunk = next(reader)
    if i == 0:
        test_final = process_user_log(user_log_chunk, 1)
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    else:
        test_final = test_final.append(process_user_log(user_log_chunk, 1))
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    del user_log_chunk

print(len(test_final))
print(test_final.columns)

test_final = process_user_log(test_final, 1)
test_final.columns = test_final.columns.get_level_values(0)

test_final.to_csv("../input/processed_features_test_final_v2.csv", index=False)

del test_final
gc.collect()

print('All Done')