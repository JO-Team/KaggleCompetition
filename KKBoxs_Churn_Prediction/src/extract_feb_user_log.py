import gc
import time

import pandas as pd


def extract_feb_log(train):
    print('Before: ' + str(len(train)))
    train = train[(train['date'] < 20170301) & (train['date'] > 20170131)]
    print('After: ' + str(len(train)))

    return train


gc.enable()

size = 4e7
reader = pd.read_csv('../input/user_logs.csv', chunksize=size)
start_time = time.time()
for i in range(10):
    user_log_chunk = next(reader)
    if i == 0:
        train_final = extract_feb_log(user_log_chunk)
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    else:
        train_final = train_final.append(extract_feb_log(user_log_chunk))
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    del user_log_chunk
    gc.collect()

print(len(train_final))
print(train_final.columns)

train_final = extract_feb_log(train_final)
train_final.columns = train_final.columns.get_level_values(0)

train_final.to_csv("../input/user_log_feb.csv", index=False)

print('Done')
