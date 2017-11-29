import numpy as np
import pandas as pd

transactions_v1 = pd.read_csv('../input/transactions.csv', nrows=1000)
transactions_v2 = pd.read_csv('../input/transactions_v2.csv', nrows=1000)

members_v1 = pd.read_csv('../input/members.csv', nrows=1000)
members_v2 = pd.read_csv('../input/members.csv', nrows=1000)

user_log_v1 = pd.read_csv('../input/processed_user_log.csv', nrows=1000)
user_log_v2 = pd.read_csv('../input/processed_user_log_v2.csv', nrows=1000)


train_v1 = pd.read_csv('../input/train.csv', nrows=1000)
train_v2 = pd.read_csv('../input/train_v2.csv', nrows=1000)

test = pd.read_csv('../input/sample_submission_v2.csv', nrows=1000)

transactions = transactions_v1.append(transactions_v2)
members = members_v1.append(members_v2)
user_log = user_log_v1.append(user_log_v2)
train = train_v1.append(train_v2)
