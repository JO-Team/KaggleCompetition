import gc

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout
from keras.models import Sequential
from numpy import random as rm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

gc.enable()

transactions_train = pd.read_csv('../input/processed_transaction_features_feb.csv', index_col=0)
transactions_test = pd.read_csv('../input/processed_transaction_features_mar.csv', index_col=0)
transactions = pd.read_csv('../input/processed_transaction_features.csv', index_col=0)

transactions = transactions[
    ['msno', 'discount', 'amt_per_day', 'is_discount', 'membership_days', 'transaction_date_year',
     'transaction_date_month',
     'transaction_date_day', 'membership_expire_date_year', 'membership_expire_date_month',
     'membership_expire_date_day']]

members = pd.read_csv('../input/members_v3.csv')

user_log_train = pd.read_csv('../input/processed_features_user_log_feb.csv')
user_log_test = pd.read_csv('../input/processed_features_user_log_mar.csv')
user_log_all = pd.read_csv('../input/processed_user_log_all.csv')

train = pd.read_csv('../input/train_v2.csv')

test = pd.read_csv('../input/sample_submission_v2.csv')

# Merge Data

train = pd.merge(train, transactions_train, how='left', on='msno')
test = pd.merge(test, transactions_test, how='left', on='msno')

train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')

train = pd.merge(train, user_log_train, how='left', on='msno')
test = pd.merge(test, user_log_test, how='left', on='msno')

train = pd.merge(train, user_log_all, how='left', on='msno')
test = pd.merge(test, user_log_all, how='left', on='msno')

train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno')

del transactions, members, user_log_train, user_log_test
gc.collect()

# Drop duplicates first
test = test.drop_duplicates('msno')

gender = {'male': 1, 'female': 2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)

train['bd'] = train['bd'].replace(0, train['bd'].mode())
test['bd'] = test['bd'].replace(0, test['bd'].mode())

train['gender'] = train['gender'].replace(0, train['gender'].mean())
test['gender'] = test['gender'].replace(0, test['gender'].mean())

# Delete date for now
train = train.drop(['transaction_date', 'membership_expire_date', 'registration_init_time'], axis=1)
test = test.drop(['transaction_date', 'membership_expire_date', 'registration_init_time'], axis=1)

# Remove Features with 0 feature importance
train = train.drop(
    ['payment_method_id14',
     'payment_method_id18',
     'payment_method_id21',
     'payment_method_id26',
     'payment_method_id35',
     'transaction_date_month_x',
     'transaction_date_day_x',
     'membership_expire_date_year_x',
     'membership_expire_date_month_x',
     'membership_expire_date_day_x',
     'transaction_date_day_y',
     'membership_expire_date_day_y'], axis=1)
test = test.drop(
    ['payment_method_id14',
     'payment_method_id18',
     'payment_method_id21',
     'payment_method_id26',
     'payment_method_id35',
     'transaction_date_month_x',
     'transaction_date_day_x',
     'membership_expire_date_year_x',
     'membership_expire_date_month_x',
     'membership_expire_date_day_x',
     'transaction_date_day_y',
     'membership_expire_date_day_y'], axis=1)

# Remove Features with feature importance less than 100
train = train.drop(
    ['payment_method_id16',
     'payment_method_id17',
     'payment_method_id19',
     'payment_method_id23',
     'payment_method_id27',
     'payment_method_id28',
     'payment_method_id31',
     'is_discount_x',
     'transaction_date_year_x'], axis=1)
test = test.drop(
    ['payment_method_id16',
     'payment_method_id17',
     'payment_method_id19',
     'payment_method_id23',
     'payment_method_id27',
     'payment_method_id28',
     'payment_method_id31',
     'is_discount_x',
     'transaction_date_year_x'], axis=1)

train['autorenew_&_not_cancel'] = ((train.is_auto_renew == 1) == (train.is_cancel == 0)).astype(np.int8)
test['autorenew_&_not_cancel'] = ((test.is_auto_renew == 1) == (test.is_cancel == 0)).astype(np.int8)

train['notAutorenew_&_cancel'] = ((train.is_auto_renew == 0) == (train.is_cancel == 1)).astype(np.int8)
test['notAutorenew_&_cancel'] = ((test.is_auto_renew == 0) == (test.is_cancel == 1)).astype(np.int8)

train = train.replace([np.inf, -np.inf], np.nan)

train = train.fillna(0)
test = test.fillna(0)

train_0 = train[train['is_churn'] == 0]
train_1 = train[train['is_churn'] == 1]



'''
# Enlarge train_1 for 17 times
train_append = train_1

for _ in range(17):
    train_append = train_append.append(train_1)

train = train_0.append(train_append)
'''


# train1 random sample 1/17
def rand_rows(df, num_rows=5):
    subset = rm.choice(df.index.values, size=num_rows)
    return df.loc[subset]


train_0 = rand_rows(train_0, len(train_1))
train = train_0.append(train_1)

cols = [c for c in train.columns if c not in ['is_churn', 'msno']]

# Add Normalize
min_max_scaler = preprocessing.MinMaxScaler()
train[cols] = min_max_scaler.fit_transform(train[cols])

print(train.head(5))

X_train, X_test = train_test_split(train, test_size=0.2, random_state=47, shuffle=True)
y_train = X_train['is_churn']
X_train = X_train.drop(['msno', 'is_churn'], axis=1)

y_test = X_test['is_churn']
X_test = X_test.drop(['msno', 'is_churn'], axis=1)

X_train = X_train.values
X_test = X_test.values

input_dim = X_train.shape[1]

autoencoder = Sequential()
autoencoder.add(Dense(input_dim, input_dim=input_dim))
autoencoder.add(Dense(int(input_dim / 2), activation='relu'))
autoencoder.add(Dropout(0.4))
autoencoder.add(Dense(int(input_dim / 2), activation='relu'))
autoencoder.add(Dropout(0.2))
# autoencoder.add(Dense(int(input_dim / 2), activation='relu'))
autoencoder.add(Dense(1, activation='sigmoid'))

autoencoder.summary()

nb_epoch = 200
batch_size = 32

autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=1,
                               save_best_only=True)

tensorboard = TensorBoard(log_dir='./log',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

print(X_train.shape)

history = autoencoder.fit(X_train, y_train,
                          epochs=nb_epoch,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(X_test, y_test),
                          verbose=1,
                          callbacks=[checkpointer, tensorboard]).history

# autoencoder = load_model('model.h5')

predictions = autoencoder.predict(test.drop(['msno', 'is_churn'], axis=1).values)

test['is_churn'] = predictions
test = test[['msno', 'is_churn']]

test.to_csv('submission_autoencoder_features_selection_fractional_200_32_Dec_15.csv', index=False)
