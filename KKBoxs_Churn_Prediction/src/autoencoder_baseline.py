import gc

import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense
from keras.models import Sequential
from numpy import random as rm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

gc.enable()

transactions = pd.read_csv('../input/processed_transaction_all.csv')

members_v1 = pd.read_csv('../input/members.csv')
members_v2 = pd.read_csv('../input/members_v2.csv')
members = members_v1.append(members_v2, ignore_index=True)

user_log_train = pd.read_csv('../input/processed_features_user_log_feb.csv')
user_log_test = pd.read_csv('../input/processed_features_user_log_mar.csv')
user_log_all = pd.read_csv('../input/processed_user_log_all.csv')

train_v1 = pd.read_csv('../input/train.csv')
train_v2 = pd.read_csv('../input/train_v2.csv')
train = train_v1.append(train_v2, ignore_index=True)

test = pd.read_csv('../input/sample_submission_v2.csv')

# Merge Data

train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')

train = pd.merge(train, user_log_train, how='left', on='msno')
test = pd.merge(test, user_log_test, how='left', on='msno')

train = pd.merge(train, user_log_all, how='left', on='msno')
test = pd.merge(test, user_log_all, how='left', on='msno')

train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno')

# Drop duplicates first
test = test.drop_duplicates('msno')

del transactions, members_v1, members_v2, members, user_log_train, user_log_test, user_log_all, train_v1, train_v2
gc.collect()

gender = {'male': 1, 'female': 2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)

train = train.fillna(0)
test = test.fillna(0)

# Delete date for now
train = train.drop(['transaction_date', 'membership_expire_date', 'expiration_date', 'registration_init_time'], axis=1)
test = test.drop(['transaction_date', 'membership_expire_date', 'expiration_date', 'registration_init_time'], axis=1)
# Delete date for now

cols = [c for c in train.columns if c not in ['is_churn', 'msno']]

# train['is_churn'] = keras.utils.to_categorical(train['is_churn'], num_classes=2)

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


# Add Normalize
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(train)
train_normalized = pd.DataFrame(np_scaled)

print(train_normalized.head(5))

X_train, X_test = train_test_split(train_normalized, test_size=0.2, random_state=47, shuffle=True)
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
autoencoder.add(Dense(int(input_dim / 2), activation='relu'))
# autoencoder.add(Dense(int(input_dim / 2), activation='relu'))
autoencoder.add(Dense(1, activation='sigmoid'))

autoencoder.summary()

nb_epoch = 100
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
test.drop(cols, axis=1, inplace=True)

test.to_csv('submission_autoencoder_all_features_Normalize_fractional_balanced_200_32_Dec_14.csv', index=False)
