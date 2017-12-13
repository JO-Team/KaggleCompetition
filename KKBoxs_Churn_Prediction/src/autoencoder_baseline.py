import gc

import pandas as pd
from keras import regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

gc.enable()

transactions = pd.read_csv('../input/processed_transaction_all.csv')

members_v1 = pd.read_csv('../input/members.csv')
members_v2 = pd.read_csv('../input/members_v2.csv')
members = members_v1.append(members_v2, ignore_index=True)

user_log_train = pd.read_csv('../input/processed_features_user_log_feb.csv')
user_log_test = pd.read_csv('../input/processed_features_user_log_mar.csv')

train_v1 = pd.read_csv('../input/train.csv')
train_v2 = pd.read_csv('../input/train_v2.csv')
train = train_v1.append(train_v2, ignore_index=True)

test = pd.read_csv('../input/sample_submission_v2.csv')

# Merge Data

train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')

train = pd.merge(train, user_log_train, how='left', on='msno')
test = pd.merge(test, user_log_test, how='left', on='msno')

train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno')

# Drop duplicates first
test = test.drop_duplicates('msno')

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

X_train, X_test = train_test_split(train, test_size=0.2, random_state=47)
X_train = X_train.drop(['msno', 'is_churn'], axis=1)

y_test = X_test['is_churn']
X_test = X_test.drop(['msno', 'is_churn'], axis=1)

X_train = X_train.values
X_test = X_test.values

input_dim = X_train.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim,))

encoder = Dense(encoding_dim, activation="tanh",
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)

decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

# nb_epoch = 100
nb_epoch = 1
# batch_size = 32
batch_size = 5

autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = autoencoder.fit(X_train, X_train,
                          epochs=nb_epoch,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(X_test, X_test),
                          verbose=1,
                          callbacks=[checkpointer, tensorboard]).history

# autoencoder = load_model('model.h5')

predictions = autoencoder.predict(test.drop(['msno', 'is_churn'], axis=1).values)
test['is_churn'] = predictions
test.drop(cols, axis=1, inplace=True)

print(len(test))

# test.to_csv('submission_autoencoder_features_user_log_transactions_baseline_Dec_13.csv', index=False)
