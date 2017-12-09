import gc

import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, LSTM, Dropout
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.models import Model


gc.enable()

transactions = pd.read_csv('../input/processed_transaction_all.csv')

members_v1 = pd.read_csv('../input/members.csv')
members_v2 = pd.read_csv('../input/members_v2.csv')
members = members_v1.append(members_v2, ignore_index=True)

user_log = pd.read_csv('../input/processed_user_log_all.csv')

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

# Delete date for now
train = train.drop(['transaction_date', 'membership_expire_date', 'expiration_date', 'registration_init_time'], axis=1)
test = test.drop(['transaction_date', 'membership_expire_date', 'expiration_date', 'registration_init_time'], axis=1)
# Delete date for now

cols = [c for c in train.columns if c not in ['is_churn', 'msno']]

# LSTM Model
BATCH_SIZE = 1025

lstm_layer = LSTM(75, recurrent_dropout=0.2)

features_input = Input(shape=(train.shape[1],), dtype="float32")
features_dense = BatchNormalization()(features_input)
features_dense = Dense(200, activation="relu")(features_dense)
features_dense = Dropout(0.2)(features_dense)

merged = lstm_layer(features_dense)
merged = BatchNormalization()(merged)
merged = GaussianNoise(0.1)(merged)

merged = Dense(150, activation="relu")(merged)
merged = Dropout(0.2)(merged)
merged = BatchNormalization()(merged)

out = Dense(1, activation="sigmoid")(merged)

model = Model(inputs=[features_input], outputs=out)
model.compile(loss="binary_crossentropy",
              optimizer="nadam")
early_stopping = EarlyStopping(monitor="val_loss", patience=5)
best_model_path = "../input/best_model.h5"
model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([train[cols]], train['is_churn'],
                 epochs=15, batch_size=BATCH_SIZE, shuffle=True,
                 callbacks=[early_stopping, model_checkpoint], verbose=1)

model.load_weights(best_model_path)
print("validation loss:", min(hist.history["val_loss"]))

preds = model.predict([test], batch_size=BATCH_SIZE, verbose=1)

submission = pd.DataFrame({"test_id": test["msno"], "is_churn": preds.ravel()})
print(len(submission))
submission.to_csv("submission_lstm_baseline_Dec_8.csv", index=False)
