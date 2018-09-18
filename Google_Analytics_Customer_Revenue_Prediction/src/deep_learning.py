import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import optimizers
from keras.layers import Dense, BatchNormalization, Activation
from keras.models import Sequential
from pandas.io.json import json_normalize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


def add_time_features(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['weekday'] = df['date'].apply(lambda x: x.weekday())

    return df


def plot_metrics(loss, val_loss):
    fig, (ax1) = plt.subplots(1, 1, sharex='col', figsize=(20, 7))
    ax1.plot(loss, label='Train loss')
    ax1.plot(val_loss, label='Validation loss')
    ax1.legend(loc='best')
    ax1.set_title('Loss')
    plt.xlabel('Epochs')


def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path, dtype={'fullVisitorId': 'str'}, nrows=nrows)

    for column in JSON_COLUMNS:
        df = df.join(pd.DataFrame(df.pop(column).apply(pd.io.json.loads).values.tolist(), index=df.index))

    return df


train = load_df("../input/train.csv")
test = load_df("../input/test.csv")

print('TRAIN SET')
print('Rows: %s' % train.shape[0])
print('Columns: %s' % train.shape[1])
print('Features: %s' % train.columns.values)
print()
print('TEST SET')
print('Rows: %s' % test.shape[0])
print('Columns: %s' % test.shape[1])
print('Features: %s' % test.columns.values)

train = add_time_features(train)
test = add_time_features(test)
# Convert target feature to 'float' type.
train["transactionRevenue"] = train["transactionRevenue"].astype('float')

time_agg = train.groupby('date')['transactionRevenue'].agg(['count', 'sum'])
year_agg = train.groupby('year')['transactionRevenue'].agg(['sum'])
month_agg = train.groupby('month')['transactionRevenue'].agg(['sum'])
day_agg = train.groupby('day')['transactionRevenue'].agg(['sum'])
weekday_agg = train.groupby('weekday')['transactionRevenue'].agg(['count', 'sum'])

train = train.drop(['adwordsClickInfo'], axis=1)
test = test.drop(['adwordsClickInfo'], axis=1)
# Drop column that exists only in train data
train = train.drop(['campaignCode'], axis=1)
# Input missing transactionRevenue values
train["transactionRevenue"].fillna(0, inplace=True)

test_ids = test["fullVisitorId"].values

# Unwanted columns
unwanted_columns = ['fullVisitorId', 'sessionId', 'visitId', 'visitStartTime',
                    'browser', 'browserSize', 'browserVersion', 'flashVersion',
                    'mobileDeviceInfo', 'mobileDeviceMarketingName', 'mobileDeviceModel',
                    'mobileInputSelector', 'operatingSystemVersion', 'screenColors',
                    'metro', 'networkDomain', 'networkLocation', 'adContent', 'campaign',
                    'isTrueDirect', 'keyword', 'referralPath', 'source', 'operatingSystem',
                    'date', 'day']

train = train.drop(unwanted_columns, axis=1)
test = test.drop(unwanted_columns, axis=1)
# Constant columns
constant_columns = [c for c in train.columns if train[c].nunique() <= 1]
print('Columns with constant values: ', constant_columns)
train = train.drop(constant_columns, axis=1)
test = test.drop(constant_columns, axis=1)
# Columns with more than 50% null data
high_null_columns = [c for c in train.columns if train[c].count() <= len(train) * 0.5]
print('Columns more than 50% null values: ', high_null_columns)
train = train.drop(high_null_columns, axis=1)
test = test.drop(high_null_columns, axis=1)

reduce_features = ['city', 'year', 'medium', 'channelGrouping',
                   'region', 'subContinent', 'country']
train = train.drop(reduce_features, axis=1)
test = test.drop(reduce_features, axis=1)

categorical_features = ['deviceCategory', 'isMobile', 'continent', 'month', 'weekday']
train = pd.get_dummies(train, columns=categorical_features)
test = pd.get_dummies(test, columns=categorical_features)

# Get labels
train_labels = train['transactionRevenue'].values
train = train.drop(['transactionRevenue'], axis=1)
# Log transform the labels
train_labels = np.log1p(train_labels)

# align both data sets (by outer join), to make they have the same amount of features,
# this is required because of the mismatched categorical values in train and test sets
train, test = train.align(test, join='outer', axis=1)

# replace the nan values added by align for 0
train.replace(to_replace=np.nan, value=0, inplace=True)
test.replace(to_replace=np.nan, value=0, inplace=True)

X_train, X_val, Y_train, Y_val = train_test_split(train, train_labels, test_size=0.1, random_state=1)

normalized_features = ['visitNumber', 'hits', 'pageviews']

# Normalize using Min-Max scaling
scaler = preprocessing.MinMaxScaler()
X_train[normalized_features] = scaler.fit_transform(X_train[normalized_features])
X_val[normalized_features] = scaler.transform(X_val[normalized_features])
test[normalized_features] = scaler.transform(test[normalized_features])

BATCH_SIZE = 64
EPOCHS = 70
LEARNING_RATE = 0.001

model = Sequential()

model.add(Dense(128, kernel_initializer='glorot_normal', activation='relu', input_dim=X_train.shape[1]))

model.add(Dense(64, kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(16, kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1))

adam = optimizers.adam(lr=LEARNING_RATE)
model.compile(loss='mse', optimizer=adam)

print('Dataset size: %s' % X_train.shape[0])
print('Epochs: %s' % EPOCHS)
print('Learning rate: %s' % LEARNING_RATE)
print('Batch size: %s' % BATCH_SIZE)
print('Input dimension: %s' % X_train.shape[1])
print('Features used: %s' % train.columns.values)

print(model.summary())

history = model.fit(x=X_train.values, y=Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    verbose=1, validation_data=(X_val.values, Y_val))

plot_metrics(history.history['loss'], history.history['val_loss'])

predictions = model.predict(test)

submission = pd.DataFrame({"fullVisitorId": test_ids})
predictions[predictions < 0] = 0
submission["PredictedLogRevenue"] = predictions
# submission["PredictedLogRevenue"] = np.expm1(predictions)
submission = submission.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
submission.columns = ["fullVisitorId", "PredictedLogRevenue"]
# submission["PredictedLogRevenue"] = np.log1p(submission["PredictedLogRevenue"])
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"]
submission.to_csv("submission.csv", index=False)
