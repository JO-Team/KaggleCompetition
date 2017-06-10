import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import statsmodels.api as sm

micro_humility_factor = 1  # range from 0 (complete humility) to 1 (no humility)
macro_humility_factor = 1

macro = pd.read_csv('../input/macro.csv')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Macro data monthly medians
macro["timestamp"] = pd.to_datetime(macro["timestamp"])
macro["year"] = macro["timestamp"].dt.year
macro["month"] = macro["timestamp"].dt.month
macro["yearmonth"] = 100 * macro.year + macro.month
macmeds = macro.groupby("yearmonth").median()

# Price data monthly medians
train["timestamp"] = pd.to_datetime(train["timestamp"])
train["year"] = train["timestamp"].dt.year
train["month"] = train["timestamp"].dt.month
train["yearmonth"] = 100 * train.year + train.month
prices = train[["yearmonth", "price_doc"]]
p = prices.groupby("yearmonth").median()

# Join monthly prices to macro data
df = macmeds.join(p)

# Function to process Almon lags

import numpy.matlib as ml


def almonZmatrix(X, maxlag, maxdeg):
    """
    Creates the Z matrix corresponding to vector X.
    """
    n = len(X)
    Z = ml.zeros((len(X) - maxlag, maxdeg + 1))
    for t in range(maxlag, n):
        # Solve for Z[t][0].
        Z[t - maxlag, 0] = sum([X[t - lag] for lag in range(maxlag + 1)])
        for j in range(1, maxdeg + 1):
            s = 0.0
            for i in range(1, maxlag + 1):
                s += (i) ** j * X[t - i]
            Z[t - maxlag, j] = s
    return Z


# Prepare data for macro model
y = df.price_doc.div(df.cpi).apply(np.log).loc[201108:201506]
lncpi = df.cpi.apply(np.log)
tblags = 5  # Number of lags used on PDL for Trade Balance
mrlags = 5  # Number of lags used on PDL for Mortgage Rate
cplags = 5  # Number of lags used on PDL for CPI
ztb = almonZmatrix(df.balance_trade.loc[201103:201506].as_matrix(), tblags, 1)
zmr = almonZmatrix(df.mortgage_rate.loc[201103:201506].as_matrix(), mrlags, 1)
zcp = almonZmatrix(lncpi.loc[201103:201506].as_matrix(), cplags, 1)
columns = ['tb0', 'tb1', 'mr0', 'mr1', 'cp0', 'cp1']
z = pd.DataFrame(np.concatenate((ztb, zmr, zcp), axis=1), y.index.values, columns)
X = sm.add_constant(z)

# Fit macro model
eq = sm.OLS(y, X)
fit = eq.fit()

# Predict with macro model
test_cpi = df.cpi.loc[201507:201605]
test_index = test_cpi.index
ztb_test = almonZmatrix(df.balance_trade.loc[201502:201605].as_matrix(), tblags, 1)
zmr_test = almonZmatrix(df.mortgage_rate.loc[201502:201605].as_matrix(), mrlags, 1)
zcp_test = almonZmatrix(lncpi.loc[201502:201605].as_matrix(), cplags, 1)
z_test = pd.DataFrame(np.concatenate((ztb_test, zmr_test, zcp_test), axis=1),
                      test_index, columns)
X_test = sm.add_constant(z_test)
pred_lnrp = fit.predict(X_test)
pred_p = np.exp(pred_lnrp) * test_cpi

# Merge with test cases and compute mean for macro prediction
test["timestamp"] = pd.to_datetime(test["timestamp"])
test["year"] = test["timestamp"].dt.year
test["month"] = test["timestamp"].dt.month
test["yearmonth"] = 100 * test.year + test.month
test_ids = test[["yearmonth", "id"]]
monthprices = pd.DataFrame({"yearmonth": pred_p.index.values, "monthprice": pred_p.values})
macro_mean = np.exp(test_ids.merge(monthprices, on="yearmonth").monthprice.apply(np.log).mean())
macro_mean

# Naive macro model assumes housing prices will simply follow CPI
naive_pred_lnrp = y.mean()
naive_pred_p = np.exp(naive_pred_lnrp) * test_cpi
monthnaive = pd.DataFrame({"yearmonth": pred_p.index.values, "monthprice": naive_pred_p.values})
macro_naive = np.exp(test_ids.merge(monthnaive, on="yearmonth").monthprice.apply(np.log).mean())
macro_naive

# Combine naive and substantive macro models
macro_mean = macro_naive * (macro_mean / macro_naive) ** macro_humility_factor

df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])

# Altervative validation

# df_train = df_train[df_train.price_doc > 1000000]


# Subsampling
trainsub = df_train[df_train.timestamp < '2015-01-01']
trainsub = trainsub[trainsub.product_type == "Investment"]

ind_1m = trainsub[trainsub.price_doc <= 1000000].index
ind_2m = trainsub[trainsub.price_doc == 2000000].index
ind_3m = trainsub[trainsub.price_doc == 3000000].index

train_index = set(df_train.index.copy())

for ind, gap in zip([ind_1m, ind_2m, ind_3m], [10, 3, 2]):
    ind_set = set(ind)
    ind_set_cut = ind.difference(set(ind[::gap]))

    train_index = train_index.difference(ind_set_cut)

df_train = df_train.loc[train_index]

y_train = np.log1p(df_train['price_doc'].values)
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

# Build df_all = (df_train+df_test).join(df_macro)
num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')
print(df_all.shape)

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)

# In[5]:

factorize = lambda t: pd.factorize(t[1])[0]

df_obj = df_all.select_dtypes(include=['object'])

X_all = np.c_[
    df_all.select_dtypes(exclude=['object']).values,
    np.array(list(map(factorize, df_obj.iteritems()))).T
]
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]

# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)

# Convert to numpy values
X_all = df_values.values
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]

df_columns = df_values.columns

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)

cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
                   verbose_eval=20, show_stdv=True)
cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()
num_boost_round = len(cv_result)

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_round)

y_pred = model.predict(dtest)
lnm = np.log(macro_mean)

# I'm not sure whether this makes any sense or not.
# 1+lny.mean()-lnm term is meant to offest the scale effect of the logarithmic mean shift
#   while allowing the new logarithmic mean to remain at lnm.
y_trans = lnm + micro_humility_factor * (y_pred - y_pred.mean()) * (1 + y_pred.mean() - lnm)
y_predict = np.exp(y_trans)

df_sub = pd.DataFrame({'id': id_test, 'price_doc': np.exp(y_trans)})

df_sub.to_csv('sub.csv', index=False)