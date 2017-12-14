import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
import pickle
import gc

from scipy import stats
from pylab import rcParams
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from matplotlib import offsetbox
from matplotlib.ticker import NullFormatter
from sklearn import preprocessing, cross_validation, svm, manifold
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.ensemble import RandomForestClassifier # Load scikit's random forest classifier library
from sklearn.grid_search import GridSearchCV
from time import time
from datetime import datetime, timedelta
from collections import defaultdict
from multiprocessing import Pool, cpu_count


# Helper Functions

# holistic summary of the given data set.
# "remove_bad_rowCol" can be turned on to remove non-informative col / row
def holistic_summary(df, remove_bad_rowCol=False, verbose=True):
    # remove non-informative columns
    if (remove_bad_rowCol):
        df = df.drop(df.columns[df.isnull().sum() >= .9 * len(df)], axis=1)
        df = df.drop(df.index[df.isnull().sum(axis=1) >= .5 * len(df.columns)], axis=0)

    # fix column names:
    df.columns = [c.replace(" ", "_").lower() for c in df.columns]

    print('***************************************************************')
    print('Begin holistic summary: ')
    print('***************************************************************\n')

    print('Dimension of df: ' + str(df.shape))
    print('Percentage of good observations: ' + str(1 - df.isnull().any(axis=1).sum() / len(df)))
    print('---------------------------------------------------------------\n')

    print("Rows with nan values: " + str(df.isnull().any(axis=1).sum()))
    print("Cols with nan values: " + str(df.isnull().any(axis=0).sum()))
    print('Breakdown:')
    print(df.isnull().sum()[df.isnull().sum() != 0])
    print('---------------------------------------------------------------\n')

    print('Columns details: ')
    print('Columns with known dtypes: ')
    good_cols = pd.DataFrame(df.dtypes[df.dtypes != 'object'], columns=['type'])
    good_cols['nan_num'] = [df[col].isnull().sum() for col in good_cols.index]
    good_cols['unique_val'] = [df[col].nunique() for col in good_cols.index]
    good_cols['example'] = [df[col][1] for col in good_cols.index]
    good_cols = good_cols.reindex(good_cols['type'].astype(str).str.len().sort_values().index)
    print(good_cols)
    print('\n')

    try:
        print('Columns with unknown dtypes:')
        bad_cols = pd.DataFrame(df.dtypes[df.dtypes == 'object'], columns=['type'])
        bad_cols['nan_num'] = [df[col].isnull().sum() for col in bad_cols.index]
        bad_cols['unique_val'] = [df[col].nunique() for col in bad_cols.index]
        bad_cols['example(sliced)'] = [str(df[col][1])[:10] for col in bad_cols.index]
        bad_cols = bad_cols.reindex(bad_cols['example(sliced)'].str.len().sort_values().index)
        print(bad_cols)
    except Exception as e:
        print('No columns with unknown dtypes!')
    print('_______________________________________________________________\n\n\n')
    # if not verbose: enablePrint()
    return df


# fixing dtypes: time and numeric variables
def fix_dtypes(df, time_cols, num_cols):
    print('***************************************************************')
    print('Begin fixing data types: ')
    print('***************************************************************\n')

    def fix_time_col(df, time_cols):
        for time_col in time_cols:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce', format='%Y%m%d')
        print('---------------------------------------------------------------')
        print('The following time columns has been fixed: ')
        print(time_cols)
        print('---------------------------------------------------------------\n')

    def fix_num_col(df, num_cols):
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        print('---------------------------------------------------------------')
        print('The following number columns has been fixed: ')
        print(num_cols)
        print('---------------------------------------------------------------\n')

    if (len(num_cols) > 0):
        fix_num_col(df, num_cols)
    fix_time_col(df, time_cols)

    print('---------------------------------------------------------------')
    print('Final data types:')
    result = pd.DataFrame(df.dtypes, columns=['type'])
    result = result.reindex(result['type'].astype(str).str.len().sort_values().index)
    print(result)
    print('_______________________________________________________________\n\n\n')
    return df


# Load in user_logs
def transform_df(df):
    df = pd.DataFrame(df)
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df


def transform_df2(df):
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df


# Memory Reduction
def change_datatype(df):
    int_cols = list(df.select_dtypes(include=['int']).columns)
    for col in int_cols:
        if ((np.max(df[col]) <= 127) and (np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and (np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and (np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)


def plot_roc_curve(svm_clf, X_test, y_test, preds, isRF=False):
    from sklearn.metrics import roc_curve, roc_auc_score

    if isRF:
        y_score = svm_clf.predict_proba(X_test)[:, 1]
    else:
        y_score = svm_clf.decision_function(X_test)
    (false_positive_rate, true_positive_rate, threshold) = roc_curve(y_test, y_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Plot ROC curve
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], ls="--")
    # plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.show()


# Print out the memory usage
def memo(df):
    mem = df.memory_usage(index=True).sum()
    print(mem / 1024 ** 2, " MB")


# Print all the available files
def print_file():
    print(check_output(["ls", "../input"]).decode("utf8"))


gc.enable()

# Load in train and test
train = pd.read_csv('../input/train.csv')
train = train.append(pd.read_csv('../input/train_v2.csv'))
train.index = range(len(train))
test = pd.read_csv('../input/sample_submission_v2.csv')
# test = test.append(pd.read_csv('../input/sample_submission_zero.csv'))
# test.index = range(len(test))

# Load in other files
members = pd.read_csv('../input/members_v3.csv')
change_datatype(members)
print("Memo of members: ")
memo(members)

trans = pd.read_csv('../input/transactions.csv')
trans = trans.append(pd.read_csv('../input/transactions_v2.csv'))
trans.index = range(len(trans))
change_datatype(trans)
print("Memo of trans: ")
memo(trans)

# Loading in user_logs_v2.csv
df_iter = pd.read_csv('../input/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)
last_user_logs = []
i = 0 #~400 Million Records - starting at the end but remove locally if needed
for df in df_iter:
    if i>35: # used to be 35, just testing
        if len(df)>0:
            print(df.shape)
            p = Pool(cpu_count())
            df = p.map(transform_df, np.array_split(df, cpu_count()))
            df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
            df = transform_df2(df)
            p.close(); p.join()
            last_user_logs.append(df)
            print('...', df.shape)
            df = []
    i+=1

last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)
last_user_logs = transform_df2(last_user_logs)
# last_user_logs =  last_user_logs[['msno','num_100', 'num_25', 'num_unq', 'total_secs', 'date']]
print("Memo of last_user_logs: ")
memo(last_user_logs)
