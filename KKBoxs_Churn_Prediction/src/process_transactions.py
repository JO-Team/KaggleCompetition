import time
import pandas as pd
import numpy as np

transactions_v1 = pd.read_csv('../input/transactions.csv')
transactions_v2 = pd.read_csv('../input/transactions_v2.csv')

transactions = transactions_v1.append(transactions_v2)

# h=change the type of these series
transactions['payment_method_id'] = transactions['payment_method_id'].astype('int8')
transactions['payment_plan_days'] = transactions['payment_plan_days'].astype('int16')
transactions['plan_list_price'] = transactions['plan_list_price'].astype('int16')
transactions['actual_amount_paid'] = transactions['actual_amount_paid'].astype('int16')
transactions['is_auto_renew'] = transactions['is_auto_renew'].astype('int8')  # changing the type to boolean
transactions['is_cancel'] = transactions['is_cancel'].astype('int8')  # changing the type to boolean
transactions['membership_expire_date'] = pd.to_datetime(transactions['membership_expire_date'].astype(str),
                                                        infer_datetime_format=True, exact=False)
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'].astype(str),

                                                  infer_datetime_format=True, exact=False)
newdf = transactions.join(
    pd.get_dummies(transactions['payment_method_id']))  # creating a new columns for paymenth method id dummyvariable

payment_method_id = {}
for i in transactions['payment_method_id'].unique():
    payment_method_id.update(
        {i: 'payment_method_id{}'.format(i)})  # create a dictionary to automatic renaming of columns

newdf = newdf.rename(columns=payment_method_id)  # renaming the new columns
del newdf['payment_method_id']  # deleting the extra columns

newdf_grouped = newdf.groupby('msno').agg({
                                           'payment_plan_days': np.mean, 'plan_list_price': np.mean,
                                           'actual_amount_paid': np.mean,
                                           'is_auto_renew': np.mean, 'transaction_date': min,
                                           'membership_expire_date': max,
                                           'is_cancel': np.mean,
                                           'payment_method_id2': np.mean,
                                           'payment_method_id3': sum,
                                           'payment_method_id4': np.sum,
                                           'payment_method_id5': np.sum,
                                           'payment_method_id6': np.sum,
                                           'payment_method_id8': np.sum,
                                           'payment_method_id10': np.sum,
                                           'payment_method_id11': np.sum, 'payment_method_id12': np.sum,
                                           'payment_method_id13': np.sum,
                                           'payment_method_id14': np.sum, 'payment_method_id15': np.sum,
                                           'payment_method_id16': np.sum,
                                           'payment_method_id17': np.sum, 'payment_method_id18': np.sum,
                                           'payment_method_id19': np.sum,
                                           'payment_method_id20': np.sum,
                                           'payment_method_id21': np.sum,
                                           'payment_method_id22': np.sum,
                                           'payment_method_id23': np.sum,
                                           'payment_method_id24': np.sum,
                                           'payment_method_id25': np.sum,
                                           'payment_method_id26': np.sum, 'payment_method_id27': np.sum,
                                           'payment_method_id28': np.sum,
                                           'payment_method_id29': np.sum, 'payment_method_id30': np.sum,
                                           'payment_method_id31': np.sum,
                                           'payment_method_id32': np.sum, 'payment_method_id33': np.sum,
                                           'payment_method_id34': np.sum,
                                           'payment_method_id35': np.sum, 'payment_method_id36': np.sum,
                                           'payment_method_id37': np.sum,
                                           'payment_method_id38': np.sum, 'payment_method_id39': np.sum,
                                           'payment_method_id40': np.sum,
                                           'payment_method_id41': np.sum})
