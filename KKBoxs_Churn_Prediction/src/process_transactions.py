import numpy as np
import pandas as pd

transactions_v1 = pd.read_csv('../input/transactions.csv')
transactions_v2 = pd.read_csv('../input/transactions_v2.csv')

transactions = transactions_v1.append(transactions_v2, ignore_index=True)

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
    'payment_method_id3': np.sum,
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

# ################################################################################################################

newdf_grouped = newdf_grouped.sort_values(by=['transaction_date'], ascending=[False]).reset_index()
newdf_grouped = newdf_grouped.drop_duplicates(subset=['msno'], keep='first')

# discount
newdf_grouped['discount'] = newdf_grouped['plan_list_price'] - newdf_grouped['actual_amount_paid']
# amt_per_day
newdf_grouped['amt_per_day'] = newdf_grouped['actual_amount_paid'] / newdf_grouped['payment_plan_days']
# is_discount
newdf_grouped['is_discount'] = newdf_grouped.discount.apply(lambda x: 1 if x > 0 else 0)
# membership_duration
newdf_grouped['membership_days'] = pd.to_datetime(newdf_grouped['membership_expire_date']).subtract(pd.to_datetime(
    newdf_grouped['transaction_date'])).dt.days.astype(int)

newdf_grouped['transaction_date_year'] = newdf_grouped['transaction_date'].apply(lambda x: int(str(x)[:4]))
newdf_grouped['transaction_date_month'] = newdf_grouped['transaction_date'].apply(lambda x: int(str(x)[4:6]))
newdf_grouped['transaction_date_day'] = newdf_grouped['transaction_date'].apply(lambda x: int(str(x)[-2:]))

newdf_grouped['membership_expire_date_year'] = newdf_grouped['membership_expire_date'].apply(lambda x: int(str(x)[:4]))
newdf_grouped['membership_expire_date_month'] = newdf_grouped['membership_expire_date'].apply(
    lambda x: int(str(x)[4:6]))
newdf_grouped['membership_expire_date_day'] = newdf_grouped['membership_expire_date'].apply(lambda x: int(str(x)[-2:]))

newdf_grouped['transaction_date_year'] = newdf_grouped['transaction_date_year'].astype(np.int16)
newdf_grouped['transaction_date_month'] = newdf_grouped['transaction_date_month'].astype(np.int8)
newdf_grouped['transaction_date_day'] = newdf_grouped['transaction_date_day'].astype(np.int8)

newdf_grouped['membership_expire_date_year'] = newdf_grouped['membership_expire_date_year'].astype(np.int16)
newdf_grouped['membership_expire_date_month'] = newdf_grouped['membership_expire_date_month'].astype(np.int8)
newdf_grouped['membership_expire_date_day'] = newdf_grouped['membership_expire_date_day'].astype(np.int8)

#################################################################################################################
print(newdf_grouped.head())
print(len(newdf_grouped))

newdf_grouped.to_csv('../input/processed_transaction_features.csv')
