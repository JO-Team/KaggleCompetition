# *coding=utf-8*

import pandas as pd

#读取每个order,每个product的再购率
X_test=pd.read_csv('kunx_test.csv')
#按再购率进行由高到低排序
X_test=X_test.sort_values(by='score',ascending=False)



import F1_kruegger
X_test['true']=0.0
tmp = X_test.groupby(['order_id'])
#利用NIPS2011的算法计算F1求出每个order最可能再次购买的k个product
df_ef1 = tmp.apply(lambda x: F1_kruegger.calc_approx_ef1(x))

#输出的结果不完整，只有product_list没有order_id!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
df_ef1.to_csv('subsub.csv',index=False)
print(df_ef1)

