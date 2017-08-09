# *coding=utf-8*

import pandas as pd
import numpy as np

#读取每个order,每个product的再购率
X_test=pd.read_csv('kunx_test.csv')
#按再购率进行由高到低排序
X_test=X_test.sort_values(by='score',ascending=False)

def calculate(test_group):

    #test['preds']=np.array(test_group.groupby('order_id')['score'].apply(set))
    # score=[]
    # score=np.array(score.append(test_group['score'][:len(test_group)]))
    # print(score)
    # print_best_prediction(test['preds'])

    df = test_group.copy()
    order_id = np.int(df.iloc[0]['order_id'])
    print(order_id)

    products, preds = (zip(*df.sort_values('score', ascending=False)[['product_id', 'score']].values))
    print(products)
    products=np.array(products).astype(int)
    print(products)

    (topk,best_none)=F1_faron.cal_ef1(preds)

    print(best_none)
    productslist=np.hstack((products[:topk],best_none))
    print(productslist)

    return pd.DataFrame({'order_id':order_id,'products':productslist})


import F1_faron
#同一个order的product聚合在一起
tmp = X_test.groupby(['order_id'])
#利用ICML2012的算法计算F1求出每个order最可能再次购买的k个product
df_ef1 = tmp.apply(lambda x: calculate(x))

#去除每个product_id的末尾.0
def filterProduct(x):
    if x.endswith('.0'):
        return x[:-2]
    else:
        return x

df_ef1.products=df_ef1.products.map(filterProduct)
#同一个order的product放入一个list中
result=df_ef1.groupby(['order_id']).apply(lambda x: ' '.join(x['products']))
result.to_csv('sub.csv', index=False)