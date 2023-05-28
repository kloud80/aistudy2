import pandas as pd
from glob import glob
import os
import numpy as np

import matplotlib.pyplot as plt
import sklearn.metrics
import statsmodels.api as sm

from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')

flist = glob('*.csv')

data = pd.read_csv(flist[0], encoding='cp949', sep=',', skiprows=15, dtype='str')

data['거래금액(만원)'] = data['거래금액(만원)'].str.replace(',', '').astype('int')
data['전용면적(㎡)'] = data['전용면적(㎡)'].astype('float')
data['층'] = data['층'].astype('int')
data['건축년도'] = data['건축년도'].astype('int')
data.dtypes


data['시군구'].value_counts()

data['년식'] = 2023- data['건축년도']
data['년식제곱'] = data['년식'] **2


X = np.array(data[['전용면적(㎡)', '층', '년식', '년식제곱']].values)
y = np.array(data['거래금액(만원)'].values)
y = y.reshape([y.shape[0],1])

results = sm.OLS(y, X).fit()
print(results.summary())

pred_y = results.predict(X)
pred_y = pred_y.reshape([pred_y.shape[0], 1])
test = np.concatenate([y, pred_y], axis=1)

test = pd.DataFrame(test, columns=['y', 'pred_y'])

test['error'] = test['y'] - test['pred_y']
test['error'] = test['error']**2

test['error'].sum()

SE = 16142862400034.926
TE = sum((y - y.mean())**2)
TE = TE[0]

TE - SE

16142862400034.926
31556939826895.582

r2_score(y, pred_y)


data['시구구명'] = data['시군구'].apply(lambda x : x.split(' ')[1])

data['시구구명']  = data['시군구'].apply(lambda x : x.split(' ')[1])

data = data.reset_index()
data.drop(columns='index', inplace=True)

datapv=data.pivot_table(index='index', columns='시구구명', values='시군구', aggfunc='count')
datapv= datapv.fillna(0)

data_ch = pd.concat([data, datapv], axis=1)


data_ch.columns.values[19:]

data_ch.columns.values[19:]].values

data_ch[data_ch.columns.values[19:]]


cols = np.array(['전용면적(㎡)', '층', '년식', '년식제곱'])
cols = np.concatenate([cols, np.array(data_ch.columns.values[19:])])

X = np.array(data_ch[cols].values)
y = np.array(data_ch['거래금액(만원)'].values)
y = y.reshape([y.shape[0],1])


results = sm.OLS(y, X).fit()
print(results.summary())