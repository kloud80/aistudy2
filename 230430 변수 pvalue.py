import pandas as pd
from glob import glob
import os
import numpy as np

import warnings
warnings.filterwarnings('ignore')

flist = glob('*.csv')

data = pd.read_csv(flist[0], encoding='cp949', sep=',', skiprows=15, dtype='str')

data['거래금액(만원)'] = data['거래금액(만원)'].str.replace(',', '').astype('int')
data['전용면적(㎡)'] = data['전용면적(㎡)'].astype('float')
data['층'] = data['층'].astype('int')
data['건축년도'] = data['건축년도'].astype('int')
data.dtypes


data['거래금액(만원)'].describe()
data['전용면적(㎡)'].describe()
data['층'].describe()
data['건축년도'].describe()


data['년식'] = 2023- data['건축년도']
data['년식제곱'] = data['년식'] **2




X = np.array(data[['전용면적(㎡)', '층', '년식', '년식제곱']].values)
y = np.array(data['거래금액(만원)'].values)

if len(X.shape) == 1 :
    X = X.reshape([X.shape[0], 1])
y = y.reshape([y.shape[0], 1])

import statsmodels.api as sm
results = sm.OLS(y, sm.add_constant(X)).fit()
print(results.summary())


pred_y = X * np.array([1506.7843, 1221.3978, -4269.0432, 109.3591])
pred_y = pred_y.sum(axis=1)
pred_y = pred_y + 2903.4392
pred_y = pred_y.reshape([pred_y.shape[0], 1])



import matplotlib.pyplot as plt

plt.scatter(x= y, y= pred_y)
plt.show()

X[:,2:].shape


data.iloc[:20, [1,3,5,7,9]]

pred_y = X[:,2:] * np.array([-4269.0432, 109.3591])
pred_y = pred_y.sum(axis=1)
# pred_y = pred_y + 2903.4392
pred_y = pred_y.reshape([pred_y.shape[0], 1])

plt.scatter(x=X[:,2], y=pred_y)
plt.show()




print(results.summary())




X2 = np.random.rand(7092,1) * 1
X2 = X2 + X[:,2:3]

X3 = np.random.rand(7092,1) * 1
X3 = X3 + X[:,2:3]

X4 = np.random.rand(7092,1) * 1
X4 = X4 + X[:,2:3]

X2 = np.concatenate([X[:,:2], X2, X3, X4], axis=1)
results = sm.OLS(y, sm.add_constant(X2)).fit()
print(results.summary())

results = sm.OLS(y, sm.add_constant(X2[:, [0,1,3,4]])).fit()
print(results.summary())

results = sm.OLS(y, sm.add_constant(X2[:, [0,1,3]])).fit()
print(results.summary())