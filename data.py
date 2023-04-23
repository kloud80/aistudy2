import pandas as pd
from glob import glob
import os

import warnings
warnings.filterwarnings('ignore')

flist = glob('*.csv')

data = pd.read_csv(flist[0], encoding='cp949', sep=',', skiprows=15, dtype='str')

data['거래금액(만원)'] = data['거래금액(만원)'].str.replace(',', '').astype('int')
data['전용면적(㎡)'] = data['전용면적(㎡)'].astype('float')


data.dtypes


data['거래금액(만원)'].describe()
data['전용면적(㎡)'].describe()

import numpy as np
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

X = np.array(data['전용면적(㎡)'].values)
y = np.array(data['거래금액(만원)'].values)

X = X.reshape([X.shape[0], 1])
y = y.reshape([y.shape[0], 1])

lm.fit(X, y)

# 거래금액 = 전용면적 * [[1462.40190762]] + [-7466.66266318]

from sklearn.metrics import r2_score

y_pred = lm.predict(X)
r2_score(y, y_pred)

import matplotlib.pyplot as plt

plt.scatter(y, y_pred)
plt.show()

잔차 = y - y_pred

plt.plot(잔차)

plt.hist(잔차, bins=100)
plt.show()


y_평균잔차 = y - y.mean()

plt.hist(y_평균잔차, bins=100)
plt.show()

TERROR = y_평균잔차 * y_평균잔차
TERROR.sum()

EERROR = 잔차 * 잔차
EERROR.sum()

설명해낸에러 = TERROR.sum() - EERROR.sum()

설명해낸에러 / TERROR.sum()


#추가
data.dtypes
data['건축년도'] = data['건축년도'].astype('int')

X = np.array(data[['전용면적(㎡)', '건축년도']].values)
y = np.array(data['거래금액(만원)'].values)

y = y.reshape([y.shape[0], 1])


lm.fit(X, y)
y_pred = lm.predict(X)
r2_score(y, y_pred)



data['년식'] = 2023- data['건축년도']
data['년식제곱'] = data['년식'] **2

data[['건축년도', '년식', '년식제곱']]

data = data.reset_index()
data['index']


data[data['년식'] < 20]


X = np.array(data[['전용면적(㎡)', '건축년도', '년식', '년식제곱']].values)
y = np.array(data['거래금액(만원)'].values)

X.shape

# X = X.reshape([X.shape[0], 1])
y = y.reshape([y.shape[0], 1])

lm = LinearRegression()
type(lm)

lm.fit(X, y)
y_pred = lm.predict(X)
r2_score(y, y_pred)

lm.coef_

y2 = X[:,0] * lm.coef_[0,0] + X[:,1] * lm.coef_[0,1] + X[:,3] * lm.coef_[0,3] + lm.intercept_
y2 = np.array(y2)
y2 = y2.reshape([y2.shape[0], 1])

y2 = y - y2
plt.scatter(data['년식'].values, data['거래금액(만원)'])
plt.scatter(X[:,2], y2, color='red')
plt.show()



plt.scatter(data['전용면적(㎡)'].values, data['년식'].values, data['거래금액(만원)'])
plt.show()

X = np.array(data[['전용면적(㎡)', '년식',  '년식제곱']].values)
y = np.array(data['거래금액(만원)'].values)

X.shape

# X = X.reshape([X.shape[0], 1])
y = y.reshape([y.shape[0], 1])

lm = LinearRegression()
type(lm)

lm.fit(X, y)

lm.coef_




