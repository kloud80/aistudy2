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




X = np.array(data['전용면적(㎡)'].values)
y = np.array(data['거래금액(만원)'].values)

X = X.reshape([X.shape[0], 1])
y = y.reshape([y.shape[0], 1])

import statsmodels.api as sm
results = sm.OLS(y, sm.add_constant(X)).fit()
print(results.summary())