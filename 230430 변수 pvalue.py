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