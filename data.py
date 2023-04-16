import pandas as pd
from glob import glob
import os

flist = glob('*.csv')

data = pd.read_csv(flist[0], encoding='cp949', sep=',', skiprows=15, dtype='str')