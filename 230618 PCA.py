import pandas as pd
import geopandas as gpd
""" geopandas로 변경한다. """
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import math


plt.rc('font', family='gulim') # For Windows
print(plt.rcParams['font.family'])

data = pd.read_csv('data/grid_base_data_v2_202302_seoul', sep='|', dtype='str')

data[["경도최솟값","위도최솟값","경도최댓값","위도최댓값"]] = data[["경도최솟값","위도최솟값","경도최댓값","위도최댓값"]].astype('float')

data['geom'] = data[["경도최솟값","위도최솟값","경도최댓값","위도최댓값"]].apply(lambda x :
                                                       Polygon(zip(x[['경도최솟값', '경도최댓값', '경도최댓값', '경도최솟값', '경도최솟값']].values,
                                                                   x[['위도최댓값', '위도최댓값', '위도최솟값', '위도최솟값', '위도최댓값']].values)), axis=1)


gdata = gpd.GeoDataFrame(data, geometry=data['geom'], crs="EPSG:4326")
gdata = gdata.to_crs('EPSG:5174')

cols = gdata.columns
gdata[cols[14:-4]] = gdata[cols[14:-4]].astype('float')



print(cols.values)

plt.figure(figsize=(35,25))
ax = plt.axes()
ax.axis('off')
ax = gdata.plot(column='직장인구 추정소득', cmap='Reds', ax = ax)
plt.show()



cdata = gdata[gdata['20대미만 남성 거주인구수'] > 20].copy()


cdata = cdata[['20대미만 남성 거주인구수','60대 여성 거주인구수']].copy()

plt.scatter(y = cdata['20대미만 남성 거주인구수'],  x = cdata['60대 여성 거주인구수'])
plt.show()




