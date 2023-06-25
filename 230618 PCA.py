import pandas as pd
import geopandas as gpd
""" geopandas로 변경한다. """
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')


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

gdata.columns.values

print(cols.values)

plt.figure(figsize=(35,25))
ax = plt.axes()
ax.axis('off')
ax = gdata.plot(column='직장인구 추정소득', cmap='Reds', ax = ax)
plt.show()



gdata['거주인구수'] = gdata[['20대미만 남성 거주인구수', '20대 남성 거주인구수',
       '30대 남성 거주인구수', '40대 남성 거주인구수', '50대 남성 거주인구수', '60대 남성 거주인구수',
       '70대이상 남성 거주인구수', '20대미만 여성 거주인구수', '20대 여성 거주인구수', '30대 여성 거주인구수',
       '40대 여성 거주인구수', '50대 여성 거주인구수', '60대 여성 거주인구수', '70대이상 여성 거주인구수']].sum(axis=1)


# cdata = gdata[(gdata['거주인구수'] > 20) & (gdata['직장인구수'] > 20)].copy()
# cdata.columns.values
cdata = gdata.copy()
cdata.columns.values

plt.scatter(y = cdata['거주인구수'],  x = cdata['직장인구수'])
plt.show()




from sklearn.decomposition import PCA

X = np.array(cdata[['거주인구수', '직장인구수']].values)

pca = PCA(n_components = 2) # feature 변수 개수가 2개
pca.fit(X)


print(pca.explained_variance_) # 이것은 eigen value를 의미함
pca.components_
PCscore = pca.transform(X)
# PCscore[0:5]


# ret = cdata.apply(lambda x : x['20대미만 남성 거주인구수'] * pca.components_[0,0] + x['60대 여성 거주인구수'] * pca.components_[0,1] - pca.noise_variance_, axis=1)


PCscore.shape

plt.scatter(y = PCscore[:,0], x= PCscore[:,1])
plt.show()

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)


new_pca = np.array(PCscore[:,:2])

PCscore.max(axis=0)
np.where(new_pca == new_pca[:,1].max())

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6)
kmeans.fit(PCscore)
group = kmeans.labels_

gdata['group'] = pd.DataFrame(data=group)

plt.figure(figsize=(35,25))
ax = plt.axes()
ax.axis('off')
ax = gdata.plot(column='group', cmap='Accent', ax = ax)
plt.show()

