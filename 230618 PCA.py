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

pca.components_
print(pca.explained_variance_) # 이것은 eigen value를 의미함
pca.components_

print(pca.explained_variance_ratio_) # 이것은 eigen value를 의미함

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

plt.scatter(y = PCscore[:,0], x= PCscore[:,1], c=group)
plt.show()


gdata['group'] = pd.DataFrame(data=group)

plt.figure(figsize=(35,25))
ax = plt.axes()
ax.axis('off')
ax = gdata.plot(column='group', cmap='Accent', ax = ax)
plt.show()




""""그냥 클러스터링"""

kmeans = KMeans(n_clusters=6)
kmeans.fit(X)
group = kmeans.labels_

plt.scatter(y = X[:,0], x= X[:,1], c=group)
plt.show()


gdata['group'] = pd.DataFrame(data=group)

plt.figure(figsize=(35,25))
ax = plt.axes()
ax.axis('off')
ax = gdata.plot(column='group', cmap='Accent', ax = ax)
plt.show()





""" 전체 데이터 """
cdata.columns.values
cdata = cdata[['20대미만 남성 거주인구수', '20대 남성 거주인구수',
       '30대 남성 거주인구수', '40대 남성 거주인구수', '50대 남성 거주인구수', '60대 남성 거주인구수',
       '70대이상 남성 거주인구수', '20대미만 여성 거주인구수', '20대 여성 거주인구수', '30대 여성 거주인구수',
       '40대 여성 거주인구수', '50대 여성 거주인구수', '60대 여성 거주인구수', '70대이상 여성 거주인구수',
       '직장인구수', '1인 가구수', '1세대 가구수', '2세대 가구수', '3세대 가구수', '생산가능연령인구',
       '급여근로자수', '주거인구 추정소득', '직장인구 추정소득', '아파트 세대수', '연립다세대 세대수',
       '단독다가구 세대수', '오피스텔 세대수', '기타 거처 세대수', '10평미만 주거면적 세대수',
       '10평대 주거면적 세대수', '20평대 주거면적 세대수', '30평대 주거면적 세대수', '40평대 주거면적 세대수',
       '50평대 주거면적 세대수', '60평이상 주거면적 세대수', '아파트 최고 공시가격', '아파트 최저 공시가격',
       '아파트 평균 공시가격', '아파트 중앙값 공시가격', '아파트 최고 실거래가', '아파트 최저 실거래가',
       '아파트 평균 실거래가', '아파트 중앙값 실거래가', '연립다세대 최고 공시가격', '연립다세대 최저 공시가격',
       '연립다세대 평균 공시가격', '연립다세대 중앙값 공시가격', '연립다세대 최고 실거래가',
       '연립다세대 최저 실거래가', '연립다세대 평균 실거래가', '연립다세대 중앙값 실거래가',
       '단독다가구 최고 공시가격', '단독다가구 최저 공시가격', '단독다가구 평균 공시가격',
       '단독다가구 중앙값 공시가격', '단독다가구 최고 실거래가', '단독다가구 최저 실거래가',
       '단독다가구 평균 실거래가', '단독다가구 중앙값 실거래가', '아파트 실거래 건 수', '연립다세대 실거래 건 수',
       '단독다가구 실거래 건 수', '제1금융권 타행 지점 수', '제2금융권 타행 지점 수', '식음료 사업체 수',
       '유통 사업체 수', '숙박 사업체 수', '오락/레저 사업체 수', '교육 사업체 수', '병원/약국 사업체 수',
       '초중고/특수학교 수', '대학교 수', '관공서 수', '전통시장 수', '지하철 라인 수', '버스정류장 수',
       '제1종근린생활시설 면적', '제2종근린생활시설 면적', '업무시설 면적', '문화및집회시설 면적', '숙박시설 면적',
       '종교시설 면적', '위락시설 면적', '판매시설 면적', '공장시설 면적', '운수시설 면적',
       '위험물저장및처리시설 면적', '교육연구시설 면적', '자동차관련시설 면적', '노유자시설 면적', '운동시설 면적',
       '녹지면적',  '거주인구수'']]

from sklearn.decomposition import PCA

X = np.array(cdata[['거주인구수', '직장인구수']].values)

pca = PCA(n_components = 2) # feature 변수 개수가 2개
pca.fit(X)

pca.components_
print(pca.explained_variance_) # 이것은 eigen value를 의미함
pca.components_

print(pca.explained_variance_ratio_) # 이것은 eigen value를 의미함

PCscore = pca.transform(X)
# PCscore[0:5]


# ret = cdata.apply(lambda x : x['20대미만 남성 거주인구수'] * pca.components_[0,0] + x['60대 여성 거주인구수'] * pca.components_[0,1] - pca.noise_variance_, axis=1)


PCscore.shape

plt.scatter(y = PCscore[:,0], x= PCscore[:,1])
plt.show()
