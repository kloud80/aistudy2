import pandas as pd
from glob import glob
import os
import numpy as np


import matplotlib.pyplot as plt
import sklearn.metrics
import statsmodels.api as sm

from sklearn.metrics import r2_score

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')



flist = glob('*.csv')

data = pd.read_csv(flist[0], encoding='cp949', sep=',', skiprows=15, dtype='str')


data['거래금액(만원)'] = data['거래금액(만원)'].str.replace(',', '').astype('int')
data['전용면적(㎡)'] = data['전용면적(㎡)'].astype('float')
data['층'] = data['층'].astype('int')
data['건축년도'] = data['건축년도'].astype('int')

data['단가'] = data['거래금액(만원)'] / data['전용면적(㎡)']
data.dtypes



data['년식'] = 2023- data['건축년도']
data['년식제곱'] = data['년식'] **2



X = np.array(data[['전용면적(㎡)', '층', '년식', '년식제곱']].values)
y = np.array(data['단가'].values)
y = y.reshape([y.shape[0],1])

X.shape

plt.scatter(x=X[:,0], y=X[:,1], s=1)
plt.show()

plt.scatter(x=X[:,0], y=X[:,1], c=y, cmap='Reds', s=5)
plt.show()


# k-means clustering 실행
kmeans = KMeans(n_clusters=3)
kmeans.fit(X[:, :2])
group = kmeans.labels_

center_k = np.array(kmeans.cluster_centers_)
center_k.shape
plt.scatter(x=X[:,0], y=X[:,1], s=5, c=kmeans.labels_, cmap='Accent')
plt.scatter(x=center_k[:,0], y=center_k[:,1], s=20, c='black')
plt.show()





cosine = np.array(cosine_similarity(X[:, :2],dense_output=True)[:,0])

cosine = np.reshape(cosine, [cosine.shape[0], 1])

kmeans = KMeans(n_clusters=4)
kmeans.fit(cosine)

plt.scatter(x=X[:,0], y=X[:,1], s=1, c=kmeans.labels_, cmap='Accent')
plt.show()

