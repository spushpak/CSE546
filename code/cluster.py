import os
os.system('CLS')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import load_data

np.set_printoptions(precision=5, suppress=True)

# Main program
ret_data, vol_data, comp_list = load_data.read_data()
print("Returns data: \n", ret_data.shape)
print("Volatility data: \n", vol_data.shape)

# Transpose - as we are trying to group firms into different clusters
#X = ret_data.T   # For X, rows are firms and columns are returns on different days
#print(X.shape)

X = ret_data
X = (X - X.mean())/X.std()
#print("Scaled X: \n", X)


########
#X = X.iloc[0:100, 0:100]
#######

firms = X.index
#print("Firms: ", firms)

#X = X.values
#print("\nPandas dataframe has been coverted to numpy array: \n", X)

'''
# generate the linkage matrix
Z = linkage(X, 'ward')

c, coph_dists = cophenet(Z, pdist(X))
print(c)
print("\n")
#print(coph_dists)

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

print(Z.shape)
print(Z)

import scipy.cluster.hierarchy as shc

# plt.figure(figsize=(10, 7))
# plt.title("Dendograms")
# dend = shc.dendrogram(shc.linkage(X, method='ward'))
# plt.show()

##############################
# X = vol_data.T
# plt.figure(figsize=(10, 7))
# plt.title("Dendograms")
# dend = shc.dendrogram(shc.linkage(X, method='ward'))
# plt.show()
#
# from sklearn.cluster import AgglomerativeClustering
# cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
# cluster.fit_predict(X)
# print(cluster.labels_)

emp_corr = X.corr()
#print(emp_corr)
print(emp_corr.shape)

size = 7
fig, ax = plt.subplots(figsize=(size, size))
ax.matshow(emp_corr,cmap=cm.get_cmap('coolwarm'), vmin=0,vmax=1)
plt.xticks(range(len(emp_corr.columns)), emp_corr.columns, rotation='vertical', fontsize=8);
plt.yticks(range(len(emp_corr.columns)), emp_corr.columns, fontsize=8);
#plt.show()

dist = 2*(1 - emp_corr)
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(dist, 'average')
print("shape of Z: ", Z.shape)
print(Z[0])
print(Z)

print("\n")

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import pylab
c, coph_dists = cophenet(Z, pdist(dist))
print(c)
'''

# vol_ret = pd.concat([ret_data.head(1), vol_data.head(1)])
# vol_ret = vol_ret.T
# vol_ret.columns = ['return', 'volatility']
# print(vol_ret)
#
# vol_ret.plot.scatter(x='return', y='volatility', c='DarkBlue')
# plt.show()
