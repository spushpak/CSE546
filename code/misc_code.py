from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
import numpy as np
from scipy.cluster.vq import kmeans, kmeans2, vq
import pandas as pd
from math import sqrt
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import load_data

ret_data, vol_data, comp_list = load_data.read_data()
#print(ret_data.head())
#print(vol_data.head())

returns = ret_data.mean()*252   # annualized return
vol = vol_data.mean()
#print(returns)
data = np.asarray([np.asarray(returns), np.asarray(vol)]).T
X = data

max_cluster = 20
incr = 1

distortions = []
for k in range(incr, max_cluster):
    k_means = KMeans(n_clusters=k, random_state=0)
    k_means.fit(X)
    distortions.append(k_means.inertia_)

print(distortions)
print([j-i for i, j in zip(distortions[:-1], distortions[1:])])

fig = plt.figure(figsize=(8, 5))
plt.plot(range(incr, max_cluster), distortions)
plt.grid(True)
plt.title('Elbow curve')
plt.xlabel("Number of clusters")
plt.ylabel("Sum of Squared Errors")
#plt.show()

np.random.seed(5)

# computing K-Means with K = 5 (5 clusters)
centroids,_ = kmeans2(data, 5, minit='points')
# assign each sample to a cluster
idx,_ = vq(data,centroids)

# print("Cluster index: ", idx)
# print("Firm names: ", returns.index)

# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om')
plot(centroids[:,0],centroids[:,1],'sk',markersize=8)
plt.xlabel("Returns")
plt.ylabel("Volatility")
#show()

details = [(name,cluster) for name, cluster in zip(returns.index,idx)]
# for detail in details:
#     print(detail)

firms_cluster = np.column_stack((returns.index, idx))
print("Firms and cluster: \n", firms_cluster)

###
indx = np.where(firms_cluster[:, 1] == 0)
print("How many in cluster: ", len(np.array(indx).flatten()))
print("Cluster 1: ", returns.index[indx].tolist())

indx = np.where(firms_cluster[:, 1] == 1)
print("How many in cluster: ", len(np.array(indx).flatten()))
print("Cluster 2: ", returns.index[indx].tolist())

indx = np.where(firms_cluster[:, 1] == 2)
print("How many in cluster: ", len(np.array(indx).flatten()))
print("Cluster 3: ", returns.index[indx].tolist())

indx = np.where(firms_cluster[:, 1] == 3)
print("How many in cluster: ", len(np.array(indx).flatten()))
print("Cluster 4: ", returns.index[indx].tolist())

indx = np.where(firms_cluster[:, 1] == 4)
print("How many in cluster: ", len(np.array(indx).flatten()))
print("Cluster 5: ", returns.index[indx].tolist())

print(comp_list)
comp_list= comp_list.drop(columns=['Security'])
comp_list.columns = ['Sector']
print(comp_list)

firms_cluster = pd.DataFrame(data=firms_cluster[:, 1], index=firms_cluster[:, 0], columns = ['cluster'])
#firms_cluster.columns = ['cluster']
print(firms_cluster)

comb_firms_clust = pd.concat([comp_list, firms_cluster], axis=1)
print(comb_firms_clust.head())

print("\n\n\n")

print(pd.crosstab(comb_firms_clust.Sector, comb_firms_clust.cluster, margins=True))
