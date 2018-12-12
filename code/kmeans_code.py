# Q4 k-means clustering
import os
os.system('CLS')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import load_data

def kmeans(X_train, k, init_cluster_indx):
    num_obs = X_train.shape[0]
    d = X_train.shape[1]  # length of feature vector

    #print("Cluster Indices: ", init_cluster_indx)

    cluster_pts = X_train[init_cluster_indx, :]
    # print("Cluster pts shape: ", cluster_pts.shape)
    # print("Initial Cluster means: \n", cluster_pts)
    # print("\n")

    itr_vs_dist = np.empty([0, 2])
    converged = False
    itr1 = 0

    #while not converged:
    while not converged and itr1 < 2:
        cluster_old = cluster_pts
        #print("while loop iteration: ", itr1)

        opt_dist = 0
        cluster_grp = np.empty([0, 2])
        for j in np.arange(num_obs):
            # print("Diff from cluster means: \n", cluster_pts - X_train[j,:])
            dist = np.linalg.norm(cluster_pts - X_train[j,:], axis=1)
            #print("Distance of obs from cluster center: ", np.linalg.norm(cluster_pts - X_train[j,:], axis=1))
            chosen_cluser = np.argmin(dist)
            #print("Smallest distance: ", dist[chosen_cluser])

            opt_dist = opt_dist + dist[chosen_cluser]
            #print("Updated optimal distance: ", opt_dist)

            cluster_grp = np.append(cluster_grp, np.array([j, chosen_cluser]).reshape(1, -1), axis=0)
            output = "Observation " + str(j) + " belongs to cluster " + str(chosen_cluser) + "\n"
            # print(output)

        #print("Cluster group:\n", cluster_grp)
        #print("Total Optimal Distance:", opt_dist)

        itr_vs_dist = np.append(itr_vs_dist, np.array([int(itr1+1), opt_dist]).reshape(1, -1), axis=0)
        cluster_means = np.empty([0, d+1])
        #print("shape of Xtrain before entering :", X_train.shape)

        for itr in np.arange(k):
            rows_indx = np.where(cluster_grp[:, 1] == itr)
            rows_indx = list(rows_indx)
            #print("Rows: ", rows_indx)
            temp_xtrain = X_train[rows_indx, :]

            #print("Corresponding elements in X_train: \n", temp_xtrain)
            avg = np.mean(temp_xtrain, axis=1, keepdims=True).reshape(1, -1)
            #print("Clust avg: ", avg.flatten())
            avg = avg.flatten()
            avg = np.insert(avg, 0, itr)
            #print("Means of each cluster is computed here: ", avg)
            cluster_means = np.append(cluster_means, avg.reshape(1,-1), axis=0)

        # print("Old Cluster Means:\n", cluster_old)
        # print("Current Cluster Means:\n", cluster_means)
        # print("Diif in means:\n", cluster_old - cluster_means[:, 1:])
        cluster_pts = cluster_means[:, 1:]
        itr1 += 1
        #print(np.allclose(cluster_old, cluster_pts, 0.01))
        converged = np.allclose(cluster_old, cluster_pts, 0.1)

    return opt_dist, cluster_pts, cluster_grp


def kmeans_plus(X_train, k):
    first_cluster_indx = np.random.choice(X_train.shape[0], 1, replace=False)
    first_cluster = X_train[first_cluster_indx, :]
    num_obs = X_train.shape[0]
    d = X_train.shape[1]  # length of feature vector

    #print("First cluster index: ", first_cluster_indx)

    index_available = np.setdiff1d(np.arange(X_train.shape[0]), first_cluster_indx)
    #print("Indx avilable shape: ", index_available.shape)

    temp_xtrain = X_train[index_available, :]
    #print("Temp x_train shape: ", temp_xtrain.shape)

    dist_prob = np.empty([num_obs-1, 2]) * np.nan
    #print("Shape of dist_prob: ", dist_prob.shape)

    dist = np.linalg.norm(first_cluster - temp_xtrain, axis=1)**2
    #print("Shape of dist: ", dist.shape)
    dist_prob[:, 0] = dist
    #print("sum of distance: ", np.sum(dist_prob[:, 0]))
    dist_prob[:, 1] = dist_prob[:, 0] / np.sum(dist_prob[:, 0])
    #print("Dist prob shape", dist_prob[:,1].shape)

    chosen_clusters = first_cluster_indx
    num_chosen_cluster = 1

    while num_chosen_cluster < k:
        #print("which iteration: ", num_chosen_cluster)
        next_cluster = np.random.choice(index_available, 1, p=dist_prob[:, 1])
        #print("Next cluster chosen: ", next_cluster)
        chosen_clusters = chosen_clusters.flatten()
        chosen_clusters = np.insert(chosen_clusters, 0, next_cluster)

        index_available = np.setdiff1d(np.arange(X_train.shape[0]), chosen_clusters)
        #print("After: ", index_available.shape)
        num_chosen_cluster = num_chosen_cluster + 1

        dist_prob = np.empty([num_obs - num_chosen_cluster, 2]) * np.nan
        temp_xtrain = X_train[index_available, :]
        cluster_pts = X_train[chosen_clusters, :]

        for j in np.arange(temp_xtrain.shape[0]):
            dist = np.linalg.norm(cluster_pts - temp_xtrain[j,:], axis=1)
            #print("Distance of obs from cluster center: ", np.linalg.norm(cluster_pts - X_train[j,:], axis=1))
            chosen_cluser = np.argmin(dist)
            #print("Smallest distance: ", dist[chosen_cluser])
            dist_prob[:, 0] = dist[chosen_cluser]

        dist_prob[:, 1] = dist_prob[:, 0] / np.sum(dist_prob[:, 0])
        #print("dist prob shape after rechecking distance", dist_prob.shape)
        #print("Number of chosen cluster so far: ", num_chosen_cluster)

    #print("Chosen_clusters: ", chosen_clusters)
    return chosen_clusters


# Main program
ret_data, vol_data, comp_list = load_data.read_data()
print("Returns data: \n", ret_data.shape)
print("Volatility data: \n", vol_data.shape)

# Transpose - as we are trying to group firms into different clusters
X = ret_data.T   # For X, rows are firms and columns are returns on different days
#print(X.shape)

########
#X = X.iloc[0:100, 0:100]
#######

firms = X.index
#print("Firms: ", firms)

X = X.values
#print("\nPandas dataframe has been coverted to numpy array: \n", X)

###############################################################################
clust_num_dist = np.empty([0, 2])
max_clust_num = 50


for i in np.arange(max_clust_num):
    cluster_num = i+1
    # init_indx = kmeans_plus(X, cluster_num)  # inputs are matrix X and how many clusters
    # print("Initial chosen (kmeans++) cluster center:\n", init_indx)
    init_indx = np.random.choice(X.shape[0], cluster_num, replace=False)
    inclust_dist, clust_center, cluster_grps = kmeans(X, cluster_num, init_indx)
    print("Optimal istance:\n", inclust_dist)
    #print("Cluster classification: ", cluster_grps)
    clust_num_dist = np.append(clust_num_dist, np.array([i, inclust_dist]).reshape(1, -1), axis=0)


print(clust_num_dist)

plt.plot(np.arange(max_clust_num)+1, clust_num_dist[:, 1])
plt.xlabel('Number of clusters')
plt.ylabel('Aggregate in-cluster distance')
plt.title("Cluster number vs. incluster distance")
plt.show()





# # An "interface" to matplotlib.axes.Axes.hist() method
# n, bins, patches = plt.hist(x=cluster_grps[:, 1], bins='auto', alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Clusters')
# plt.ylabel('Frequency')
# plt.title('Cluster Distribution: Number of clusters = ' + str(cluster_num))
# plt.show()
