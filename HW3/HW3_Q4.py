# Q4 k-means clustering
import os
os.system('CLS')
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

def load_dataset():
    mndata = MNIST('C:\\local\\MNIST_data\\')
    mndata.gz = True
    x_train, labels_train = map(np.array, mndata.load_training())
    x_test, labels_test = map(np.array, mndata.load_testing())
    x_train = x_train/255.0
    x_test = x_test/255.0
    return x_train, labels_train, x_test, labels_test


def kmeans(X_train, k, init_cluster_indx):
    num_obs = X_train.shape[0]
    d = X_train.shape[1]  # length of feature vector

    #print("Cluster Indices: ", init_cluster_indx)

    cluster_pts = X_train[init_cluster_indx, :]
    # print("Cluster pts shape: ", cluster_pts.shape)
    #print("Initial Cluster means: \n", cluster_pts)
    # print("\n")

    itr_vs_dist = np.empty([0, 2])
    converged = False
    itr1 = 0

    #while not converged and itr1<3:
    while not converged:
        cluster_old = cluster_pts
        #print("while loop iteration: ", itr1)

        opt_dist = 0
        cluster_grp = np.empty([0, 2])
        for j in np.arange(num_obs):
            #print("Diff from cluster means: \n", cluster_pts - X_train[0,:])
            dist = np.linalg.norm(cluster_pts - X_train[j,:], axis=1)
            #print("Distance of obs from cluster center: ", np.linalg.norm(cluster_pts - X_train[j,:], axis=1))
            chosen_cluser = np.argmin(dist)
            #print("Smallest distance: ", dist[chosen_cluser])

            opt_dist = opt_dist + dist[chosen_cluser]
            #print("Updated optimal distance: ", opt_dist)

            cluster_grp = np.append(cluster_grp, np.array([j, chosen_cluser]).reshape(1, -1), axis=0)
            # output = "Observation " + str(j) + " belongs to cluster " + str(chosen_cluser) + "\n"
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
        converged = np.allclose(cluster_old, cluster_pts, 0.01)

    return itr_vs_dist, cluster_pts



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

X_train, train_labels, X_test, test_labels = load_dataset()
#print(X_train.shape)

#X_train = np.array([1,2,3,4,5,6,7,8]).reshape(-1,2)
# print(X_train.shape)
# print("X_train:\n", X_train)

'''
###############################################################################
cluster_num = 5
init_indx = np.random.choice(X_train.shape[0], cluster_num, replace=False)
itr_dist, clust_center = kmeans(X_train, cluster_num, init_indx)
print("Iteration and distance:\n", itr_dist)
print("Cluster center matrix shape: ", clust_center.shape)

plt.clf()
plt.plot(itr_dist[:, 0], itr_dist[:, 1])
plt.xlabel("Iteration")
plt.ylabel("Optimal Distance from Cluster")
title_string = "Iteration vs. Distance: No. of clusters k = " + str(cluster_num)
plt.title(title_string)
#plt.show()
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/itr_vs_funcval_k_5.png')

fig = plt.figure()
fig.suptitle("Cluster centers when k = 5")
for i in range(1, 6):
    plt.subplot(2, 3, i)
    X = clust_center[i-1, :].reshape([28, 28])
    plt.gray()
    plt.imshow(X)
#plt.show()
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/clust_center_k_5.png')

###############################################################################
cluster_num = 10
init_indx = np.random.choice(X_train.shape[0], cluster_num, replace=False)
itr_dist, clust_center = kmeans(X_train, cluster_num, init_indx)
print("Iteration and distance:\n", itr_dist)
print("Cluster center matrix shape: ", clust_center.shape)

plt.clf()
plt.plot(itr_dist[:, 0], itr_dist[:, 1])
plt.xlabel("Iteration")
plt.ylabel("Optimal Distance from Cluster")
title_string = "Iteration vs. Distance: No. of clusters k = " + str(cluster_num)
plt.title(title_string)
#plt.show()
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/itr_vs_funcval_k_10.png')

fig = plt.figure()
fig.suptitle("Cluster centers when k = 10")
for i in range(1, 11):
    plt.subplot(3, 4, i)
    X = clust_center[i-1, :].reshape([28, 28])
    plt.gray()
    plt.imshow(X)
#plt.show()
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/clust_center_k_10.png')

###############################################################################
cluster_num = 20
init_indx = np.random.choice(X_train.shape[0], cluster_num, replace=False)
itr_dist, clust_center = kmeans(X_train, cluster_num, init_indx)
print("Iteration and distance:\n", itr_dist)
print("Cluster center matrix shape: ", clust_center.shape)

plt.clf()
plt.plot(itr_dist[:, 0], itr_dist[:, 1])
plt.xlabel("Iteration")
plt.ylabel("Optimal Distance from Cluster")
title_string = "Iteration vs. Distance: No. of clusters k = " + str(cluster_num)
plt.title(title_string)
#plt.show()
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/itr_vs_funcval_k_20.png')

fig = plt.figure()
fig.suptitle("Cluster centers when k = 20")
for i in range(1, 21):
    plt.subplot(4, 5, i)
    X = clust_center[i-1, :].reshape([28, 28])
    plt.gray()
    plt.imshow(X)
#plt.show()
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/clust_center_k_20.png')
'''
##############################################################################
##############################################################################
##############################################################################
# Q4.b
cluster_num = 5
init_indx = kmeans_plus(X_train, cluster_num)
itr_dist, clust_center = kmeans(X_train, cluster_num, init_indx)
print("Iteration and distance:\n", itr_dist)
print("Cluster center matrix shape: ", clust_center.shape)

plt.clf()
plt.plot(itr_dist[:, 0], itr_dist[:, 1])
plt.xlabel("Iteration")
plt.ylabel("Optimal Distance from Cluster")
title_string = "kmeans++: Iteration vs. Distance: No. of clusters k = " + str(cluster_num)
plt.title(title_string)
#plt.show()
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/itr_vs_funcval_kplus_5.png')

fig = plt.figure()
fig.suptitle("Cluster centers when k = 5")
for i in range(1, 6):
    plt.subplot(4, 5, i)
    X = clust_center[i-1, :].reshape([28, 28])
    plt.gray()
    plt.imshow(X)
#plt.show()
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/clust_center_kplus_5.png')



cluster_num = 10
init_indx = kmeans_plus(X_train, cluster_num)
itr_dist, clust_center = kmeans(X_train, cluster_num, init_indx)
print("Iteration and distance:\n", itr_dist)
print("Cluster center matrix shape: ", clust_center.shape)

plt.clf()
plt.plot(itr_dist[:, 0], itr_dist[:, 1])
plt.xlabel("Iteration")
plt.ylabel("Optimal Distance from Cluster")
title_string = "kmeans++: Iteration vs. Distance: No. of clusters k = " + str(cluster_num)
plt.title(title_string)
#plt.show()
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/itr_vs_funcval_kplus_10.png')

fig = plt.figure()
fig.suptitle("Cluster centers when k = 10")
for i in range(1, 11):
    plt.subplot(4, 5, i)
    X = clust_center[i-1, :].reshape([28, 28])
    plt.gray()
    plt.imshow(X)
#plt.show()
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/clust_center_kplus_10.png')



cluster_num = 20
init_indx = kmeans_plus(X_train, cluster_num)
itr_dist, clust_center = kmeans(X_train, cluster_num, init_indx)
print("Iteration and distance:\n", itr_dist)
print("Cluster center matrix shape: ", clust_center.shape)

plt.clf()
plt.plot(itr_dist[:, 0], itr_dist[:, 1])
plt.xlabel("Iteration")
plt.ylabel("Optimal Distance from Cluster")
title_string = "kmeans++: Iteration vs. Distance: No. of clusters k = " + str(cluster_num)
plt.title(title_string)
#plt.show()
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/itr_vs_funcval_kplus_20.png')

fig = plt.figure()
fig.suptitle("Cluster centers when k = 20")
for i in range(1, 21):
    plt.subplot(4, 5, i)
    X = clust_center[i-1, :].reshape([28, 28])
    plt.gray()
    plt.imshow(X)
#plt.show()
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/clust_center_kplus_20.png')
