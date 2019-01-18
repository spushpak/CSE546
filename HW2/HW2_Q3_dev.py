# Q3
import numpy as np
import os
os.system('CLS')


#k = 100
k = 5
#d = 1000
d = 10
#n = 500
n = 50

sigma = 1

W = np.full(d, 0.0)


# print(np.arange(k)+1)
# print(1.*(np.arange(k)+1)/k)

W[0:k] = 1.*(np.arange(k)+1)/k
#print(W)
#print(W.shape)

#for i in np.arange(n):
epsilon = np.random.normal(0, sigma, n)
X = np.random.randn(n, d)
#print(X.shape)
#print(np.mean(X, axis=1))

cov_mat = np.cov(X, rowvar=False)
# print(cov_mat.shape)
# print(np.diag(cov_mat))
# print(cov_mat[0, :])

Y  = np.dot(X, W)
#print(Y)
#print(Y.shape)
#print("Mean of Y: ", np.mean(Y))

mask_idx_mat = np.full((n, d), False)

lam_values = 2*abs(np.dot(X.T, (Y - np.mean(Y))))
lam = np.amax(lam_values)
print("Starting lambda: ", lam)

# num_lam = int(np.log(lam) / np.log(1.5))
# print("How many terms: ", num_lam)
# lam_seq = [lam * (1/1.5) ** (p - 1) for p in range(1, num_lam + 1)]
# print("Computed lambda seq: ", lam_seq)

##############################
# Coordinate Descent Algorithm
wvec = np.full(d, 0.0)
W_est = np.full(d, 5.0)
W_mat = []


conv_limit = 0.01
conv_cond = np.any(abs(W_est - wvec)) > conv_limit
#print(conv_cond)

itr = 0
max_iter = 10

condition = True
while condition:
    hyp_par = lam/(1.5)**itr
    print("What lambda: ", hyp_par)

    while conv_cond:
        #print("wvec before entering for loop of k: ", wvec)
        #print("W_est before entering for loop of k: ", W_est)
        #b=0.0
        b = np.mean(Y - np.dot(X, wvec))
        #print("b: ", b)

        W_est = list(wvec)
        #print("W_est after updation: ", W_est)

        for k in np.arange(d):
            #print("Inside for loop...which k", k)
            a = 2*sum(X[:, k]**2)
            #print("a: ", a)

            mask_idx_mat[:, k] = True
            X_masked = np.ma.array(X, mask = mask_idx_mat)
            W_masked = np.ma.array(W_est, mask = mask_idx_mat[0, :])
            wx_mask_prod = np.ma.dot(X_masked, W_masked)

            diff = Y - (b + wx_mask_prod)
            #print(diff)
            c = 2*np.dot(X[:, k], diff)
            #print("c: ", c)

            if c < - hyp_par:
                w = 1.*((c + hyp_par) / a)
            elif - hyp_par <= c <= hyp_par:
                w = 0.0
            else:
                w = 1.*((c - hyp_par) / a)
            wvec[k] = w
            mask_idx_mat[:, k] = False
            #print("Last W_est: ", W_est)
            #print("w: ", w)

        # print("New wvec after exiting for loop for k: ", wvec)
        # print("Old W_est after foor loop: ", W_est)
        #print("Abs diff: ", abs(np.asarray(W_est) - np.asarray(wvec)))

        conv_cond = np.any(abs(np.asarray(W_est) - np.asarray(wvec))) > conv_limit
        #print("Test whether true: ", abs(W_est - wvec) > conv_limit)
        #print("Conv criteria after for loop: ", np.any(abs(W_est - wvec) > conv_limit))

    # Storing the optimal w for particular lambda
    W_mat.append(wvec)
    print("How many non zero: ", np.count_nonzero(wvec))
    condition = np.count_nonzero(wvec) != d
    itr = itr + 1



W_mat = np.array(W_mat)
print("Shape of final W_mat: ", W_mat.shape)

#print("How many times while loop has run: ", num_loops)
#print("W_est: ", W_est)
print("Final W: ", wvec)
print("W for all lambda:", W_mat)
