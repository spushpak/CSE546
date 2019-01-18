# Q3
import numpy as np
import matplotlib.pyplot as plt
import os
os.system('CLS')

k = 100
#k = 5
non_zero_coef = k
d = 1000
#d = 10
n = 500
#n = 50

sigma = 1

W = np.full(d, 0.0)
W[0:k] = 1.*(np.arange(k)+1)/k

epsilon = np.random.normal(0, sigma, n)
X = np.random.randn(n, d)
#cov_mat = np.cov(X, rowvar=False)
Y  = np.dot(X, W) + epsilon
Y = Y.reshape(n, 1)
#print("Shape of Y: ", Y.shape)

mask_idx_mat = np.full((n, d), False)

lam_values = 2*abs(np.dot(X.T, (Y - np.mean(Y))))
lam = np.amax(lam_values)
#print("Starting lambda: ", lam)

wvec = np.full(d, 0.0)
W_est = np.full(d, 5.0)
W_mat = np.empty([0, d])
nzero_coefmat = np.empty([0, 2])
#print("Print empty array shape: ", W_mat.shape)

conv_limit = 0.5
conv_cond = np.all(abs(np.asarray(W_est) - np.asarray(wvec)) < conv_limit)

#print(abs(np.asarray(W_est) - np.asarray(wvec)) < conv_limit)
#print("conv cond: ", conv_cond)

itr = 0
max_iter = 10

condition = True
counter = 0

while condition:
#while itr <= 5:
    hyp_par = lam/(1.5**itr)
    print("\n\nlambda: ", hyp_par)

    conv_cond = False
    while not conv_cond:
        counter += 1
        print("\nIteration: ", counter)
        #print("what wvec is used for b: ", wvec)
        b = np.mean(Y - np.dot(X, wvec))
        #print("b: ", b)
        W_est = list(wvec)

        for j in np.arange(d):
            a = 2*sum(X[:, j]**2)
            #print("\na: ", a)
            mask_idx_mat[:, j] = True
            X_masked = np.ma.array(X, mask = mask_idx_mat)
            #print("Shape of X_masked: ", X_masked.shape)
            W_masked = np.ma.array(W_est, mask = mask_idx_mat[0, :])
            #print("Shape of W_masked: ", W_masked.shape)
            wx_mask_prod = np.ma.dot(X_masked, W_masked).reshape(-1,1)
            #print("wx_mask_prod shape: ", wx_mask_prod.shape)
            #print("adding b to wx_mask_prod: ", (b + wx_mask_prod).shape)
            #print("Shape of Y: ", Y.shape)

            diff = Y - (b + wx_mask_prod)
            #print("shape of diff: ", diff.shape)
            #c = np.asscalar(2*np.dot(X[:, k], diff))
            c = 2*np.dot(X[:, j], diff)
            #print("c: ", c)
            if c < - hyp_par:
                w = 1.*((c + hyp_par) / a)
            elif - hyp_par <= c <= hyp_par:
                w = 0.0
            else:
                w = 1.*((c - hyp_par) / a)
            #print("w after lambda comparison: ", w)

            wvec[j] = w
            mask_idx_mat[:, j] = False

        if np.count_nonzero(wvec) == 0:
            #print("wvec when all w's are zero: ", wvec)
            break
        else:
            conv_cond = np.all(abs(np.asarray(W_est) - np.asarray(wvec)) < conv_limit)
            # print("what is previous iteratio's wvec", W_est)
            # print("wvec when not all w's are zero: ", wvec)
            # print("diff of these two vectors: ", abs(np.asarray(W_est) - np.asarray(wvec)))
            # print("Boolean values: ", abs(np.asarray(W_est) - np.asarray(wvec)) < conv_limit)

        print("Convergence has been achieved: ", conv_cond)


    # Storing the optimal w for particular lambda
    #print("Either all w zero or convergence takes me here")
    #print("Wmat before appending new row: ", W_mat)
    #print(wvec.shape)
    W_mat = np.append(W_mat, wvec.reshape(1, d), axis=0)
    #print("Printing Wmat for each lambda: ", W_mat)
    #print("How many non zero: ", np.count_nonzero(wvec))
    nzero_coefmat = np.append(nzero_coefmat, np.array([hyp_par, np.count_nonzero(wvec)]).reshape(1, 2), axis=0)

    condition = np.count_nonzero(wvec) <= d
    #print("what is lambda loop condition: ", condition)
    itr = itr + 1
    #print("what lambda was used: ", hyp_par)
    #print("this lambda itr: ", itr)


#W_mat = np.array(W_mat)
print("Shape of final W_mat:\n", W_mat.shape)
print("Non zero coef num:\n", nzero_coefmat)
print("lambda inverse:\n", 1/nzero_coefmat[:, 0])

#print("How many times while loop has run: ", num_loops)
#print("W_est: ", W_est)
#print("Final W: ", wvec)
print("Final W_mat for all lambda:\n", W_mat)

plt.plot(1/nzero_coefmat[:, 0], nzero_coefmat[:, 1], label="Non-zero coef and lambda")
plt.xlabel("lambda inverse")
plt.ylabel("Number of non-zero coefs")
#plt.show()
plt.savefig('./HW2/nonzero_coef.png')

#print("Current working directory: ", os.getcwd())
