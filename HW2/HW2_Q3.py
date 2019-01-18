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
#n = 20

sigma = 1

W = np.full(d, 0.0)
W[0:k] = 1.*(np.arange(k)+1)/k

epsilon = np.random.normal(0, sigma, n)
X = np.random.randn(n, d)
#cov_mat = np.cov(X, rowvar=False)
Y  = np.dot(X, W) + epsilon
Y = Y.reshape(n, 1)

lam_values = 2*abs(np.dot(X.T, (Y - np.mean(Y))))
lam = np.amax(lam_values)
#print("Starting lambda: ", lam)

wvec = np.full(d, 0.0)
W_est = np.full(d, 5.0)
W_mat = np.empty([0, d])
nzero_coefmat = np.empty([0, 2])
#print("Print empty array shape: ", W_mat.shape)

conv_limit = 0.1
#conv_cond = np.all(abs(np.asarray(W_est) - np.asarray(wvec)) < conv_limit)
conv_cond = np.allclose(W_est, wvec, conv_limit)
#print("conv cond: ", conv_cond)

itr = 0

condition = True

while condition and itr <= 30:
    hyp_par = lam/(1.5**itr)
    print("\nlambda: ", hyp_par)
    print("This is lambda iteration: ", itr)

    conv_cond = False
    counter = 0
    while not conv_cond and counter <= 300:
        counter += 1
        print("\n This is w convergence iteration: ", counter)
        #print("what wvec is used for b: ", wvec)
        b = np.mean(Y - np.dot(X, wvec))
        #print("b: ", b)
        W_est = list(wvec)

        for j in np.arange(d):
            a = 2*sum(X[:, j]**2)
            #print("\na: ", a)
            w_ex = np.copy(W_est)
            w_ex[j] = 0
            #print("w_ex after j=0", w_ex)

            wx_prod = np.dot(X, w_ex).reshape(-1,1)
            # print("wx_prod shape: ", wx_prod.shape)
            # print("wx_prod", wx_prod)
            # print("adding b to wx_prod: ", (b + wx_prod).shape)
            # print("adding b to wx_prod: ", (b + wx_prod))
            # print("Shape of Y: ", Y.shape)

            diff = Y - (b + wx_prod)
            # print("shape of diff: ", diff.shape)
            # print("diff: ", diff)

            c = np.asscalar(2*np.dot(X[:, j], diff))
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

        if np.count_nonzero(wvec) == 0:
            #print("wvec when all w's are zero: ", wvec)
            break
        else:
            #conv_cond = np.all(abs(np.asarray(W_est) - np.asarray(wvec)) < conv_limit)
            conv_cond = np.allclose(W_est, wvec, conv_limit)
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



#W_mat = np.array(W_mat)
print("Shape of final W_mat:\n", W_mat.shape)
print("Non zero coef num:\n", nzero_coefmat)
print("Lambda:\n", nzero_coefmat[:, 0])

print("How many times while loop for lambda has run: ", itr)
#print("W_est: ", W_est)
#print("Final W: ", wvec)
print("Final W_mat for all lambda:\n", W_mat)

plt.plot(nzero_coefmat[:, 0], nzero_coefmat[:, 1], label="Non-zero coef and lambda")
plt.xlabel("Lambda")
plt.ylabel("Number of non-zero coefs")
plt.show()
#plt.savefig('./HW2/nonzero_coef.png')

fdr_tpr = np.empty([0, 3])
for m in np.arange(W_mat.shape[0]):
    incorr_nonzero = np.count_nonzero(W_mat[m, 100:])/(np.count_nonzero(W_mat[m,] + 0.0000001))
    corr_nonzero = np.count_nonzero(W_mat[m, 0:99])/100
    fdr_tpr = np.append(fdr_tpr, np.array([W_mat[m, 0], incorr_nonzero, corr_nonzero]).reshape(1, 3), axis=0)

print("FDR and TPR: \n", fdr_tpr)

plt.plot(fdr_tpr[:, 1], fdr_tpr[:, 2], label="FDR vs. TPR")
plt.xlabel("FDR")
plt.ylabel("TPR")
plt.show()
#plt.savefig('./HW2/fdr_tpr.png')

# print("nonzero elemnt index:\n", np.nonzero(W_mat))
# print("nonzero elemnt index after 100:\n", np.nonzero(W_mat[:, 100:]))
# print("nonzero elemnt:\n", W_mat[np.nonzero(W_mat)])
# print("non", np.nonzero(W_mat[:, 100:999]))
# print("rows", len(np.nonzero(W_mat)[0]))
# print("cols", len(np.nonzero(W_mat)[1]))

# F_mat = np.array([[1,0,2,4], [0,1,0,5],[3,4,0,0]])
# print("Fmat: \n", F_mat)
#print("FDR \n", np.count_nonzero(W_mat[:, 100:]))
# print("FMat \n", np.count_nonzero(F_mat[:,2:]))

#print("Current working directory: ", os.getcwd())
