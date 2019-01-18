import numpy as np
import matplotlib.pyplot as plt


def gen_data(size):
    n = size
    x = np.zeros([n,])
    y = np.zeros([n,])
    fx = np.zeros([n,])

    for i in range(n):
        x[i] = np.random.uniform()
        fx[i] = 4*np.sin(np.pi*x[i])*np.cos(6*np.pi*(x[i]**2))
        epsilon = np.random.normal()
        y[i] = fx[i] + epsilon

    return x, y, fx

n = 30
X, Y, fx = gen_data(n)
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
fx = fx.reshape(-1,1)


'''
###############################################################################
############  Doing Kernel Regression on the entire data ######################
###############################################################################
# Polynomial kernel
print("Shape of X, Y, fx: ", X.shape, Y.shape, fx.shape)

K_mat = X@X.T
print("Shape of Gram Matrix: ", K_mat.shape)

d = 1
K_mat = (1 + K_mat)**d
error_mat = []

hyp_par = 1

#for hyp_par in hyp_par_seq:
A = K_mat + hyp_par * np.eye(n)
alpha_hat = np.linalg.solve(A, Y)
# print("Shape of alpha: ", alpha_hat.shape)
# print("alpha:\n", alpha_hat)
print("\n")
fx_hat = K_mat@alpha_hat
#print("f_hat from alpha*K:\n", fx_hat)
f_fhat = np.column_stack((fx, fx_hat))
print("f and fhat:\n", f_fhat)

temp = X.flatten()
#print("Falttened X: ", temp)
order = np.argsort(temp)
#print(order)
xsorted = X[order]
#print("Sorted X:\n", xsorted)
ysorted = Y[order]
fx_sorted = fx[order]
fx_hat_sorted = fx_hat[order]


plt.clf()
plt.plot(xsorted, fx_sorted, label="f(x)")
plt.plot(xsorted, fx_hat_sorted, label="f(x) hat")
plt.scatter(xsorted, ysorted, label="x vs y")
title_string = "Polynomial Kernel degree=" + str(d) + ", n=" + str(n)
plt.title(title_string)
plt.xlabel(" X")
plt.ylabel("Y/f(x)")
plt.legend(loc="best")
plt.show()

'''

# Split into train and validation set
val_indx = np.random.choice(n,1,replace=False)
#print(val_indx)
val_X = X[val_indx, :]
val_fx = fx[val_indx, :]
val_Y = Y[val_indx, :]

train_indx = np.setdiff1d(np.arange(n), val_indx)
#print(train_indx)

train_X = X[train_indx, :]
train_Y = Y[train_indx, :]

# Polynomial kernel
# I tried with d = 3 and d=4 and based on the result chose d=3
d = 3
hyp_par_seq= [10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 10, 100, 1000, 10000]
K_mat = train_X@train_X.T
K_mat = (1 + K_mat)**d

error_mat = []
dot_prod = val_X*train_X
kernel_wt = 1 + dot_prod

for hyp_par in hyp_par_seq:
    A = K_mat + hyp_par * np.eye(n-1)
    alpha_hat = np.linalg.solve(A, train_Y)
    #print(alpha_hat.shape)
    val_fx_hat = kernel_wt.T@alpha_hat
    print("prediction from fx hat: ", val_fx_hat)

    error = val_fx.flatten() - val_fx_hat.flatten()
    print("Validation error: ", error)
    error = error**2
    error_mat.append(error.tolist())
    fx_hat = K_mat@alpha_hat

plt.clf()
plt.plot(hyp_par_seq, error_mat)
plt.title('Polynomial Kernel degree=' + str(d) + ', n=' + str(n))
plt.xscale('log')
plt.xlabel("lambda values")
plt.ylabel("Sqrd. validation error")
plt.show()
#plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/polydeg3_n300.png')


hyp_par = 10  # Validation error is minimum
K_mat = X@X.T
K_mat = (1 + K_mat)**d
A = K_mat + hyp_par * np.eye(n)
alpha_hat = np.linalg.solve(A, Y)
print("\n")
fx_hat = K_mat@alpha_hat
f_fhat = np.column_stack((fx, fx_hat))
print("f and fhat:\n", f_fhat)

temp = X.flatten()
order = np.argsort(temp)
xsorted = X[order]
ysorted = Y[order]
fx_sorted = fx[order]
fx_hat_sorted = fx_hat[order]

plt.clf()
plt.plot(xsorted, fx_sorted, label="f(x)")
plt.plot(xsorted, fx_hat_sorted, label="f(x) hat")
plt.scatter(xsorted, ysorted, label="x vs y")
title_string = "Polynomial Kernel degree=" + str(d) + ", n=" + str(n)
plt.title(title_string)
plt.xlabel(" X")
plt.ylabel("Y/f(x)")
plt.legend(loc="best")
plt.show()
#plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/xy_fx_fxhat_poly_n300.png')

'''
# RBF kernel
hyp_par_seq= [10**(-3), 10**(-2), 10**(-1), 1, 10, 100, 1000, 10000]

sq_terms = (train_X**2)
print(np.ones(n-1).shape)
x_i_sqr = sq_terms @ np.ones(n-1).reshape(-1,1).T
x_j_sqr =  np.ones(n-1) @ sq_terms
X_rbf = (-x_i_sqr -x_j_sqr + 2*train_X @ train_X.T)
gamma = 1/np.sqrt(2)
K_rbf = np.exp(gamma*X_rbf)
error_mat = []

for hyp_par in hyp_par_seq:
    A = K_rbf + hyp_par * np.eye(n-1)
    alpha_hat = np.linalg.solve(A, train_Y)
    print(alpha_hat.shape)

    w_hat = train_X.T@alpha_hat
    val_fx_hat = w_hat*val_X
    error = val_Y.flatten() - val_fx_hat.flatten()
    error = error**2
    print("Validation error: ", error)
    error_mat.append(error.tolist())
    fx_hat = K_rbf@alpha_hat

plt.clf()
plt.plot(hyp_par_seq, error_mat)
plt.title('RBF Kernel, n=300')
plt.xscale('log')
plt.xlabel("lambda values")
plt.ylabel("Sqrd. validation error")
#plt.show()
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/rbf_n300.png')

hyp_par = 10000 # this is optimal lambda
A = K_rbf + hyp_par * np.eye(n-1)
alpha_hat = np.linalg.solve(A, train_Y)
w_hat = train_X.T@alpha_hat
fx_hat = K_rbf@alpha_hat

plt.clf()
plt.plot(X, fx, label="f(x)")
plt.plot(train_X, fx_hat, label="f(x) hat")
plt.scatter(X, Y, label="x vs y")
plt.title('RBF kernel n=30')
plt.xlabel(" X")
plt.ylabel("Y/f(x)")
plt.legend(loc="best")
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/xy_fx_fxhat_rbf_n300.png')
'''
