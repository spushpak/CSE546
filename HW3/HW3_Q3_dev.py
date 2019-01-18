import numpy as np
import matplotlib.pyplot as plt


def gen_data(size):
    n = size
    x = np.zeros([n,])
    for i in range(n):
        x[i] = (i/n)

    epsilon= np.random.normal(0, 1, n)
    fx= 4*np.sin(np.pi*x)*np.cos(6*np.pi*x**2)
    y = fx + epsilon
    return x, y, fx

n = 30
X, Y, fx = gen_data(n)
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
fx = fx.reshape(-1,1)

# Polynomial kernel
# K_mat = X@X.T
#print("Shape of Gram Matrix: ", K_mat.shape)

# Split into train and validation set
val_indx = np.random.choice(n,1,replace=False)
#print(val_indx)
val_X = X[val_indx, :]
val_Y = Y[val_indx, :]

train_indx = np.setdiff1d(np.arange(n), val_indx)
#print(train_indx)

train_X = X[train_indx, :]
train_Y = Y[train_indx, :]

'''
# Polynomial kernel
# I tried with d = 3 and d=4 and based on the result chose d=3
d = 3
hyp_par_seq= [10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 10, 100, 1000, 10000]
K_mat = train_X@train_X.T
K_mat = (1 + K_mat)**d

error_mat = []

for hyp_par in hyp_par_seq:
    A = K_mat + hyp_par * np.eye(n-1)
    alpha_hat = np.linalg.solve(A, train_Y)
    #print(alpha_hat.shape)
    w_hat = train_X.T@alpha_hat

    #dot_prod = val_X*train_X
    #print(dot_prod.shape)
    # kernel_wt = 1 + dot_prod
    # val_fx_hat = kernel_wt.T@alpha_hat

    val_fx_hat = w_hat*val_X
    error = val_Y.flatten() - val_fx_hat.flatten()
    error = error**2
    print("Validation error: ", error)
    error_mat.append(error.tolist())
    fx_hat = K_mat@alpha_hat

plt.clf()
plt.plot(hyp_par_seq, error_mat)
plt.title('Polynomial Kernel degree=' + str(d) + ', n=30')
plt.xscale('log')
plt.xlabel("lambda values")
plt.ylabel("Sqrd. validation error")
#plt.show()
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/polydeg3_n30.png')

hyp_par = 10  # Validation error is minimum
A = K_mat + hyp_par * np.eye(n-1)
alpha_hat = np.linalg.solve(A, train_Y)
w_hat = train_X.T@alpha_hat
fx_hat = w_hat*X
print("W_hat: ", w_hat)
print("fx_hat: \n", fx_hat)
print("fx: \n", fx)

plt.clf()
plt.plot(X, fx, label="f(x)")
plt.plot(X, fx_hat, label="f(x) hat")
plt.scatter(X, Y, label="x vs y")
plt.title('n=30')
plt.xlabel(" X")
plt.ylabel("Y/f(x)")
plt.legend(loc="best")
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/xy_fx_fxhat_poly_n30.png')

'''
# RBF kernel
hyp_par_seq= [10**(-3), 10**(-2), 10**(-1), 1, 10, 100, 1000, 10000]

sq_terms = (X**2)
x_i_sqr = sq_terms @ np.ones(n)
x_j_sqr =  np.ones(n) @ sq_terms
X_rbf = (-x_i_sqr -x_j_sqr + 2*x @ x)
gamma = 1/np.sqrt(2)
kernel_rbf = np.exp(gamma*X_rbf)
kernel_rbf_diag = np.eye(n)
error_mat = []

for hyp_par in hyp_par_seq:
    A = K_mat + hyp_par * np.eye(n-1)
    alpha_hat = np.linalg.solve(A, train_Y)
    #print(alpha_hat.shape)

    dot_prod = val_X*train_X
    #print(dot_prod.shape)
    kernel_wt = 1 + dot_prod
    val_fx_hat = kernel_wt.T@alpha_hat
    error = val_Y.flatten() - val_fx_hat.flatten()
    error = error**2
    print("Validation error: ", error)
    error_mat.append(error.tolist())
    fx_hat = K_mat@alpha_hat

plt.clf()
plt.plot(hyp_par_seq, error_mat)
plt.title('Polynomial Kernel degree=' + str(d) + ', n=30')
plt.xscale('log')
plt.xlabel("lambda values")
plt.ylabel("Validation error")
plt.show()
#plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/polydeg3_n30.png')

plt.clf()
plt.plot(X, fx, label="f(x)")
plt.plot(train_X, fx_hat, label="f(x) hat")
plt.scatter(X, Y, label="x vs y")
plt.title('n=30')
plt.xlabel(" X")
plt.ylabel("Y/f(x)")
plt.legend(loc="best")
plt.savefig('C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/xy_fx_fxhat_poly_n30.png')
'''
