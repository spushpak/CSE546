import os
os.system('CLS')

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def gen_data(n):
    # Generate data
    np.random.seed(1)
    k = np.array([1, 2, 3, 4])
    K = np.tile(k, (n, 1))

    x = np.arange(n)/(n-1)
    x = x.reshape(-1,1)

    X = (x >= k/5)
    #print(X)

    fx = np.sum(X, axis=1)
    epsilon = np.random.normal(size=n)
    y = fx + epsilon
    y[24]=0

    x = x.flatten()

    return x, y, fx

# Using the code example of http://www.cvxpy.org/
def loss_fn(Kmat, Ymat, alpha_coef):
    return cp.pnorm(cp.matmul(Kmat, alpha_coef) - Ymat, p=2)**2

def regularizer(alpha_coef, Kmat):
    return cp.quad_form(alpha_coef, Kmat)

def tv_regularizer(alpha_coef, Kmat, Dmat):
    return cp.norm(Dmat@Kmat@alpha_coef, 1)

def objective_fn(Kmat, Ymat, alpha_coef, lambd):
    return loss_fn(Kmat, Ymat, alpha_coef) + lambd * regularizer(alpha_coef, Kmat)

def huber_loss_fn(Kmat, Ymat, alpha_coef):
    return cp.sum(cp.huber(Kmat.T*alpha_coef - Ymat, 1))

def huber_obj_fn(Kmat, Ymat, alpha_coef, lambd):
    return huber_loss_fn(Kmat, Ymat, alpha_coef) + lambd*regularizer(alpha_coef, Kmat)

def tv_obj_fn(Kmat, Ymat, Dmat, alpha_coef, lambd1, lambd2):
    return loss_fn(Kmat, Ymat, alpha_coef) + lambd1*tv_regularizer(alpha_coef, Kmat, Dmat) + lambd2*regularizer(alpha_coef, Kmat)

def quad_obj_fn(Kmat, Ymat, alpha_coef, lambd):
    return loss_fn(Kmat, Ymat, alpha_coef) + lambd*regularizer(alpha_coef, Kmat)


def mse(Kmat, Ymat, alpha_coef):
    return (1.0 / Kmat.shape[0]) * loss_fn(Kmat, Ymat, alpha_coef).value


def make_kernel(x, gamma):
    x1 = x.flatten()
    #print(x1)
    x2 = x.reshape(-1, 1)
    #print(x2)
    xdiff = x2 - x1
    norm_diff = np.sqrt(xdiff**2)
    #print("\n")
    # print("Xdiff: \n", xdiff)

    K = np.exp(- gamma*norm_diff)
    return K

def plot_train_test_errors(train_errors, loocv_errors, lambd_values, type_reg):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, loocv_errors, label="Cross-Val error")
    plt.xscale("log")
    plt.legend(loc="best")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (MSE): " + type_reg)
    plt.show()

# Main program
num_obs = 50
x, y, fx = gen_data(num_obs)
print("Shape of x, y, fx: \n", x.shape, y.shape, fx.shape)
# print(fx)


# Split into train and validation set
#val_indx = np.random.choice(num_obs, 1, replace=False)
#print("Validation obs index: ", val_indx)

val_indx = 49
val_x = x[val_indx]
val_fx = fx[val_indx]
val_y = y[val_indx]

train_indx = np.setdiff1d(np.arange(num_obs), val_indx)
# print("Training obs indices: \n", train_indx)
# print("How many train indices: ", len(train_indx))

train_x = x[train_indx]
train_y = y[train_indx]

# Make kernel matrix based on the training set
ls_gamma = 1/np.sqrt(2)
# ls_gamma = 0.5
K = make_kernel(x, ls_gamma)
# print("K shape: ", K.shape)
# print("K: \n", K)

K_train = K[:len(train_indx), :len(train_indx)]
# print("K_train shape: ", K_train.shape)

K_val = K[:len(train_indx), len(train_indx): ]
# print("K_val shape: ", K_val.shape)
# print("K val: \n", K_val)


# Q2a - Least square
nn = num_obs-1
alpha = cp.Variable(nn)
lambd = cp.Parameter(nonneg=True)
lambd_values = np.logspace(-10, 3, 50)
problem = cp.Problem(cp.Minimize(objective_fn(K_train, train_y, alpha, lambd)))

train_errors = []
loocv_errors = []
alpha_values = []

for v in lambd_values:
    lambd.value = v
    problem.solve()
    train_errors.append(mse(K_train, train_y, alpha))
    # val_err = (np.dot(kx_xv.flatten(), alpha.value) - val_y)**2
    # print("Val error: ", val_err)
    # loocv_errors.append(val_err)
    loocv_errors.append(mse(K_val.T, val_y, alpha))
    alpha_values.append(alpha.value)

#plot_train_test_errors(train_errors, loocv_errors, lambd_values, "Least Square")
#print(loocv_errors)

# Find out whcih lambda gives smallest CV error_mat
#print(np.argmin(loocv_errors))
opt_lambd = lambd_values[np.argmin(loocv_errors)]
print("Least square optimal lambda: ", opt_lambd)

# Re-run the optimization with optimal lamba value
lambd.value = opt_lambd
problem.solve()
alpha_hat = alpha.value
fx_hat = K_train@alpha_hat
#print("LS fx hat: \n", fx_hat)
# print('alpha hat: ', alpha.value)

# To be used for plotting
# order = np.argsort(x_train)
# xsorted = x[order]
# fx_sorted = fx[order]
# ysorted = y[order]
# fx_hat = fx_hat[order]

# plt.clf()
# plt.plot(xsorted, fx_sorted, label="f(x)")
# plt.plot(xsorted, fx_hat, label="f(x) hat")
# plt.scatter(xsorted, ysorted, label="x vs y")
# plt.title("Least Square")
# plt.xlabel("X")
# plt.ylabel("f(x)")
# plt.legend(loc="best")
# plt.show()


plt.clf()
plt.plot(x, fx, label="f(x)")
plt.plot(train_x, fx_hat, label="f(x) hat")
plt.scatter(x, y, label="x vs y")
plt.title("Least Square")
plt.xlabel("X")
plt.ylabel("f(x)")
plt.legend(loc="best")
#plt.show()


################################################################################
################################################################################
# Q2.b - Huber loss
#sum_entries(huber(K.T*alpha - train_y, 1))
nn = num_obs-1
alpha = cp.Variable(nn)
lambd = cp.Parameter(nonneg=True)
lambd_values = np.logspace(-10, 3, 50)
problem = cp.Problem(cp.Minimize(huber_obj_fn(K_train, train_y, alpha, lambd)))

huber_train_errors = []
huber_loocv_errors = []
huber_alpha_values = []

for v in lambd_values:
    lambd.value = v
    problem.solve(solver='ECOS')
    huber_train_errors.append(mse(K_train, train_y, alpha))
    huber_loocv_errors.append(mse(K_val.T, val_y, alpha))
    huber_alpha_values.append(alpha.value)

#plot_train_test_errors(huber_train_errors, huber_loocv_errors, lambd_values, "Huber Loss")
#print(huber_loocv_errors)

# Find out whcih lambda gives smallest CV error_mat
opt_lambd = lambd_values[np.argmin(huber_loocv_errors)]
print("Huber optimal lambda: ", opt_lambd)

# Re-run the optimization with optimal lamba value
lambd.value = opt_lambd
problem.solve()
huber_alpha_hat = alpha.value
huber_fx_hat = K_train@huber_alpha_hat
#print("Huber fx hat: \n", huber_fx_hat)

# To be used for plotting
# huber_fx_hat = huber_fx_hat[order]

plt.clf()
plt.plot(x, fx, label="f(x)")
plt.plot(train_x, huber_fx_hat, label="f(x) hat")
plt.scatter(x, y, label="x vs y")
plt.title("Huber Loss")
plt.xlabel("X")
plt.ylabel("f(x)")
plt.legend(loc="best")
#plt.show()


#############################################################################
# Q2c
n=50
D = np.zeros((n-2,n-1))

for i in np.arange(n-2):
    for j in np.arange(n-1):
        if i==j:
            D[i, j] = 1
        elif j == i+1:
            D[i, j] = -1
        else:
            D[i, j] = 0

print("D matrix: \n", D)


nn = num_obs-1
alpha = cp.Variable(nn)
lambd1 = cp.Parameter(nonneg=True)
lambd2 = cp.Parameter(nonneg=True)
lambd1_values = np.logspace(-5, 3, 50)
lambd2_values = np.logspace(-5, 3, 50)
problem = cp.Problem(cp.Minimize(tv_obj_fn(K_train, train_y, D, alpha, lambd1, lambd2)))

tv_train_errors2 = []
tv_loocv_errors2 = []
tv_alpha_values2 = []

for v1 in lambd1_values:
    tv_train_errors = []
    tv_loocv_errors = []
    tv_alpha_values = []

    lambd1.value = v1

    for v2 in lambd2_values:
        lambd2.value = v2
        problem.solve()
        tv_train_errors.append(mse(K_train, train_y, alpha))
        tv_loocv_errors.append(mse(K_val.T, val_y, alpha))
        tv_alpha_values.append(alpha.value)
        #print(alpha.value)
    tv_train_errors2.append(tv_train_errors)
    tv_loocv_errors2.append(tv_loocv_errors)
    tv_alpha_values2.append(tv_alpha_values)

tv_train_errors2 = np.array(tv_train_errors2)
tv_loocv_errors2 = np.array(tv_loocv_errors2)

#print(tv_train_errors2.shape)
#print(tv_train_errors2)
print("\n\n\n")
print(tv_loocv_errors2.shape)
print(tv_loocv_errors2)

# from mpl_toolkits import mplot3d
# from mpl_toolkits.mplot3d.axes3d import Axes3D
# XX,YY = np.meshgrid(lambd1_values, lambd2_values)
# Z1 = tv_train_errors
# Z2 = tv_loocv_errors
# fig = plt.figure()
# ax = plt.axes(projection='3d')
#
# ax.plot_surface(XX, YY, Z1, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# ax.set_title('surface');


# Find out whcih lambda gives smallest CV error_mat
min_row, min_col = np.where(tv_loocv_errors2 == np.amin(tv_loocv_errors2))
print(min_row, min_col)
opt_lambd1 = np.asscalar(lambd1_values[min_row])
opt_lambd2 = np.asscalar(lambd2_values[min_col])
print(opt_lambd1)
print(opt_lambd2)

# Re-run the optimization with optimal lamba values
lambd1.value = opt_lambd1
lambd2.value = opt_lambd2
problem.solve()
tv_alpha_hat = alpha.value
tv_fx_hat = K_train@tv_alpha_hat
print("TV fx hat: \n", tv_fx_hat)

plt.clf()
plt.plot(x, fx, label="f(x)")
plt.plot(train_x, tv_fx_hat, label="f(x) hat")
plt.scatter(x, y, label="x vs y")
plt.title("Total Variation")
plt.xlabel("X")
plt.ylabel("f(x)")
plt.legend(loc="best")
plt.show()


##############################################################################
# Q2d
#from qcqp import *

nn = num_obs-1
alpha = cp.Variable(nn)
lambd = cp.Parameter(nonneg=True)
lambd_values = np.logspace(-5, 3, 50)
constraints = [0 <= D@K_train@alpha]
problem = cp.Problem(cp.Minimize(quad_obj_fn(K_train, train_y, alpha, lambd)), constraints)

# Create a QCQP handler.
#qcqp = QCQP(problem)

quad_train_errors = []
quad_loocv_errors = []
quad_alpha_values = []

for v in lambd_values:
    lambd.value = v
    problem.solve()
    quad_train_errors.append(mse(K_train, train_y, alpha))
    quad_loocv_errors.append(mse(K_val.T, val_y, alpha))
    quad_alpha_values.append(alpha.value)
    #print(constraints[0].dual_value)

plot_train_test_errors(quad_train_errors, quad_loocv_errors, lambd_values, "Quadratic Program")
print(quad_loocv_errors)

# Find out whcih lambda gives smallest CV error_mat
opt_lambd = lambd_values[np.argmin(quad_loocv_errors)]
#print("Quadratic optimal lambda: ", opt_lambd)

# Re-run the optimization with optimal lamba value
lambd.value = opt_lambd
#lambd.value = 10
problem.solve()
quad_alpha_hat = alpha.value
print("Quad alpha hat: \n", quad_alpha_hat)
quad_fx_hat = K_train@quad_alpha_hat
print("QP fx hat: \n", quad_fx_hat)

plt.clf()
plt.plot(x, fx, label="f(x)")
plt.plot(train_x, quad_fx_hat, label="f(x) hat")
plt.scatter(x, y, label="x vs y")
plt.title("Quadratic Program")
plt.xlabel("X")
plt.ylabel("f(x)")
plt.legend(loc="best")
plt.show()
