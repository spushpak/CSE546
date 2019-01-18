import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def loss_fn(X, Y, beta):
    return cp.pnorm(cp.matmul(X, beta) - Y, p=2)**2

def regularizer(beta):
    return cp.pnorm(beta, p=2)**2

def objective_fn_ls(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)

def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value

def plot_train_test_errors(train_errors, test_errors, lambd_values):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (MSE)")
    plt.show()

def huber_fun(X, Y, beta):
    return cp.sum(cp.huber(cp.matmul(X, beta) - Y, 1))

def objective_fn_lub(X, Y, beta, lambd):
    return huber_fun(X, Y, beta) + lambd * regularizer(beta)

def kernel(x, gamma):
    # Creating kernel function:
    x_list = x.tolist()
    x_ker_use = x.reshape(-1, 1)
    diff_mat = x_ker_use- x_list
    diff_sqr= np.sqrt(diff_mat **2)
    #diff_check = np.linalg.norm(diff_mat)
    return  np.exp(- gamma*diff_sqr)
# Problem data.
#m = 30
n = 50
n_train = 49
n_test = 1
np.random.seed(1)
x= np.arange(50)/(n-1)
x_train = x[:n_train]
x_test = x[n_train: ]

x_new=[[1 if x[i]>=k/5 else 0 for k in [1, 2, 3, 4]] for i in range(n)]
x_generate= np.sum(x_new, axis=1)
f_x = np.asanyarray(x_generate)

mu, sigma = 0, 1 # mean and standard deviation
epsilon = np.random.normal(mu, sigma, n)

y= f_x + epsilon
y[24] = 0
y_train = y[:n_train]
y_test = y[n_train: ]

# print("Shape of x, y, fx: ", x.shape, y.shape, f_x.shape)
# print("Shape of train x, y, fx: ", x_train.shape, y_train.shape)
# print("Shape of test x, y, fx: ", x_test.shape, y_test.shape)
#print("f_x: ", f_x)


# Creating kernel function:
x_list = x.tolist()
x_ker_use = x.reshape(-1, 1)
diff_mat = x_ker_use- x_list
diff_sqr= np.sqrt(diff_mat **2)
#diff_check = np.linalg.norm(diff_mat)

#gamma_ls = 1/np.sqrt(2)
gamma_ls =0.5
kernel_rbf_ls = kernel(x, gamma_ls)
print("Full kernel: \n", kernel_rbf_ls)


kernel_rbf_ls_train = kernel_rbf_ls[:n_train, : n_train ]
kernel_rbf_ls_test = kernel_rbf_ls[:n_train, n_train:]

print("Train kernel: ", kernel_rbf_ls_train.shape)
print("Test kernel: ", kernel_rbf_ls_test.shape)
print("Test kernel: ", kernel_rbf_ls_test)

## CVXPY ridge regression:
alpha_ls = cp.Variable(n_train )
lambd_ls = cp.Parameter(nonneg=True)
problem_ls = cp.Problem(cp.Minimize(objective_fn_ls(kernel_rbf_ls_train, y_train, alpha_ls, lambd_ls)))

lambd_values_ls = np.logspace(-5, 3, 50)
train_errors_ls = []
test_errors_ls = []
alpha_values_ls = []
for v in lambd_values_ls:
    lambd_ls.value = v
    problem_ls.solve()
    train_errors_ls.append(mse(kernel_rbf_ls_train, y_train, alpha_ls))
    test_errors_ls.append(mse(kernel_rbf_ls_test.T, y_test, alpha_ls))
    alpha_values_ls.append(alpha_ls.value)

##################################################################
"""
lambd = cp.Parameter(nonneg=True)
alpha = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.sum_squares(y - kernel_rbf*alpha) + lambd *alpha.T*kernel_rbf*alpha ))
constraints = [alpha.T @ kernel_rbf @ alpha ==0]
result = prob.solve()
"""

print(alpha_ls.value)
#regress= lambd*(alpha* kernel_rbf *alpha)
#alpha.value = alpha.value + regress

# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
#print(constraints[0].dual_value)

# Plot for optimal lambda:

plt.figure(1)
plt.plot(lambd_values_ls, train_errors_ls, label="Train error")
plt.plot(lambd_values_ls, test_errors_ls, label="Test error")
plt.xscale("log")
plt.legend(loc="upper left")
plt.xlabel(r"$\lambda$", fontsize=16)
plt.title("Mean Squared Error (MSE) part a")
plt.show()

print ("Optimal value of lambda in least square is : ", lambd_values_ls[12], "at gamma : ", gamma_ls)

# Taking optimal value of alpha at lambda =0.001
f_cap = alpha_values_ls[12]@ kernel_rbf_ls_train

#y_plot = np.arange(n)

plt.figure(2)
plt.scatter(x, y, label="X vs y")
plt.plot(x, f_x, label = "original_fx")
plt.plot(x_train, f_cap, label = "estimated_fx")
plt.title("Least square loss kernel part a")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()



###################################################################################
# Part b:

# Form and solve the Huber regression problem.
gamma_lub = 1/np.sqrt(2)
kernel_rbf_lub = kernel(x, gamma_lub)
kernel_rbf_lub_train = kernel_rbf_lub[:n_train, : n_train ]
kernel_rbf_lub_test = kernel_rbf_lub[:n_train, n_train:]

## CVXPY ridge regression:
alpha_lub = cp.Variable(n_train )
lambd_lub = cp.Parameter(nonneg=True)
problem_lub = cp.Problem(cp.Minimize(objective_fn_lub(kernel_rbf_lub_train, y_train, alpha_lub, lambd_lub)))

lambd_values_lub = np.logspace(-5, 3, 50)
train_errors_lub = []
test_errors_lub = []
alpha_values_lub = []
for v in lambd_values_lub:
    lambd_lub.value = v
    problem_lub.solve()
    train_errors_lub.append(mse(kernel_rbf_lub_train, y_train, alpha_lub))
    test_errors_lub.append(mse(kernel_rbf_lub_test.T, y_test, alpha_lub))
    alpha_values_lub.append(alpha_lub.value)

print(alpha_lub.value)

plt.figure(3)
plt.plot(lambd_values_lub, train_errors_lub, label="Train error")
plt.plot(lambd_values_lub, test_errors_lub, label="Test error")
plt.xscale("log")
plt.legend(loc="upper left")
plt.xlabel(r"$\lambda$", fontsize=16)
plt.title("Mean Squared Error (MSE) part b")
plt.show()

print ("Optimal value of lambda in huber loss is : ", lambd_values_ls[12], "at gamma : ", gamma_ls)

# Taking optimal value of alpha at lambda =0.001
f_cap_lub = alpha_values_lub[12]@ kernel_rbf_lub_train

#y_plot = np.arange(n)

plt.figure(4)
plt.scatter(x, y, label="X vs y")
plt.plot(x, f_x, label = "original_fx")
plt.plot(x_train, f_cap_lub, label = "estimated_fx")
plt.title("Huber loss kernel part b")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
