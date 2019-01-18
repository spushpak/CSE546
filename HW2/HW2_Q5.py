import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set()


def load_dataset():
    mndata = MNIST('C:\\local\\MNIST_data\\')
    mndata.gz = True
    x_train, labels_train = map(np.array, mndata.load_training())
    x_test, labels_test = map(np.array, mndata.load_testing())
    x_train = x_train/255.0
    x_test = x_test/255.0
    return x_train, labels_train, x_test, labels_test


def mu(wvec, b, xvec, y):
    return 1/(1 + np.exp(-y*(b + np.dot(xvec, wvec))))


def grad_b(w, b, X, Y):
    sum = 0
    n = X.shape[0]
    for i in np.arange(n):
        sum = sum + mu(w, b, X[i, :], Y[i, 0])

    sum = sum / n
    sum = sum - np.mean(Y)
    return sum


def hess_b(w, b, X, Y):
    n = X.shape[0]
    prod = np.full(n, 0.0)
    for i in np.arange(n):
        temp = mu(w, b, X[i, :], Y[i, 0])
        prod[i] = temp*(1 - temp)*Y[i, 0]**2
    return np.mean(prod)


def grad_w(w, b, X, Y, lam):
    n = X.shape[0]
    d = X.shape[1]
    #grad_vec = np.full(d, 0.0)
    Mu = np.full(n, 0.0)
    sum = 0
    for i in np.arange(n):
        Mu[i] = mu(w, b, X[i, :], Y[i, 0]) - 1
    Mu = Mu.reshape(-1, 1)
    #print("Shape of Mu inside function: ", Mu.shape)

    Mu_Y = np.multiply(Mu, Y).reshape(-1, 1)  # this is of order (n X 1)
    #print("Shape of Mu_Y inside function: ", Mu_Y.shape)

    # X.T is (d X n); X.T@Mu_Y is (d X 1); so grad vec is (d X 1)
    grad_vec = (X.T@Mu_Y)/n + 2*lam*w.reshape(-1,1)
    grad_vec = grad_vec.reshape(-1, 1)
    #print("Shape of grad vec: ", grad_vec.shape)
    return grad_vec

def hess_w(w, b, X, Y, lam):
    n = X.shape[0]
    d = X.shape[1]

    Mu = np.full(n, 0.0)
    sum = 0
    for i in np.arange(n):
        Mu[i] = mu(w, b, X[i, :], Y[i, 0]) - 1
    Mu = Mu.reshape(-1, 1)
    prod = np.diag(np.diag(Mu@(1- Mu).T))
    hess_mat = X.T@prod@X + 2*lam*np.identity(d)
    #print("Shape of hess: ", hess_mat.shape)
    return hess_mat


def Jfun_val(wvec, b, X, Y, lam):
    temp1 = np.multiply(X@wvec.reshape(-1, 1) + b, Y)  # should be nX1
    #print("Elements of temp1", temp1[0:10,: ])
    temp2 = np.log(1 + np.exp(temp1))
    val = np.mean(temp2) + lam*np.linalg.norm(wvec)
    return val


X_train, train_labels, X_test, test_labels = load_dataset()
# print(train_labels.dtype)
# print(train_labels[0:20])

p = (train_labels == 7)
q = (train_labels == 2)
r = np.logical_or(p, q)
s = np.where(r)
# print("r: ", r)
# print("s: ", s)

train_labels = train_labels[s]
X_train = X_train[s]
# print(train_labels[0:20])
# print(type(train_labels))

t = np.where(train_labels == 7)
u = np.where(train_labels == 2)

# print("Where are 7's: ", t)
# print("Where are 2's: ", u)

# Changing data from unsigned int to signed int
train_labels.dtype = np.int8
# print("After changing type: ", train_labels[0:20])

# Y = 1 for 7 and Y = -1 for 2
train_labels[t] = 1
train_labels[u] = -1

# print(train_labels[0:20])
# print(train_labels.dtype)
# print("Shape of X_train: ", X_train.shape)

train_labels = train_labels.reshape(-1,1)

# print("Shape of train labels: ", train_labels.shape)

#############################################################################
# Test set
p = (test_labels == 7)
q = (test_labels == 2)
r = np.logical_or(p, q)
s = np.where(r)
test_labels = test_labels[s]
X_test = X_test[s]
t = np.where(test_labels == 7)
u = np.where(test_labels == 2)

# Changing data from unsigned int to signed int
test_labels.dtype = np.int8
test_labels[t] = 1
test_labels[u] = -1
test_labels = test_labels.reshape(-1,1)

lam = 0.1
step_size = 0.1

##############################################################################
# Gradient Descent Method
# Run it for the training set
X = X_train
Y = train_labels
n = X.shape[0]
d = X.shape[1]
wvec = np.full(d, 0.0)
b = 0
fun_val = np.empty([0, 3])  # iteration num, func val for train, func val for test
misclass_error = np.empty([0, 3])  # iteration num, misclass for train, misclass for test
norm_w = np.empty([0, 2])  # itr, norm of w

itr = 0
converged = False

while not converged:
    #print("Gradient Descent Iteration: ", itr)
    fun_val_old = Jfun_val(wvec, b, X, Y, lam)
    #print("fun_val_old: ", fun_val_old)
    grad_wval = grad_w(wvec, b, X, Y, lam)
    #print("First few elements of grad_vec", grad_wval[0:3,:])
    grad_bval = grad_b(wvec, b, X, Y)
    #print("grad_bval", grad_bval)

    w_update = wvec.reshape(-1, 1) - step_size*grad_wval
    #print("First few elements of w_update: ", w_update[0:3,:])
    b_update = b - step_size*grad_bval
    #print("b_update: ", b_update)
    fun_val_new = Jfun_val(w_update, b_update, X, Y, lam)
    fun_val_test = Jfun_val(w_update, b_update, X_test, test_labels, lam)
    #print("New func value: ", fun_val_new)
    converged = np.allclose(fun_val_old, fun_val_new, 0.001)
    wvec = w_update
    b = b_update
    itr = itr + 1
    no_match_train = np.count_nonzero(train_labels - np.sign(X_train@wvec + b))
    misclass_err_train = no_match_train/X_train.shape[0]

    no_match_test = np.count_nonzero(test_labels - np.sign(X_test@wvec + b))
    misclass_err_test = no_match_test/X_test.shape[0]

    fun_val = np.append(fun_val, np.array([itr, fun_val_new, fun_val_test]).reshape(1, 3), axis=0)
    misclass_error = np.append(misclass_error, np.array([itr, misclass_err_train, misclass_err_test]).reshape(1, 3), axis=0)
    norm_w = np.append(norm_w, np.array([itr, np.linalg.norm(wvec)]).reshape(1, 2), axis=0)


# Save estimated b and w's from training Set - to be used on test set prediction
wopt_train = wvec
bopt_train = b

# plt.clf()
# plt.plot(fun_val[:, 0], fun_val[:, 1])
# plt.xlabel("Iteration")
# plt.ylabel("J(w,b)")
# plt.title("Function Value: Training Set")
# #plt.show()
# plt.savefig('./HW2/J_func_training.png')

# plt.clf()
# plt.plot(misclass_error[:, 0], misclass_error[:, 1])
# plt.xlabel("Iteration")
# plt.ylabel("Misclassification error")
# plt.title("Misclassification error: Training Set")
# #plt.show()
# plt.savefig('./HW2/misclass_training.png')

# plt.clf()
# plt.plot(norm_w[:, 0], norm_w[:, 1])
# plt.xlabel("Iteration")
# plt.ylabel("Norm of W")
# plt.savefig('./HW2/norm_training.png')

# plt.clf()
# plt.plot(fun_val[:, 0], fun_val[:, 2])
# plt.xlabel("Iteration")
# plt.ylabel("J(w,b)")
# plt.title("Function Value: Test Set")
# #plt.show()
# plt.savefig('./HW2/J_func_test.png')

# plt.clf()
# plt.plot(misclass_error[:, 0], misclass_error[:, 2])
# plt.xlabel("Iteration")
# plt.ylabel("Misclassification error")
# plt.title("Misclassification error: Test Set")
# #plt.show()
# plt.savefig('./HW2/misclass_test.png')

# Plot func value on the same plot
plt.clf()
plt.plot(fun_val[:, 0], fun_val[:, 1], label = "Training Set")
plt.plot(fun_val[:, 0], fun_val[:, 2], label = "Test Set")
plt.xlabel("Iteration")
plt.ylabel("J(w,b)")
plt.legend(loc='best')
plt.title("Gradient Descent")
#plt.show()
plt.savefig('./HW2/J_func_train_test.png')

plt.clf()
plt.plot(misclass_error[:, 0], misclass_error[:, 1], label = "Training Set")
plt.plot(misclass_error[:, 0], misclass_error[:, 2], label = "Test Set")
plt.xlabel("Iteration")
plt.ylabel("Misclassification error")
plt.title("Gradient Descent")
plt.legend(loc='best')
#plt.show()
plt.savefig('./HW2/misclass_train_test.png')

###############################################################################
# Stochastic Gradient Descent with batch size = 1
step_size = 0.1
wvec = np.full(d, 0.0)
b = 0
fun_val = np.empty([0, 3])  # iteration num, func val for train, func val for test
misclass_error = np.empty([0, 3])  # iteration num, misclass for train, misclass for test
train_indx = np.random.choice(np.arange(X_train.shape[0]), 1, replace=False)

itr = 0
converged = False

while not converged:
    #print("Stochastic Gradient Descent with batch = 1 Iteration: ", itr)
    fun_val_old = Jfun_val(wvec, b, X_train[train_indx, :], train_labels[train_indx, :], lam)
    grad_wval = grad_w(wvec, b, X_train[train_indx, :], train_labels[train_indx, :], lam)
    grad_bval = grad_b(wvec, b, X_train[train_indx, :], train_labels[train_indx, :])

    w_update = wvec.reshape(-1, 1) - step_size*grad_wval
    b_update = b - step_size*grad_bval
    fun_val_new = Jfun_val(w_update, b_update, X_train[train_indx, :], train_labels[train_indx, :], lam)
    fun_val_test = Jfun_val(w_update, b_update, X_test, test_labels, lam)
    converged = np.allclose(fun_val_old, fun_val_new, 0.001)
    wvec = w_update
    b = b_update
    itr = itr + 1
    no_match_train = np.count_nonzero(train_labels - np.sign(X_train@wvec + b))
    misclass_err_train = no_match_train/X_train.shape[0]

    no_match_test = np.count_nonzero(test_labels - np.sign(X_test@wvec + b))
    misclass_err_test = no_match_test/X_test.shape[0]

    fun_val = np.append(fun_val, np.array([itr, fun_val_new, fun_val_test]).reshape(1, 3), axis=0)
    misclass_error = np.append(misclass_error, np.array([itr, misclass_err_train, misclass_err_test]).reshape(1, 3), axis=0)

# Plot func value on the same plot
plt.clf()
plt.plot(fun_val[:, 0], fun_val[:, 1], label = "Training Set")
plt.plot(fun_val[:, 0], fun_val[:, 2], label = "Test Set")
plt.xlabel("Iteration")
plt.ylabel("J(w,b)")
plt.legend(loc='best')
plt.title("Stochastic Gradient Descent with batch=1")
#plt.show()
plt.savefig('./HW2/J_func_train_test_sgd1.png')

plt.clf()
plt.plot(misclass_error[:, 0], misclass_error[:, 1], label = "Training Set")
plt.plot(misclass_error[:, 0], misclass_error[:, 2], label = "Test Set")
plt.xlabel("Iteration")
plt.ylabel("Misclassification error")
plt.legend(loc='best')
plt.title("Stochastic Gradient Descent with batch=1")
#plt.show()
plt.savefig('./HW2/misclass_train_test_sgd1.png')


# Stochastic Gradient Descent with batch size = 100
step_size = 0.1
wvec = np.full(d, 0.0)
b = 0
fun_val = np.empty([0, 3])  # iteration num, func val for train, func val for test
misclass_error = np.empty([0, 3])  # iteration num, misclass for train, misclass for test
train_indx = np.random.choice(np.arange(X_train.shape[0]), 100, replace=False)

itr = 0
converged = False

while not converged:
    #print("Stochastic Gradient Descent with batch = 100 Iteration: ", itr)
    fun_val_old = Jfun_val(wvec, b, X_train[train_indx, :], train_labels[train_indx, :], lam)
    grad_wval = grad_w(wvec, b, X_train[train_indx, :], train_labels[train_indx, :], lam)
    grad_bval = grad_b(wvec, b, X_train[train_indx, :], train_labels[train_indx, :])

    w_update = wvec.reshape(-1, 1) - step_size*grad_wval
    b_update = b - step_size*grad_bval
    fun_val_new = Jfun_val(w_update, b_update, X_train[train_indx, :], train_labels[train_indx, :], lam)
    fun_val_test = Jfun_val(w_update, b_update, X_test, test_labels, lam)
    converged = np.allclose(fun_val_old, fun_val_new, 0.001)
    wvec = w_update
    b = b_update
    itr = itr + 1
    no_match_train = np.count_nonzero(train_labels - np.sign(X_train@wvec + b))
    misclass_err_train = no_match_train/X_train.shape[0]

    no_match_test = np.count_nonzero(test_labels - np.sign(X_test@wvec + b))
    misclass_err_test = no_match_test/X_test.shape[0]

    fun_val = np.append(fun_val, np.array([itr, fun_val_new, fun_val_test]).reshape(1, 3), axis=0)
    misclass_error = np.append(misclass_error, np.array([itr, misclass_err_train, misclass_err_test]).reshape(1, 3), axis=0)


# Plot func value on the same plot
plt.clf()
plt.plot(fun_val[:, 0], fun_val[:, 1], label = "Training Set")
plt.plot(fun_val[:, 0], fun_val[:, 2], label = "Test Set")
plt.xlabel("Iteration")
plt.ylabel("J(w,b)")
plt.legend(loc='best')
plt.title("Stochastic Gradient Descent with batch=100")
#plt.show()
plt.savefig('./HW2/J_func_train_test_sgd100.png')

plt.clf()
plt.plot(misclass_error[:, 0], misclass_error[:, 1], label = "Training Set")
plt.plot(misclass_error[:, 0], misclass_error[:, 2], label = "Test Set")
plt.xlabel("Iteration")
plt.ylabel("Misclassification error")
plt.title("Stochastic Gradient Descent with batch=100")
plt.legend(loc='best')
#plt.show()
plt.savefig('./HW2/misclass_train_test_sgd100.png')

###############################################################################
# Newton method
# We need the Hessian for it. The Hessian for this case can be expressed
# as H = X_TSX where S = diag(mu_i(1 - mu_i))
# Run it for the training set
n = X_train.shape[0]
d = X_train.shape[1]
wvec = np.full(d, 0.0)
b = 0
fun_val = np.empty([0, 3])  # iteration num, func val for train, func val for test
misclass_error = np.empty([0, 3])  # iteration num, misclass for train, misclass for test

itr = 0
converged = False

while not converged:
    fun_val_old = Jfun_val(wvec, b, X_train, train_labels, lam)
    grad_wval = grad_w(wvec, b, X_train, train_labels, lam)
    hess_wval = hess_w(wvec, b, X_train, train_labels, lam)
    grad_bval = grad_b(wvec, b, X_train, train_labels)
    hess_bval = hess_b(wvec, b, X_train, train_labels)
    hess_inv = np.linalg.solve(hess_wval, np.identity(d))
    w_update = wvec.reshape(-1, 1) - step_size*(hess_inv@grad_wval)
    b_update = b - (step_size*grad_bval)/hess_bval
    fun_val_new = Jfun_val(w_update, b_update, X, Y, lam)
    fun_val_test = Jfun_val(w_update, b_update, X_test, test_labels, lam)
    converged = np.allclose(fun_val_old, fun_val_new, 0.001)
    wvec = w_update
    b = b_update
    itr = itr + 1
    no_match_train = np.count_nonzero(train_labels - np.sign(X_train@wvec + b))
    misclass_err_train = no_match_train/X_train.shape[0]

    no_match_test = np.count_nonzero(test_labels - np.sign(X_test@wvec + b))
    misclass_err_test = no_match_test/X_test.shape[0]

    fun_val = np.append(fun_val, np.array([itr, fun_val_new, fun_val_test]).reshape(1, 3), axis=0)
    misclass_error = np.append(misclass_error, np.array([itr, misclass_err_train, misclass_err_test]).reshape(1, 3), axis=0)

# Plot func value on the same plot
plt.clf()
plt.plot(fun_val[:, 0], fun_val[:, 1], label = "Training Set")
plt.plot(fun_val[:, 0], fun_val[:, 2], label = "Test Set")
plt.xlabel("Iteration")
plt.ylabel("J(w,b)")
plt.title("Function Value: Newton's Method")
plt.legend(loc='best')
#plt.show()
plt.savefig('./HW2/J_func_train_test_newton.png')

plt.clf()
plt.plot(misclass_error[:, 0], misclass_error[:, 1], label = "Training Set")
plt.plot(misclass_error[:, 0], misclass_error[:, 2], label = "Test Set")
plt.xlabel("Iteration")
plt.ylabel("Misclassification error")
plt.title("Misclassification error: Newton's Method")
plt.legend(loc='best')
#plt.show()
plt.savefig('./HW2/misclass_train_test_newton.png')
