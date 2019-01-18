# Q5 Joke recommender system
import os
os.system('CLS')
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt


def read_joke_rating(fname):
    file = open(fname, "r")
    n = 24983
    m = 100
    R = np.empty([n, m]) * np.nan

    for line in file:
        elements = line.split(",")
        elements[0] = int(elements[0])
        elements[1] = int(elements[1])
        elements[2] = float(elements[2])
        row_indx = elements[0] - 1
        col_indx = elements[1] - 1
        rating = elements[2]
        R[row_indx, col_indx] = rating

    file.close()
    return R


def alternate_LS(num_factors, data_rating, hyper_par):
    d = num_factors
    n, m = data_rating.shape

    # Initialize U and V
    U = np.random.normal(scale = 1./d, size = (n, d))
    V = np.random.normal(scale = 1./d, size = (m, d))

    num_iter = 10
    itr = 0

    while itr < num_iter:
        # Alternating least squares for users treating jokes fixed
        VTV = V.T.dot(V)
        reg_part = np.eye(VTV.shape[0]) * hyper_par

        for itr_u in np.arange(U.shape[0]):
            U[itr_u, :] = np.linalg.solve((VTV + reg_part), data_rating[itr_u, :].dot(V))


        # Alternating least squares for jokes treating users fixed
        UTU = U.T.dot(U)
        reg_part = np.eye(UTU.shape[0]) * hyper_par

        for itr_j in np.arange(V.shape[0]):
            V[itr_j, :] = np.linalg.solve((UTU + reg_part), data_rating[:, itr_j].T.dot(U))

        itr = itr + 1

    return U, V


def alternate_LS_pooled(num_factors, data_rating):
    d = num_factors
    n, m = data_rating.shape

    # Initialize U and V
    U = np.ones((n, d))
    V = np.random.normal(scale = 1./d, size = (m, d))

    num_iter = 10
    itr = 0

    while itr < num_iter:
        # Alternating least squares for jokes, treating users fixed
        UTU = U.T.dot(U)
        for itr_j in np.arange(V.shape[0]):
            V[itr_j, :] = np.linalg.solve((UTU), data_rating[:, itr_j].T.dot(U))
        itr = itr + 1
    return U, V


def calc_error(true_data, estimated_data):
    error = true_data - estimated_data
    usr_rating = (~ np.isnan(error))*1
    num_usr_rating = np.sum(usr_rating)

    mean_sq_err = np.nansum(error**2)/num_usr_rating
    mean_abs_err = np.abs(error)
    mean_abs_err = np.nansum(mean_abs_err)/num_usr_rating
    return mean_sq_err, mean_abs_err


def plot_func(MSQ, MAE, num_factors):
    plt.clf()
    plt.plot(hyper_par_seq, MSQ[:, 1], label = "Training Set")
    plt.plot(hyper_par_seq, MSQ[:, 2], label = "Test Set")
    plt.xlabel("lambda")
    plt.ylabel("MSQ")
    plt.legend(loc='best')
    title_str = "Mean squared error: " + str(num_factors)
    plt.title(title_str)
    file_name = "C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/Q5c_msq_" + str(num_factors) + ".png"
    plt.savefig(file_name)

    plt.clf()
    plt.plot(hyper_par_seq, MAE[:, 1], label = "Training Set")
    plt.plot(hyper_par_seq, MAE[:, 2], label = "Test Set")
    plt.xlabel("lambda")
    plt.ylabel("MAE")
    plt.legend(loc='best')
    title_str = "Mean absolute error: " + str(num_factors)
    plt.title(title_str)
    file_name = "C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/Q5c_mae_" + str(num_factors) + ".png"
    plt.savefig(file_name)


# Main program

n = 24983
m = 100

file_train = "C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/jester/train.txt"
data_train = read_joke_rating(file_train)

file_test = "C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/jester/test.txt"
data_test = read_joke_rating(file_test)

# Let's find which users have rated jokes
usr_rated = (~ np.isnan(data_train))*1
#print(usr_rated)


# Q5.a
num_factors = 1
data_rating = data_train
data_rating[np.where(usr_rated == 0)] = 0

user_mat, jokes_mat = alternate_LS_pooled(num_factors, data_rating)
print("User matrix: ", user_mat.shape)
print("Jokes matrix: ", jokes_mat.shape)
rating_hat = user_mat.dot(jokes_mat.T)
print("Shape of reconstructed rating mat:\n", rating_hat.shape)
mse_tr, mae_tr = calc_error(data_train, rating_hat)
mse_test, mae_test = calc_error(data_test, rating_hat)
print("Training set - mean square error: ", mse_tr)
print("Test set - mean square error: ", mse_test)
print("\n")
print("Training set - mean absolute error: ", mae_tr)
print("Test set - mean absolute error: ", mae_test)

# Training set - mean square error:  9.68
# Test set - mean square error:  25.94
#
# Training set - mean absolute error:  1.87
# Test set - mean absolute error:  4.27


# Q5.b
data_rating = data_train
data_rating[np.where(usr_rated == 0)] = 0  # replace 'NaN' by zeros

num_factors_seq = [1, 2, 5, 10, 20, 50]

svd_factor_mse = np.empty([0, 3])
svd_factor_mae = np.empty([0, 3])


for num_fac in num_factors_seq:
    U, S, V = svds(data_rating, num_fac)
    # print(U.shape)
    # print(S.shape)
    # print(V.shape)
    rating_svd = U@np.diag(S)@V
    #print(rating_svd.shape)
    mse_train, mae_train = calc_error(data_train, rating_svd)
    mse_test, mae_test = calc_error(data_test, rating_svd)

    svd_factor_mse = np.append(svd_factor_mse, np.array([num_fac, mse_train, mse_test]).reshape(1, -1), axis=0)
    svd_factor_mae = np.append(svd_factor_mae, np.array([num_fac, mae_train, mae_test]).reshape(1, -1), axis=0)

print("SVD: Mean square error: \n", svd_factor_mse)
print("SVD: Mean absolute error: \n", svd_factor_mae)

plt.clf()
plt.plot(svd_factor_mse[:, 0], svd_factor_mse[:, 1], label = "Training Set")
plt.plot(svd_factor_mse[:, 0], svd_factor_mse[:, 2], label = "Test Set")
plt.xlabel("Number of latent factors")
plt.ylabel("MSE")
plt.legend(loc='best')
plt.title("SVD: Mean squared error vs. latent factors")
file_name = "C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/Q5c_svd_mse.png"
plt.savefig(file_name)

plt.clf()
plt.plot(svd_factor_mae[:, 0], svd_factor_mae[:, 1], label = "Training Set")
plt.plot(svd_factor_mae[:, 0], svd_factor_mae[:, 2], label = "Test Set")
plt.xlabel("Number of latent factors")
plt.ylabel("MAE")
plt.legend(loc='best')
plt.title("SVD: Mean absolute error vs. latent factors")
file_name = "C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/Q5c_svd_mae.png"
plt.savefig(file_name)


# Q5.c
data_rating = data_train
data_rating[np.where(usr_rated == 0)] = 0

factor_mse = np.empty([0, 3])
factor_mae = np.empty([0, 3])


###
num_factors = 1
#hyper_par_seq = [0.01, 0.1, 0.5, 1, 50, 100]
hyper_par = 0.01

MSQ = np.empty([0, 3])  # store MSQ for train and test
MAE = np.empty([0, 3])  # store MAE for train and test


# fine tuned the hyper parameters by testing on a sequence of values
# after choosing the optimal hyper para, the loop indentaion is removed

#for hyper_par in hyper_par_seq:
user_mat, jokes_mat = alternate_LS(num_factors, data_rating, hyper_par)
print("User matrix: ", user_mat.shape)
print("Jokes matrix: ", jokes_mat.shape)
rating_hat = user_mat.dot(jokes_mat.T)
print("Shape of reconstructed rating mat:\n", rating_hat.shape)
msq_tr, mae_tr = calc_error(data_train, rating_hat)
msq, mae = calc_error(data_test, rating_hat)
# MSQ = np.append(MSQ, np.array([hyper_par, msq_tr, msq]).reshape(1, -1), axis=0)
# MAE = np.append(MAE, np.array([hyper_par, mae_tr, mae]).reshape(1, -1), axis=0)
#plot_func(MSQ, MAE, num_factors)

# indx = np.argmin(MSQ, axis=0)
# reg_para = MSQ[indx[1], 0]
# mse_train = MSQ[indx[1], 1]
# mse_test = MSQ[indx[1], 2]
# factor_mse = np.append(factor_mse, np.array([reg_para, mse_train, mse_test]).reshape(1, -1), axis=0)
#
# indx = np.argmin(MAE, axis=0)
# reg_para = MAE[indx[1], 0]
# mae_train = MAE[indx[1], 1]
# mae_test = MAE[indx[1], 2]
# factor_mae = np.append(factor_mae, np.array([reg_para, mae_train, mae_test]).reshape(1, -1), axis=0)

factor_mse = np.append(factor_mse, np.array([num_factors, msq_tr, msq]).reshape(1, -1), axis=0)
factor_mae = np.append(factor_mae, np.array([num_factors, mae_tr, mae]).reshape(1, -1), axis=0)


###
num_factors = 2
#hyper_par_seq = [0.01, 0.1, 0.5, 1, 20]
hyper_par = 0.1

MSQ = np.empty([0, 3])  # store MSQ for train and test
MAE = np.empty([0, 3])  # store MAE for train and test

#for hyper_par in hyper_par_seq:
user_mat, jokes_mat = alternate_LS(num_factors, data_rating, hyper_par)
print("User matrix: ", user_mat.shape)
print("Jokes matrix: ", jokes_mat.shape)
rating_hat = user_mat.dot(jokes_mat.T)
print("Shape of reconstructed rating mat:\n", rating_hat.shape)
msq_tr, mae_tr = calc_error(data_train, rating_hat)
msq, mae = calc_error(data_test, rating_hat)
#     MSQ = np.append(MSQ, np.array([hyper_par, msq_tr, msq]).reshape(1, -1), axis=0)
#     MAE = np.append(MAE, np.array([hyper_par, mae_tr, mae]).reshape(1, -1), axis=0)
# plot_func(MSQ, MAE, num_factors)

factor_mse = np.append(factor_mse, np.array([num_factors, msq_tr, msq]).reshape(1, -1), axis=0)
factor_mae = np.append(factor_mae, np.array([num_factors, mae_tr, mae]).reshape(1, -1), axis=0)

###
num_factors = 5
#hyper_par_seq = [0.01, 0.1, 0.5, 1, 50, 100]
hyper_par = 1

MSQ = np.empty([0, 3])  # store MSQ for train and test
MAE = np.empty([0, 3])  # store MAE for train and test

#for hyper_par in hyper_par_seq:
user_mat, jokes_mat = alternate_LS(num_factors, data_rating, hyper_par)
print("User matrix: ", user_mat.shape)
print("Jokes matrix: ", jokes_mat.shape)
rating_hat = user_mat.dot(jokes_mat.T)
print("Shape of reconstructed rating mat:\n", rating_hat.shape)
msq_tr, mae_tr = calc_error(data_train, rating_hat)
msq, mae = calc_error(data_test, rating_hat)
#     MSQ = np.append(MSQ, np.array([hyper_par, msq_tr, msq]).reshape(1, -1), axis=0)
#     MAE = np.append(MAE, np.array([hyper_par, mae_tr, mae]).reshape(1, -1), axis=0)
# plot_func(MSQ, MAE, num_factors)

factor_mse = np.append(factor_mse, np.array([num_factors, msq_tr, msq]).reshape(1, -1), axis=0)
factor_mae = np.append(factor_mae, np.array([num_factors, mae_tr, mae]).reshape(1, -1), axis=0)

###
num_factors = 10
#hyper_par_seq = [0.1, 0.5, 1, 50, 100, 120, 200]
hyper_par = 5

MSQ = np.empty([0, 3])  # store MSQ for train and test
MAE = np.empty([0, 3])  # store MAE for train and test

#for hyper_par in hyper_par_seq:
user_mat, jokes_mat = alternate_LS(num_factors, data_rating, hyper_par)
print("User matrix: ", user_mat.shape)
print("Jokes matrix: ", jokes_mat.shape)
rating_hat = user_mat.dot(jokes_mat.T)
print("Shape of reconstructed rating mat:\n", rating_hat.shape)
msq_tr, mae_tr = calc_error(data_train, rating_hat)
msq, mae = calc_error(data_test, rating_hat)
#     MSQ = np.append(MSQ, np.array([hyper_par, msq_tr, msq]).reshape(1, -1), axis=0)
#     MAE = np.append(MAE, np.array([hyper_par, mae_tr, mae]).reshape(1, -1), axis=0)
# plot_func(MSQ, MAE, num_factors)

factor_mse = np.append(factor_mse, np.array([num_factors, msq_tr, msq]).reshape(1, -1), axis=0)
factor_mae = np.append(factor_mae, np.array([num_factors, mae_tr, mae]).reshape(1, -1), axis=0)

###
num_factors = 20
#hyper_par_seq = [0.1, 0.5, 1, 50, 100, 120, 150, 200]
hyper_par = 100

MSQ = np.empty([0, 3])  # store MSQ for train and test
MAE = np.empty([0, 3])  # store MAE for train and test

#for hyper_par in hyper_par_seq:
user_mat, jokes_mat = alternate_LS(num_factors, data_rating, hyper_par)
print("User matrix: ", user_mat.shape)
print("Jokes matrix: ", jokes_mat.shape)
rating_hat = user_mat.dot(jokes_mat.T)
print("Shape of reconstructed rating mat:\n", rating_hat.shape)
msq_tr, mae_tr = calc_error(data_train, rating_hat)
msq, mae = calc_error(data_test, rating_hat)
#     MSQ = np.append(MSQ, np.array([hyper_par, msq_tr, msq]).reshape(1, -1), axis=0)
#     MAE = np.append(MAE, np.array([hyper_par, mae_tr, mae]).reshape(1, -1), axis=0)
# plot_func(MSQ, MAE, num_factors)

factor_mse = np.append(factor_mse, np.array([num_factors, msq_tr, msq]).reshape(1, -1), axis=0)
factor_mae = np.append(factor_mae, np.array([num_factors, mae_tr, mae]).reshape(1, -1), axis=0)

###
num_factors = 50
#hyper_par_seq = [170, 200, 250, 300]
hyper_par = 250

MSQ = np.empty([0, 3])  # store MSQ for train and test
MAE = np.empty([0, 3])  # store MAE for train and test

#for hyper_par in hyper_par_seq:
user_mat, jokes_mat = alternate_LS(num_factors, data_rating, hyper_par)
print("User matrix: ", user_mat.shape)
print("Jokes matrix: ", jokes_mat.shape)
rating_hat = user_mat.dot(jokes_mat.T)
print("Shape of reconstructed rating mat:\n", rating_hat.shape)
msq_tr, mae_tr = calc_error(data_train, rating_hat)
msq, mae = calc_error(data_test, rating_hat)
#     MSQ = np.append(MSQ, np.array([hyper_par, msq_tr, msq]).reshape(1, -1), axis=0)
#     MAE = np.append(MAE, np.array([hyper_par, mae_tr, mae]).reshape(1, -1), axis=0)
# plot_func(MSQ, MAE, num_factors)

factor_mse = np.append(factor_mse, np.array([num_factors, msq_tr, msq]).reshape(1, -1), axis=0)
factor_mae = np.append(factor_mae, np.array([num_factors, mae_tr, mae]).reshape(1, -1), axis=0)

print("Factor MSE: \n", factor_mse)
print("Factor MAE: \n", factor_mae)

plt.clf()
plt.plot(factor_mse[:, 0], factor_mse[:, 1], label = "Training Set")
plt.plot(factor_mse[:, 0], factor_mse[:, 2], label = "Test Set")
plt.xlabel("Number of latent factors")
plt.ylabel("MSE")
plt.legend(loc='best')
plt.title("Mean squared error vs. latent factors")
file_name = "C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/Q5c_mse.png"
plt.savefig(file_name)

plt.clf()
plt.plot(factor_mae[:, 0], factor_mae[:, 1], label = "Training Set")
plt.plot(factor_mae[:, 0], factor_mae[:, 2], label = "Test Set")
plt.xlabel("Number of latent factors")
plt.ylabel("MAE")
plt.legend(loc='best')
plt.title("Mean absolute error vs. latent factors")
file_name = "C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW3/Q5c_mae.png"
plt.savefig(file_name)
