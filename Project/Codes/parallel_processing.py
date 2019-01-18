import os
os.system('CLS')
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import time
import matplotlib.pyplot as plt
import glob
from PIL import Image
import random
import string

import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error

import load_data


MULTITHREADING_TITLE="Multithreading"
MULTIPROCESSING_TITLE="Multiprocessing"

def multithreading(func, args, workers):
    begin_time = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        res = executor.map(func, args, [begin_time for i in range(len(args))])
    print("printing results in multhreading: ", res)
    return list(res)

def multiprocessing(func, args, workers):
    begin_time = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        res = executor.map(func, args, [begin_time for i in range(len(args))])
    return list(res)


def est_coefs(i, YY):
    #coefs=[]
    cross_val_par = 5
    lasso = Lasso(fit_intercept = False, max_iter = 10000, normalize = True)
    lassocv = LassoCV(alphas = None, fit_intercept = False, cv = cross_val_par, max_iter = 100000, normalize = True)

    y = YY[:, i]
    y = y.reshape(-1, 1)

    X_train = lagged_ret[train_index, :]
    X_test = lagged_ret[test_index, :]
    y_train = y[train_index, :].ravel()

    y_test = y[test_index, :]
    lassocv.fit(X_train, y_train)
    # print("\n")
    # print("Optimal cv parameter: ", lassocv.alpha_)

    lasso.set_params(alpha=lassocv.alpha_)
    lasso.fit(X_train, y_train.ravel())

    #coefs.append([col_names[i], lasso.coef_])
    coefs = lasso.coef_
    # coefs = lasso.coef_.reshape(1, -1)

    # print("MSE: ", mean_squared_error(y_test, lasso.predict(X_test)))
    # print("Lasso coefs: ", coefs)

    print("before return in est_coefs")
    return coefs


def main():
    computer = 'laptop'
    #computer = 'TS'

    if computer == 'laptop':
        data_file = 'C:/local/sandp500/sp470.csv'
        var_filename = 'C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/Project/return_adj.pkl'
    else:
        data_file = 'H:/local/sandp500/sp470.csv'
        var_filename = 'H:/CSE546/Project/return_adj.pkl'

    ret_data, vol_data, comp_list = load_data.read_data()

    ################################################
    # Doing it on a small sample of ten firms
    # ret_data = ret_data.iloc[0:100 ,0:50]
    # print(ret_data.shape)
    # print("Return data: \n", ret_data)

    col_names = ret_data.columns.tolist()
    #print("Column names: \n", col_names)

    ################################################
    lagged_ret = ret_data.shift(1).dropna()
    num_obs = lagged_ret.shape[0]
    num_firms = lagged_ret.shape[1]

    # Aligning indices of X and y ( each y is a column in Y)
    Y = ret_data[ret_data.index.isin(lagged_ret.index)]

    train_size = int(0.8*num_obs)
    # print("Num of obs in train set: ", train_size)

    train_index = np.random.choice(num_obs, train_size, replace=False)
    test_index = np.setdiff1d(np.arange(num_obs), train_index)

    # convert pandas dataframe to numpy array
    lagged_ret = lagged_ret.values
    Y = Y.values

    ##############################################################################
    # Do multiprocessing
    #import multiprocessing
    #num_cores = multiprocessing.cpu_count()
    #print("How many cores: ", num_cores)   # num of cores = 4

    inputs = np.arange(num_firms)

    with ProcessPoolExecutor(max_workers=2) as executor:
        res = executor.map(est_coefs, inputs)

    #print("Results: \n" + str(res))
    return res

if __name__ == '__main__':
    result = main()
    print(result)
    #print('main: unprocessed results {}'.format(result))
    #print('main: waiting for real results')
    real_result = result.tolist()
    print('main: results: {}',format(real_result))


'''
indx_conn = np.where(coefs != 0)
print(indx_conn)

# Create the adjacency matrix based on estimated coefs
adj_mat = np.zeros((d, d))
adj_mat[indx_conn] = 1
print("Adjacency matrix:\n", adj_mat)

'''
