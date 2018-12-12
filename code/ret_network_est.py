
import os
os.system('CLS')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import multiprocessing
from sklearn.preprocessing import scale
#from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error

import load_data

sns.set()


#def est_coefs(y, lagged_ret):
def est_coefs(i):
    #coefs=[]
    cross_val_par = 5
    lasso = Lasso(fit_intercept = False, max_iter = 10000, normalize = True)
    lassocv = LassoCV(alphas = None, fit_intercept = False, cv = cross_val_par, max_iter = 100000, normalize = True)

    y = Y[:, i]
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

    return coefs


computer = 'laptop'
#computer = 'TS'

if computer == 'laptop':
    data_file = 'C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/Project/Data/sp470_price.csv'
    var_filename = 'C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/Project/return_adj.pkl'
else:
    data_file = 'H:/local/sandp500/sp470.csv'
    var_filename = 'H:/CSE546/Project/Data/return_adj.pkl'

# price_data = pd.read_csv(data_file, index_col=0)
# index = price_data.index

ret_data, vol_data, comp_list = load_data.read_data()

# Calculate return from price data
# ret_data = 100*(np.log(price_data) - np.log(price_data.shift(1)))
# ret_data = ret_data.dropna()
# print(ret_data.head())

################################################
# Doing it on a small sample of ten firms
# ret_data = ret_data.iloc[0:100 ,0:50]
# print(ret_data.shape)
# print("Return data: \n", ret_data)

col_names = ret_data.columns.tolist()
print("Column names: \n", col_names)

################################################
lagged_ret = ret_data.shift(1).dropna()
num_obs = lagged_ret.shape[0]
num_firms = lagged_ret.shape[1]

# print(np.arange(num_obs))
# print(lagged_ret.index)

# Aligning indices of X and y ( each y is a column in Y)
Y = ret_data[ret_data.index.isin(lagged_ret.index)]

train_size = int(0.8*num_obs)
# print("Num of obs in train set: ", train_size)

train_index = np.random.choice(num_obs, train_size, replace=False)
test_index = np.setdiff1d(np.arange(num_obs), train_index)

# print("Train indices: ", train_index)
# print("Test indices: ", test_index)

# convert pandas dataframe to numpy array
lagged_ret = lagged_ret.values
Y = Y.values

print("Vol data shape: ", vol_data.shape)
'''
##############################################################################
coef_mat = []

for itr in np.arange(num_firms):
#for itr in np.arange(3):
     coefs = est_coefs(itr)
     coef_mat.append(coefs)


coef_mat = np.array(coef_mat)

print("Results: \n", )
print("Coef Mat Shape: ", coef_mat.shape)
print("Coef Mat: ", coef_mat)


indx_conn = np.where(coef_mat != 0)
print(indx_conn)

# Create the adjacency matrix based on estimated coefs
adj_mat = np.zeros((d, d))
adj_mat[indx_conn] = 1
print("Adjacency matrix:\n", adj_mat)


import dill                            #pip install dill --user
dill.dump_session(var_filename)

# and to load the session again:
#dill.load_session(filename)
'''
