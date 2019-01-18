
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
    data_file = 'C:/local/sandp500/sp470.csv'
    var_filename = 'C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/Project/return_adj.pkl'
else:
    data_file = 'H:/local/sandp500/sp470.csv'
    var_filename = 'H:/CSE546/Project/return_adj.pkl'

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

##############################################################################
# Do multiprocessing
num_cores = multiprocessing.cpu_count()
# print("How many cores: ", num_cores)

inputs = np.arange(num_firms)
results = Parallel(n_jobs=num_cores)(delayed(est_coefs)(itr) for itr in inputs)
#coefs = np.array(results)

# print("Results: \n", coefs)
# print("Results shape: \n", coefs.shape)

# coefs = ret_data.apply(est_coefs, axis=0, args=(lagged_ret, ))
# coefs = ret_data.apply(est_coefs, axis=0)


indx_conn = np.where(coefs != 0)
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
####################################
# Graphical LASSO
print(__doc__)
# author: Gael Varoquaux <gael.varoquaux@inria.fr>
# License: BSD 3 clause
# Copyright: INRIA

import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
import matplotlib.pyplot as plt

# #############################################################################
X = ret_data.dropna()
n_samples = X.shape[0]
n_features = X.shape[1]

# Estimate the covariance
# emp_cov = np.dot(X.T, X) / n_samples
emp_cov = np.cov(X.T)
print("Shape: ", emp_cov.shape)


model = GraphicalLassoCV(cv=5)
model.fit(X)
cov_ = model.covariance_
prec_ = model.precision_

lw_cov_, _ = ledoit_wolf(X)
lw_prec_ = linalg.inv(lw_cov_)

# #############################################################################
# Plot the results
plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.02, right=0.98)

# plot the covariances
covs = [('Empirical', emp_cov), ('Ledoit-Wolf', lw_cov_),
        ('GraphicalLassoCV', cov_)]
vmax = cov_.max()
for i, (name, this_cov) in enumerate(covs):
    plt.subplot(2, 4, i + 1)
    plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
               cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('%s covariance' % name)


# plot the precisions
precs = [('Empirical', linalg.inv(emp_cov)), ('Ledoit-Wolf', lw_prec_),
         ('GraphicalLasso', prec_)]
vmax = .9 * prec_.max()
for i, (name, this_prec) in enumerate(precs):
    ax = plt.subplot(2, 4, i + 5)
    plt.imshow(np.ma.masked_equal(this_prec, 0),
               interpolation='nearest', vmin=-vmax, vmax=vmax,
               cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('%s precision' % name)
    if hasattr(ax, 'set_facecolor'):
        ax.set_facecolor('.7')
    else:
        ax.set_axis_bgcolor('.7')

# plot the model selection metric
plt.figure(figsize=(4, 3))
plt.axes([.2, .15, .75, .7])
plt.plot(model.cv_alphas_, np.mean(model.grid_scores_, axis=1), 'o-')
plt.axvline(model.alpha_, color='.5')
plt.title('Model selection')
plt.ylabel('Cross-validation score')
plt.xlabel('alpha')
plt.show()

print(np.amax(cov_))
print(np.amin(cov_))

std_devs = np.sqrt(np.diag(cov_))
inv_std_devs = np.reciprocal(std_devs)
D = np.diag(inv_std_devs)
corr_mat = D@emp_cov@D
print("Correlation Matrix: \n", corr_mat)




from joblib import Parallel, delayed
import multiprocessing

# what are your inputs, and what operation do you want to
# perform on each input. For example...
inputs = range(10)
def processInput(i):
    return i * i

num_cores = multiprocessing.cpu_count()
print("How many cores: ", num_cores)

results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
print("Results: ", results)



portfolioValue = techila.peach(funcname="do_scenario",
                               steps=500,
                               files=["bs_function.py","do_scenario.py"],
                               params=['<vecidx>', scenarios, CFs, PCA_t,
                                        PCA1, PCA2, PCA3, S0, K, iv_t, ir_t,
                                        ir_r, ir_displacement, option_Maturity,
                                        volshock_surface, iv_M, isCall,
                                        couponTimes, pos],
                               jobs=nrOfScenarios,
                               stream=True,
                               callback=do_postprocess)



'''
