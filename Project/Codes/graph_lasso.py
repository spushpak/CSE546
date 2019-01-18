from __future__ import division
import os
os.system('CLS')
import numpy as np
from scipy import linalg
from numpy.linalg import inv,pinv
from scipy.optimize import minimize
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
import statsmodels.api as sm
import load_data

def calc_optPf(X, inv_covar, covar, num_firms):
    one_vec = [1]*num_firms
    one_vec = np.array(one_vec)
    one_vec = one_vec.reshape(-1, 1)

    numerator = inv_covar@one_vec
    denominator = one_vec.T@inv_covar@one_vec

    opt_weights = numerator/denominator

    avg_ret = np.mean(X, axis=0)
    avg_ret = np.array(avg_ret)
    avg_ret = avg_ret.reshape(1,-1)

    pf_ret = avg_ret@opt_weights
    pf_var = opt_weights.T@covar@opt_weights
    sharpe_ratio = pf_ret / np.sqrt(pf_var)

    return sharpe_ratio


ret_data, vol_data, comp_list = load_data.read_data()

# Take first 1000 rows (days) as training set and rest as test set
train_index = 1000
X_train = ret_data.iloc[0:train_index, :]
X_test = ret_data.iloc[train_index:, :]
#X = (X - X.mean())/X.std()

num_firms = ret_data.shape[1]
print("Number of firms: ", num_firms)

cov, corr = X_train.cov(), X_train.corr()
#print(cov)

model = GraphicalLassoCV(cv=5)
model.fit(X_train)
cov_ = model.covariance_
prec_ = model.precision_

print("What are the alphs in CV: ", model.cv_alphas_)
print("Optimal lambda parameter for graphical LASSO: ", model.alpha_)

#lw_cov_, _ = ledoit_wolf(X_train)
#lw_prec_ = linalg.inv(lw_cov_)

# print("Lasso Covariance Matrix:\n", cov_)
prec_ = model.precision_
# print("Lasso Inverse Covariance Matrix:\n", prec_)

# print("Empirical Covariance Matrix:\n", cov)
# Find inverse covariance matrix of empirical covariance
emp_inv_cov = np.linalg.solve(cov, np.eye(num_firms))
# print("Empirical inverse covariance matrix:\n", emp_inv_cov)


# # plot the model selection metric
# plt.figure(figsize=(4, 3))
# plt.axes([.2, .15, .75, .7])
# plt.plot(model.cv_alphas_, np.mean(model.grid_scores_, axis=1), 'o-')
# plt.axvline(model.alpha_, color='.5')
# plt.title('Model selection')
# plt.ylabel('Cross-validation score')
# plt.xlabel('alpha')
# plt.show()

# Plot the results
plt.figure(figsize=(4, 3))
plt.subplots_adjust(left=0.02, right=0.98)

# plot the covariances
covs = [('Empirical', cov), ('GraphicalLassoCV', cov_)]
vmax = cov_.max()
for i, (name, this_cov) in enumerate(covs):
    plt.subplot(2, 2, i + 1)
    plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
               cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('%s covariance' % name)

# plot the precisions
precs = [('Empirical', emp_inv_cov), ('GraphicalLasso', prec_)]
vmax = .9 * prec_.max()
for i, (name, this_prec) in enumerate(precs):
    ax = plt.subplot(2, 2, i + 3)
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

################################################################
# # USER INPUT
# V = cov_
# R = X/100
# rf = 0.0
#
# w0= [1/d]*d
#
# # min var optimization
# def calculate_portfolio_var(w,V):
#     w = np.matrix(w)
#     return (w*V*w.T)[0,0]
#
# # unconstrained portfolio (only sum(w) = 1 )
# cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1.0})
# res= minimize(calculate_portfolio_var, w0, args=V, method='SLSQP',constraints=cons)
# print("Optimal weights: ", res)
#
# w_g = res.x
# print(w_g)
#
# mu_g = w_g*R
# print(mu_g)
#
# var_g = np.dot(w_g*V, w_g)
# print(var_g)

#############################################################################
test_glasso_SR = calc_optPf(X_test, prec_, cov_, num_firms)
print("Sharpe ratio from Graphical LASSO: ", test_glasso_SR)

test_emp_SR = calc_optPf(X_test, emp_inv_cov, cov, num_firms)
print("Sharpe ratio from empirical covariance: ", test_emp_SR)

#####
# train_glasso_SR = calc_optPf(X_train, prec_, cov_, num_firms)
# print("Sharpe ratio from Graphical LASSO: ", train_glasso_SR)
#
# train_emp_SR = calc_optPf(X_train, emp_inv_cov, cov, num_firms)
# print("Sharpe ratio from empirical covariance: ", train_emp_SR)


# one_vec = [1]*d
# one_vec = np.array(one_vec)
# one_vec = one_vec.reshape(-1, 1)
#
# numerator = prec_@one_vec
# #numerator = np.diag(numerator)
# print("Numerator:\n", numerator)
# print("Numerator shape:\n", numerator.shape)
#
# denominator = one_vec.T@prec_@one_vec
# #denominator = np.diag(denominator)
# print("Denominator:\n", denominator)
# #print("Denominator length:\n", len(denominator))
#
# opt_weights = numerator/denominator
# print("Optimal weights:\n", opt_weights)
# print("Sum of optimal weights:\n", np.sum(opt_weights))
#
# avg_ret = np.mean(X, axis=0)
# avg_ret = np.array(avg_ret)
# avg_ret = avg_ret.reshape(1,-1)
# print("Avg ret: ", avg_ret.shape)
#
# pf_ret = avg_ret@opt_weights
# print("Portfolio return: ", pf_ret)
#
# pf_var = opt_weights.T@cov_@opt_weights
# print("Shape of pf var: ", pf_var.shape)
# print("pf variance: ", pf_var)
# print("Portfolio sharpe ratio: ", pf_ret / np.sqrt(pf_var))

'''cov_single_index
# Use cvpxy for to solve for optimal weights
n = num_firms
mu = np.mean(X_train, axis=0)
Sigma = cov


w_NS = cp.Variable(n)
#gamma = cp.Parameter(nonneg=True)
#ret = mu.T*w_NS
risk = cp.quad_form(w_NS, Sigma)
prob = cp.Problem(cp.Minimize(risk), [cp.sum(w_NS) == 1, w_NS >= 0])   # no short sale - weights non negative
prob.solve()
#print("Optimal weights - no Short sales: \n", w_NS.value.shape)
opt_wts = np.array(w_NS.value)
opt_wts = opt_wts.reshape(-1, 1)

#print("np sum:\n", np.sum(opt_wts))
#print("Shape of opt wts: ", opt_wts.shape)

avg_ret = np.mean(X_test, axis=0)
avg_ret = np.array(avg_ret)
avg_ret = avg_ret.reshape(1,-1)
#print("Shape of avg ret: ", avg_ret.shape)

pf_ret = avg_ret@opt_wts
pf_var = opt_wts.T@Sigma@opt_wts
print("Portfolio variance - no Short sales: ", pf_var)
sharpe_ratio = pf_ret / np.sqrt(np.asscalar(pf_var))
print("Sharpe ratio from no Short sales portfolio: ", sharpe_ratio)

###############################################################################
print("\n\n\n")
###############################################################################

w_LS = cp.Variable(n)
risk = cp.quad_form(w_LS, Sigma)
prob = cp.Problem(cp.Minimize(risk), [cp.sum(w_LS) == 1])   # no short sale - weights non negative
prob.solve()
#print("Optimal weights - Long-Short: \n", w_LS.value.shape)
opt_wts = np.array(w_LS.value)
opt_wts = opt_wts.reshape(-1, 1)

#print("np sum:\n", np.sum(opt_wts))
#print("Shape of opt wts: ", opt_wts.shape)

avg_ret = np.mean(X_test, axis=0)
avg_ret = np.array(avg_ret)
avg_ret = avg_ret.reshape(1,-1)
#print("Shape of avg ret: ", avg_ret.shape)

pf_ret = avg_ret@opt_wts
pf_var = opt_wts.T@Sigma@opt_wts
print("Portfolio variance - Long-Short: ", pf_var)
sharpe_ratio = pf_ret / np.sqrt(np.asscalar(pf_var))
print("Sharpe ratio from Long-Short portfolio: ", sharpe_ratio)
'''


# Estimate the covariance matrix based on the single index model
total_indx_data = pd.read_csv('C:\\GoogleDrivePushpakUW\\UW\\6thYear\\CSE546\\Project\\Data\\SP500_total_ret.csv', index_col=0)
total_indx_data.index = pd.to_datetime(total_indx_data.index)
print(total_indx_data.index)

# Calculate return from price data
total_return = 100*(np.log(total_indx_data) - np.log(total_indx_data.shift(1)))
total_return.columns = ['ret']
#total_return.index = total_return.index.strftime('%Y-%m-%d')
total_return = total_return.dropna()
print(total_return.shape)
#print(total_return.head())
total_return = total_return[total_return.index.isin(ret_data.index)]
#print(total_return.head())

# print(ret_data.index)
# newret_data = ret_data.loc[total_return.index]
# print(newret_data.shape)

comb_data = pd.merge(total_return, ret_data, left_index=True, right_index=True)
print(comb_data.head())
print(comb_data.shape)


residual = []
yy = comb_data.iloc[:, 0]
for k in range(1, 444):
#for k in range(1, 5):
    #print("k: ", k)
    xx = comb_data.iloc[:, k]
    xx = sm.add_constant(xx)
    model = sm.OLS(yy, xx).fit() ## sm.OLS(output, input)
    predictions = model.predict(xx)
    res = yy - predictions
    #print(res.shape)
    residual.append(res)

residual = np.array(residual)
print(residual.shape)

cov_single_index = residual@residual.T
print(cov_single_index.shape)

si_inv_cov = np.linalg.solve(cov_single_index, np.eye(num_firms))
test_si_SR = calc_optPf(X_test, si_inv_cov, cov_single_index, num_firms)
print("Sharpe ratio from Single Index covariance: ", test_si_SR)


plt.rcdefaults()
objects = ('Empirical Cov', 'SingleIndex Cov', 'GraphLASSO Cov')
y_pos = np.arange(len(objects))
performance = [np.asscalar(test_emp_SR), np.asscalar(test_si_SR), np.asscalar(test_glasso_SR)]

plt.bar(y_pos, performance, width = 0.1, align='center', alpha=0.8)
plt.xticks(y_pos, objects)
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe Ratio: Empirical, Single Index, Graph LASSO Covariace')
plt.show()
