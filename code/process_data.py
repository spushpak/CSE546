import os
os.system('CLS')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error

computer = 'laptop'
#computer = 'TS'

if computer == 'laptop':
    data_file = 'C:/local/sandp500/sp470.csv'
    #var_filename = 'C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/Project/globalsave.pkl'
    ret_fname = 'C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/Project/Data/return_adj.pkl'
    vol_fname = 'C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/Project/Data/vol_adj.pkl'
else:
    data_file = 'H:/local/sandp500/sp470.csv'
    var_filename = 'H:/CSE546/Project/globalsave.pkl'


import dill                            #pip install dill --user
filename = 'C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/Project/globalsave_300.pkl'
#dill.dump_session(filename)

# and to load the session again:
dill.load_session(vol_fname)


price_data = pd.read_csv(data_file, index_col=0)
#index = price_data.index

# Calculate return from price data
ret_data = 100*(np.log(price_data) - np.log(price_data.shift(1)))


print("Coef Mat:\n", coefs)
print("Adj Mat:\n", adj_mat)
print("Shape of adj Mat:\n", adj_mat.shape)
#print("Adj Mat:\n", adj_mat[0:30, 0:30])

np.fill_diagonal(adj_mat, 0)
print("This is diagonal of adjacency matrix:\n", adj_mat.diagonal())

num_firms = adj_mat.shape[0]
columns = price_data.columns[0:num_firms]
print("How many firms in this case: ", num_firms)

degree_nodes = adj_mat.sum(axis = 0)
print("Degree of nodes:\n", degree_nodes)
print("Mean degree: ", np.mean(degree_nodes))
print("Std deviation of degree: ", np.std(degree_nodes))
print("Maximum degree: ", np.amax(degree_nodes))
print("Minimum degree: ", np.amin(degree_nodes))

# Mean degree:  2.91
# Maximum degree:  84.0
# Std deviation of degree:  7.918453131767594
# Minimum degree:  0.0

# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=degree_nodes, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution')
#plt.text(23, 45, r'$\mu=15, b=3$')
#maxfreq = n.max()
# Set a clean upper y-axis limit.
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()


# G = nx.from_numpy_matrix(np.array(coefs))
# nx.draw(G, with_labels=True)

# indx = np.where(degree_nodes == np.amax(degree_nodes))
# print(indx)

indx = degree_nodes.argsort()[::-1]
print("First 20 firms having highest degrees: \n", degree_nodes[indx[0:20]])
print("First 20 firms having highest degrees: \n", columns[indx[0:20]])

not_conn = degree_nodes[np.where(degree_nodes == 0)]
print("No. of firms having no degree: ", len(not_conn))

# First 20 firms having highest degrees:
#  [84. 56. 55. 37. 31. 26. 24. 21. 20. 19. 19. 17. 13. 13. 13. 11. 10. 10.
#  10.  9.]
# First 20 firms having highest degrees:
#  Index(['AMD', 'INCY', 'CHK', 'LNT', 'ARNC', 'FCX', 'EBAY', 'DISCK', 'DISCA',
#        'CF', 'KSS', 'DVN', 'AKAM', 'ILMN', 'CNC', 'HES', 'ALGN', 'KLAC', 'EXC',
#        'KMI'],
# AMD - Advanced Micro Devices Inc - Information Technology
# INCY -
'''
D = np.diag(degree_nodes)
print("Diagonal degree node matrix: \n", D)
print("Shape of Diagonal degree node matrix: \n", D.shape)

L = D - adj_mat
print("Laplace Matrix: \n", L)

w, v = np.linalg.eig(L)
print("Laplacian eigen values in sorted order: ", w[w.argsort()])
print("Laplacian eigen vector: ", v)
'''
