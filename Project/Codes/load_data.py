import os
os.system('CLS')

import numpy as np
#import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error

sns.set()

def trim_values(dframe):
    # Trim whitespace from ends of each column in dataframe
    trim_str = lambda x: x.strip() if type(x) is str else x
    return dframe.applymap(trim_str)


def read_data():
    price_data = pd.read_csv('C:\\GoogleDrivePushpakUW\\UW\\6thYear\\CSE546\\Project\\Data\\sp470_price.csv', index_col=0)

    # Calculate return from price data
    return_data = 100*(np.log(price_data) - np.log(price_data.shift(1)))
    col_names = return_data.columns

    company_list = pd.read_csv('C:\\GoogleDrivePushpakUW\\UW\\6thYear\\CSE546\\Project\\Data\\company_sector_list.csv', index_col=0)
    company_list.sort_index(inplace=True)

    company_list = company_list.iloc[:,0:2]

    company_list = company_list[company_list.index.isin(col_names)]
    company_list = trim_values(company_list)

    col_names = company_list.index
    return_data = return_data.loc[:, return_data.columns.isin(col_names)]
    return_data = return_data.dropna()

    # Read volatility data
    volatility_data = pd.read_csv('C:\\GoogleDrivePushpakUW\\UW\\6thYear\\CSE546\\Project\\Data\\sp470_voldata.csv', index_col=0)
    volatility_data = volatility_data.loc[:, volatility_data.columns.isin(col_names)]
    volatility_data = volatility_data[volatility_data.index.isin(return_data.index)]

    # Read volume data
    volume_data = pd.read_csv('C:\\GoogleDrivePushpakUW\\UW\\6thYear\\CSE546\\Project\\Data\\sp470_volume.csv', index_col=0)
    volume_data = volume_data.loc[:, volume_data.columns.isin(col_names)]
    volume_data = volume_data[volume_data.index.isin(return_data.index)]

    return return_data, volatility_data, volume_data, company_list
