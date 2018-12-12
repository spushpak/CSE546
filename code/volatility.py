import numpy as np
import pandas as pd
from statsmodels.graphics import tsaplots
import statsmodels
import matplotlib.pyplot as plt
from arch import arch_model   # Fit ARCH models

# Read returns data
df = pd.read_csv("C:/local/sandp500/sp470.csv", parse_dates=True, index_col=0)
print(df.head())

column_names = df.columns
#print("Column names: ", column_names.tolist())
print("Column names: ", column_names)

# returns = df.iloc[:, 0]
# print("1st company: ", returns.head())
# print("How many firms: ", df.shape[1])

# Dataframe to store the volatility
vol_df = pd.DataFrame()

#for i in range(5):
for i in range(df.shape[1]):
    print("Firm number: ", i+1)
    returns = df.iloc[:, i]
    # GARCH(p, q) - Modeling residuals as an ARMA process
    garch_mdl = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1)
    result = garch_mdl.fit(disp='off')
    #print(result.summary())
    garch_vol = result.conditional_volatility
    vol_df = pd.concat([vol_df, garch_vol], axis=1, ignore_index=True)
    print(garch_vol.head())
    #print(vol_df.head())

#vol_df.columns = column_names[0:5]
#vol_df = vol_df.set_axis(column_names[0:5], axis='columns', inplace=False)
vol_df = vol_df.set_axis(column_names, axis='columns', inplace=False)
print("Volatility data shape: ", vol_df.shape)
print("Vol datarame column names: \n", vol_df.columns)
print(vol_df.head())

# Write the GARCH volatilities in a csv file
file_name = 'C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/Project/voldata_sp470.csv'
vol_df.to_csv(file_name)

'''
plt.clf()
plt.plot(garch_vol, label="GARCH(1,1)", color="blue")
plt.xlabel("Time")
plt.ylabel("GARCH volatility")
plt.title("GARCH(1,1) - constant mu")
plt.show()
#plt.savefig("test.png")
'''
