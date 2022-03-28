#!/usr/bin/python3

# Detrmination of hyperparameters value
# Input data file name: sveka_xyz_vs_0p10.asc
# Output: sveka_Vs_rmse.cvs
# Example output: sveka_Vs_rmse.png

import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from numpy import sqrt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors       import KNeighborsRegressor

import time

#def dist_plain(x1,y1,z1,x2,y2,z2,scale) : return sqrt((z2 - z1)^2 + ((x2 - x1)^2 + (y2 - y1)^2) / scale)

regname = 'sveka'
fname = 'sveka_xyz_vs_0p10.asc'

varname = 'Vs'

nb_min =  2
nb_max = 152
nb_step=  2
sc_min =  2
sc_max = 152
sc_step=  2

print('read data')
train_grid = pd.read_csv(fname, sep=' ')
X = train_grid.drop(['Vs'],axis=1).reset_index(drop=True)
y = train_grid['Vs'].reset_index(drop=True)
print('done')

cv = KFold(n_splits=5, shuffle=True, random_state=13)

# Metric
print('metric')
start_time = time.time()

nbrange = list(range(nb_min,nb_max,nb_step))
scales  = list(range(sc_min,sc_max,sc_step))
result  = pd.DataFrame()
for scale in scales:  
    X['x'] = train_grid['x'] / scale
    X['y'] = train_grid['y'] / scale
    
    for i in nbrange:
        regr = KNeighborsRegressor(n_neighbors = i, weights='distance')
        mse  = -mean(cross_val_score(regr, X, y, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error'))
        current_row = pd.DataFrame([scale, i, sqrt(mse)]).T
        result = pd.concat([result, current_row], axis=0)

result.columns=['Scale', 'NNb', 'RMSE']
result.reset_index(drop=True, inplace=True)
    
fname = regname+'_'+varname+'_rmse'
result.to_csv(fname+'.csv')

quit()

