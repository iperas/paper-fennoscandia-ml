#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score


import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning) 

moho=pd.read_csv('moho_stations_deg.csv', sep=';', index_col='Station')
def grid_for_moho(x_min, x_max, y_min, y_max, step):
    arr=[]
    
    for i in range(int((x_max-x_min)/step)+1):
        for j in range(int((y_max-y_min)/step)+1):
            x=x_min+i*step
            y=y_min+j*step
            
            cur=[x,y]
            
            arr.append(cur)
    
    grid=pd.DataFrame(arr)
    grid.columns=['x','y']
    
    return(grid)

def big_circle(a,b):
    r=111.2*np.arccos(np.sin(a[1]*np.pi/180.0)*np.sin(b[1]*np.pi/180.0) +
                      np.cos(a[1]*np.pi/180.0)*np.cos(b[1]*np.pi/180.0)*np.cos((b[0]-a[0])*np.pi/180.0))
    return(r)

grid = grid_for_moho(18.0, 35.0, 60.0, 70.0, 0.1)

X  = moho.drop(['z'],axis=1).reset_index(drop=True)
y  = moho['z'].reset_index(drop=True)
Z  = grid.reset_index(drop=True)

cv = KFold(n_splits=len(X), shuffle=True, random_state=13)
res=[]
for i in range(len(X)-1):
    regr = KNeighborsRegressor(n_neighbors=i+1, weights='distance', metric=big_circle, algorithm='auto')
    mae  =-np.mean(cross_val_score(regr, X, y, cv=cv, n_jobs=-1, scoring='neg_mean_absolute_error'))
    rmse=np.sqrt(-np.mean(cross_val_score(regr, X, y, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')))
    cur=[i+1, mae, rmse]
    res.append(cur)
    
res_df = pd.DataFrame(res)
res_df.columns=['NN', 'MAE', 'RMSE']

plt.plot(res_df.NN, res_df.MAE)

res_df.to_csv('moho_cv_score.csv', sep=';', index=False)
optimal_nn = res_df.set_index('NN').MAE.idxmin()

regr = KNeighborsRegressor(n_neighbors=optimal_nn, weights='distance', metric=big_circle)
regr.fit(X,y)

ans=pd.concat([Z, pd.DataFrame(regr.predict(Z))], axis=1)
ans.columns=['x','y','z']
df_result=pd.pivot_table(ans, values='z', index='y', columns='x')
df_result.columns=np.round(df_result.columns, 2)
df_result.index=np.round(df_result.index, 2)

df_result.to_csv('moho_map_data.csv', sep=';')


