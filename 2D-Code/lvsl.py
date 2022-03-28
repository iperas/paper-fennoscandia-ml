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

def big_circle(a,b):
    r=111.2*np.arccos(np.sin(a[1]*np.pi/180.0)*np.sin(b[1]*np.pi/180.0) +
                      np.cos(a[1]*np.pi/180.0)*np.cos(b[1]*np.pi/180.0)*np.cos((b[0]-a[0])*np.pi/180.0))
    return(r)

def class_by_proba_f(x):
    if(x<0.45):
        x=0
    if (x>0.55):
        x=1
    if(x>=0.45 and x<=0.55):
        x=0.5
    
    return(x)

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
    
# ### Vs low layer map
moho_class=pd.read_csv('moho_class_ext2.csv', sep=';', index_col='Station')
grid = grid_for_moho(18.0, 35.0, 60.0, 70.0, 0.1)

X = moho_class.drop(['class'],axis=1).reset_index(drop=True)
y = moho_class['class'].reset_index(drop=True)
Z = grid.reset_index(drop=True)

cv = KFold(n_splits=len(X), shuffle=True, random_state=13)

res2 = []
for i in range(len(X)-1):
    clf=KNeighborsClassifier(n_neighbors=i+1, weights='distance', metric=big_circle)
    store=[]
    for train, test in cv.split(X,y):
        clf.fit(X.loc[train], y.loc[train])
        store.append([test[0], np.round(clf.predict_proba(X.loc[test])[0:,1],3)])
        
        df_score=pd.DataFrame(store)
        df_score.columns=['Station', 'pred_cls']
        df_score.set_index('Station', inplace=True)
        joined_df=pd.DataFrame(y).join(df_score, how='left')
    
    ra=np.round(roc_auc_score(joined_df['class'], joined_df['pred_cls']),3)
    cur=[i+1,ra]
    res2.append(cur)
    
res2_df=pd.DataFrame(res2)
res2_df.columns=['NN', 'ROC_AUC']

plt.plot(res2_df.NN, res2_df.ROC_AUC)
optimal_nn2=res2_df.set_index('NN').ROC_AUC.idxmax()

res2_df.to_csv('res2_df.csv', index=False, sep=';')
#optimal_nn2=4
clf=KNeighborsClassifier(n_neighbors=optimal_nn2, weights='distance', metric=big_circle)

clf.fit(X,y)

ans=pd.concat([Z, pd.DataFrame(clf.predict_proba(Z)[:,1])], axis=1)
ans.columns=['x','y','class']

ans['class_by_proba']=ans['class'].apply(lambda x: class_by_proba_f(x))

df2_result=pd.pivot_table(ans, values='class_by_proba', index='y', columns='x')
df2_result.columns=np.round(df2_result.columns, 2)
df2_result.index=np.round(df2_result.index, 2)

df2_result.to_csv('Vs_low_layer_map_ext_class_nn4_v10.csv', sep=',')


