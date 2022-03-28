#!/usr/bin/python3

# Vertical profile calculation
# Input data file name: sveka_xyz_vs_0p10.asc
# Output: sveka_Vs_vert_J10-J01.cvs
# Example output: sveka_Vs_vert_J10-J01.png

import matplotlib.pyplot as plt
#import numpy as np
#from numpy import mean
from numpy import cos
from numpy import pi
from numpy import round
from numpy import sqrt

import pandas as pd
import seaborn as sns

from sklearn.neighbors       import KNeighborsRegressor

re = 6371.009
grad2rd = pi/180.0
grad2km = re * pi/180.0

class point3D:
    def __init__(self,x,y,z):
        self.lon = x
        self.lat = y
        self.z   = z
        self.x = 0.0
        self.y = 0.0

def vert_grid_generation(p1,p2, nh, nz):
    bx = p1.x
    by = p1.y
    kx = p2.x - p1.x
    ky = p2.y - p1.y
    
    z_min = p1.z # Profile depth
    z_max = p2.z
    
    step_h = 1.0 / (nh - 1.0)
    step_z = (z_max - z_min) / (nz - 1.0)
    
    grid=pd.DataFrame()
    for i in range(nh):
        for j in range(nz):
            x = bx + kx * i*step_h # stight line in the parametric form
            y = by + ky * i*step_h 
            
            z = z_min + j*step_z
            current_string = [x,y,z]
            grid = pd.concat([grid, pd.DataFrame(current_string).T], axis=0)
    
    grid.columns=['x','y','z']
    grid.reset_index(drop=True, inplace=True)
    return(grid)

regname = 'sveka'
fname   = 'sveka_xyz_vs_0p10.asc'
varname = 'Vs'
profile_name = 'J10-J01'

#22.5, 62.5
#28.6, 66.0
lat0 = 63.5
lon1 = 22.0
lat1 = 60.0

p1 = point3D(22.5, 62.5,  0.0)
p2 = point3D(28.6, 66.0, 15.0)

p1.x = (p1.lon - lon1) * grad2km * cos(lat0*grad2rd)
p2.x = (p2.lon - lon1) * grad2km * cos(lat0*grad2rd)
p1.y = (p1.lat - lat1) * grad2km
p2.y = (p2.lat - lat1) * grad2km

nb    = 40
scale = 40

nh =  61
nz = 301

print('read data')
train_grid = pd.read_csv(fname, sep=' ')
print('done')
P = train_grid.drop(['Vs'],axis=1).reset_index(drop=True) # P - coordinates of observed points (X->P)
v = train_grid['Vs'].reset_index(drop=True)               # v - values at the observed points (y->v)

P['x'] = train_grid['x'] / scale
P['y'] = train_grid['y'] / scale

print('generating grid')
vert_cs = vert_grid_generation(p1,p2,nh,nz)
print('done')

Q = vert_cs.reset_index(drop=True) 
Q['x'] = vert_cs['x'] / scale         
Q['y'] = vert_cs['y'] / scale         
Q['z'] = vert_cs['z']                 

# P - observed data
P['x'] = train_grid['x'] / scale
P['y'] = train_grid['y'] / scale

print('learning')
regr = KNeighborsRegressor(n_neighbors=nb, weights='distance') # add arg to change distance
regr.fit(P,v)
print('done')

print('interpolating')
result = pd.concat([Q, pd.DataFrame(regr.predict(Q))], axis=1)
result.columns=['x','y','z','Vs']
result['x'] = result['x'] * scale
result['y'] = result['y'] * scale
result['l'] = sqrt((result['x'] - p1.x)**2 + (result['y'] - p1.y)**2)
print('done')


print('plot')
vert_cs_table = pd.pivot_table(result, values='Vs', index='z', columns='l')
vert_cs_table.columns = round(vert_cs_table.columns, 2)

plt.figure(figsize=(5,10))
plt.title('Vertical crosssection', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)
#sns.set(font_scale=1.5)
outfile_name = regname+"_"+varname+'_vert_'+profile_name
sns.heatmap(vert_cs_table.sort_index(ascending=True), cmap = 'jet_r',
    xticklabels=10, yticklabels=10).figure.savefig(outfile_name+'.png')

#vert_cs_table.to_csv('vert_cs_table.csv')
result.reset_index(drop=True).to_csv(outfile_name+'.csv')
print('done')

plt.show()
quit()

