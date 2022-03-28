#!/usr/bin/python3

# Horizontal slice calculation
# Input data file name: sveka_xyz_vs_0p10.asc
# Output: sveka_Vs_horz_0_40_40.cvs
# Example output: sveka_Vs_horz_0_40_40.png

import matplotlib.pyplot as plt
#import numpy as np
from numpy import cos
from numpy import pi
from numpy import round
from numpy import sqrt

import pandas as pd
import seaborn as sns

from sklearn.neighbors       import KNeighborsRegressor

grad2rad = pi / 180.0
grad2km = 111.195


class point2D:
    def __init__(self,lon,lat):
        self.lon = lon
        self.lat = lat
        self.x = 0.0
        self.y = 0.0
    
    def to_kartesian(lon1,lat1,lat0):
        self.x = (self.lon - lon1) * cos(lat0*grad2rad) * grad2km
        self.y = (self.lat - lat1) * grad2km 
    
    def to_geo(lon1,lat1,lat_0):
        self.lon = lon1 + self.x / grad2km / cos(lat0*grad2rad)
        self.lat = lat1 + self.y / grad2km
    
def lat(y,lat1): return(lat1 + self.y / grad2km)
def lon(x,lon1,lat0): return(lon1 + x / grad2km / cos(lat0*grad2rad))

def horz_grid_generation(z,p1,p2, nx, ny):
    step_x = abs(p2.x - p1.x) / (nx - 1.0)
    step_y = abs(p2.y - p1.y) / (ny - 1.0)
    
    if(step_x <= 0) : print('warning: step x = ', step_x)
    if(step_y <= 0) : print('warning: step x = ', step_y)
    
    grid=pd.DataFrame()
    for i in range(nx):
        for j in range(ny):
            x = p1.x + i*step_x
            y = p1.y + j*step_y 
            
            current_string = [x,y,z]
            grid = pd.concat([grid, pd.DataFrame(current_string).T], axis=0)
    
    grid.columns=['x','y','z']
    grid.reset_index(drop=True, inplace=True)
    return(grid)

regname = 'sveka'
fname   = 'sveka_xyz_vs_0p10.asc'
varname = 'Vs'

#z = 15.0
z =  0.0

p1 = point2D(22.0, 60.0)
p2 = point2D(32.0, 67.0)

lat0 = 63.5
lon1 = 22.0
lat1 = 60.0

p1.x = (p1.lon - lon1) * cos(lat0*grad2rad) * grad2km
p2.x = (p2.lon - lon1) * cos(lat0*grad2rad) * grad2km
p1.y = (p1.lat - lat1) * grad2km 
p2.y = (p2.lat - lat1) * grad2km 

nb    = 40
scale = 40

nx = 301
ny = 301

print('read data')
train_grid = pd.read_csv(fname, sep=' ')
print('done')
P = train_grid.drop(['Vs'],axis=1).reset_index(drop=True) # P - coordinates of observed points (X->P)
v = train_grid['Vs'].reset_index(drop=True)               # v - values at the observed points (y->v)

P['x'] = train_grid['x'] / scale
P['y'] = train_grid['y'] / scale

print('generating grid')
horz_cs = horz_grid_generation(z,p1,p2,nx,ny)
print('done')

Q = horz_cs.reset_index(drop=True) 
Q['x'] = horz_cs['x'] / scale         
Q['y'] = horz_cs['y'] / scale         
Q['z'] = horz_cs['z']                 

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

result['lon'] = lon1 + result['x'] / (grad2km * cos(lat0*grad2rad))
result['lat'] = lat1 + result['y'] / grad2km 
print('done')


print('plot')
horz_cs_table = pd.pivot_table(result, values='Vs', index='lat', columns='lon')
horz_cs_table.columns = round(horz_cs_table.columns, 2)
horz_cs_table.index   = round(horz_cs_table.index, 2)

plt.figure(figsize=(5,10))
plt.title('Horizontal crosssection', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)
sns.set(font_scale=1.5)
outfile_name = regname+"_"+varname+'_horz_'+str(int(z))+'_'+str(nb)+'_'+str(scale)
result.reset_index(drop=True).to_csv(outfile_name+'.csv')
print('done')


quit()

