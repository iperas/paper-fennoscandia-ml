# Macrofracturing in the Fennoscandia upper crust as evidence of the last deglaciation derived from machine learning analysis of the receiver function data

This repository contains data and code used for the research.

# 2D-Code #
**2D-Code** contains scipts and input data for both Moho map computation (moho.py) and map of low S-velocity layer presence (lvsl.py).

**Input data**

* moho_stations_deg.csv - list of stations in the following format: `StationCode;Longitude(degrees);Latitude(degrees);MohoDepth(km)`
* moho_class_ext2.csv - list of stations in the following format:
`StationCode;Longitude(degrees);Latitude(degrees);0/1`

**Output**

*moho.py*

* moho_cv_score.csv - Mean absolute error (MAE) and root mean squared error (RMSE) vs. neighboor count (NN): `NN;MAE;RMSE`

* moho_map_data.csv - Moho depth values with rows as X and columns as Y. First column are Xs.

*lvsl.py*

res2_df.csv - ROC AUC value vs. neighboor count (NN): `NN;ROC_AUC`

Vs_low_layer_map_ext_class_nn4_v10.csv - Low S-velocity layer presence map

# 3D-Code #

**Input data**
* sveka_xyz_vs_0p10.asc - Vs values under the stations: `x y Depth Vs`

* rftomo-hyp.py - generates root mean squared error (RMSE) for various NN and Scale parameter values. Outputs sveka_Vs_rmse.csv `n,Scale,NNb,RMSE`

* rftomo-horz.py - generates horizontal profiles for various NN and Scale parameter values. Outputs sveka_Vs_horz_0_40_40.csv `n,x,y,z,Vs,lon,lat`

* rftomo-vert.py - generates vertical profiles for various NN and Scale parameter values between given points A(lon1,lat1) and B(lon2,lat2). Units are degrees. Outputs sveka_Vs_vert_J10-J01.csv `n,x,y,z,Vs,l` where l is distance from A to current point in km.








