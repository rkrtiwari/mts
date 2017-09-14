# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 08:22:16 2017

@author: ravitiwari
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import os
from sklearn.cluster import KMeans

# change to required directory
os.chdir('C:\\Users\\ravitiwari\\Documents\\miscellaneous')


def plot_location(lat, lon, margin = 0.5):
    # determine range to print based on min, max lat and lon of the data
    lat_min = min(lat) - margin
    lat_max = max(lat) + margin
    lon_min = min(lon) - margin
    lon_max = max(lon) + margin
    
    # create map using BASEMAP
    m = Basemap(llcrnrlon=lon_min,
            llcrnrlat=lat_min,
            urcrnrlon=lon_max,
            urcrnrlat=lat_max,
            lat_0=(lat_max - lat_min)/2,
            lon_0=(lon_max-lon_min)/2,
            projection='merc',
            resolution = 'h',
            area_thresh=10000.,
            )
    
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.drawmapboundary(fill_color='#46bcec')
    m.fillcontinents(color = 'white',lake_color='#46bcec')
    
    # convert lat and lon to map projection coordinates
    lons, lats = m(lon, lat)
    # plot points as red dots
    m.scatter(lons, lats, marker = 'o', color='r', zorder=2)
    plt.show()
    
    
       
# read in data to use for plotted points
location_data = pd.read_csv('locations.csv', header = None)
lat = location_data[0].values
lon = location_data[1].values
margin = 2 # buffer to add to the range

plot_location(lat, lon, margin = 2)
# clearly there are three clusters in  the data so I separate them first

lat_long = location_data.values
kmeans = KMeans(n_clusters = 3, n_init = 10).fit(lat_long)
pd.value_counts(kmeans.labels_)

ind_0 = kmeans.labels_ == 0
ind_1 = kmeans.labels_ == 1
ind_2 = kmeans.labels_ == 2

kmeans.labels_.tolist().index(1)
kmeans.labels_.tolist().index(2)

# points belonging to label 0
location_data_0 = location_data.iloc[ind_0, :]
lat = location_data_0[0].values
lon = location_data_0[1].values
plot_location(lat, lon, margin = 20)



# points belonging to label 1
location_data_1 = location_data.iloc[ind_1, :]
lat = location_data_1[0].values
lon = location_data_1[1].values
plot_location(lat, lon, margin = 20)
location_data_1


# points belonging to label 2
location_data_2 = location_data.iloc[ind_2, :]
lat = location_data_2[0].values
lon = location_data_2[1].values
plot_location(lat, lon, margin = 20)
location_data_2








