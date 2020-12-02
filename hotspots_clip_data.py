# -*- coding: utf-8 -*-
"""
Clip GPKG Files to Area of Interest

@author: Johanna Roll
"""
#import packages 
import glob
import os
import geopandas as gpd

#set datapaths
#directory with all data
path_data = os.path.dirname(os.path.realpath('hotspots_clip_data.py')) + '\\'
#path_data = ('D:\johanna_roll\h3_hotspots\germany_1995_2019\\')

#path to area of interest
#path_area = ('D:\johanna_roll\h3_hotspots\gadm36_DEU_shp\gadm36_DEU_0.shp')
path_area = glob.glob(os.path.join(path_data, '*.shp'))
path_area = str(path_area[0])

#directory for output
path_out = path_data + ('data_clipped\\')

if not os.path.exists(path_out):
    os.makedirs(path_out)

#get paths of all data
path_list = glob.glob(os.path.join(path_data, '*.gpkg'))

#read shape of area
area_shape = gpd.read_file(path_area)

#read daa
for path in path_list:
    #get dataname
    data_name = path.split('\\')[-1][:-5]
    #year
    dataname_split = data_name.split('_')
    area = dataname_split[0]
    data_date = dataname_split[-1]	
        
    #read file
    data = gpd.read_file(path)
    crs = data.crs
    print ('Processing: ', data_name)
        
    #clip data to Germany only (optional)
    data_clipped = gpd.overlay(data, area_shape, how='intersection')
    data_clipped.crs = crs
    
    #remove added columns of clipping shape
    col_o = data.columns.tolist()
    col_n = data_clipped.columns.tolist()
    col_dif = list(set(col_n) - set(col_o))

    for col in col_dif:
        del data_clipped[col]
    
    #export clipped file
    data_clipped.to_file((path_out + area + '_clip_' + data_date + '.gpkg'), driver = 'GPKG')