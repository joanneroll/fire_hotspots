# -*- coding: utf-8 -*-
"""
Fire Hotspot Analysis

SPATIAL CLUSTERING ANALYSIS

Clustering data in sight of
    - geographical location
    - adapted script "hotspots_spatial_clulstering_smalldata" for data with small amount of features
    - still in need of debugging regarging dbsacn clustering

Used methods for processing data: 
    - extraction of centroids of each polygon and use of point shapes for clustering
    - kmeans clustering with automatized selection of optimal number of clusters based on Average Silhouette method
        - with kmeans clustering, the overall distribution of detected hotspots is visualized
    - dbscan clustering with choice of epsilon based on statistical analysis of distances between the points; set minimum of neighbors (6) 
        - with dbscan clustering, dense regions with detected hotspots are highlighted 
        - dense regions are defined by a maximal distance between features labelled as neighbors
        - features outside this distance are considered as 'noice' (label -1)
        - as this script only clusters based on geographical information, it does not take into account other attributes of the detections
        - combining it with number of detections, intensity, ... is a further approach which should be pursued


Output
for each datafile (year/date) 
    - Scatterplot of clusters
    - Map with clusters 
    - GeoPackage of Centroids with label clusters as attributes
    
Textfile with information to each processed cluster   

@author: Johanna Roll
"""
#import packages
import glob
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import random
from haversine import haversine
from shapely.geometry import Point

import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from scipy.spatial.distance import pdist

##############################################################################################
#set datapaths
#directory with all data
#path_data_orig = ('D:\johanna_roll\h3_hotspots\germany_selection\\')
path_data_orig = os.path.dirname(os.path.realpath('hotspots_spatial_clustering')) + '\\'
#path_data_orig = ('D:\johanna_roll\h3_hotspots\germany_1995_2019\\')
path_data = path_data_orig + ('data_clipped\\')

#if no clipped data exists, original data is data
if not os.path.exists(path_data):
    path_data = path_data_orig

#get paths of all data
path_list = glob.glob(os.path.join(path_data, '*.gpkg'))

#path to area of interest
#path_area = ('D:\johanna_roll\h3_hotspots\gadm36_DEU_shp\gadm36_DEU_0.shp')
path_area = glob.glob(os.path.join(path_data_orig, '*.shp'))
path_area = str(path_area[0])

#directory for outputs
path_out = path_data + ('output\\')
if not os.path.exists(path_out):
    os.makedirs(path_out)
    
path_clustering = path_out + ('spatial_clustering\\')
if not os.path.exists(path_clustering):
    os.makedirs(path_clustering)
	
path_centroids = path_clustering + ('centroids_shapefiles\\')
if not os.path.exists(path_centroids):
    os.makedirs(path_centroids)
    
##############################################################################################
#function definitions        
def write_cluster_info (output_file, cluster_labels):
    #number of labelled features
    n_features = len(cluster_labels)
    num_clusters = len(set(cluster_labels))

    output_file.write('{f} features are assigned to {c} cluster\n'.format(f=n_features, c=num_clusters))  
    
    labels = np.unique(cluster_labels)
    label_min, label_max = int(labels.min()), int(labels.max())
    
    output_file.write('Features per Cluster:\n')
    for i in range (label_min, label_max+1):
        count = len(cluster_labels[cluster_labels == i])
        output_file.write('Cluster {l}: {c}\n'.format(l=i, c=count))        
        

def compute_distance_matrix(X):
    distance = pdist(X[:, :2], lambda x, y: haversine(x, y))  
    
    return distance
#############################################################################################
# Setup data dictionary
data_list = {}

# Read data
for path in path_list:
    #get dataname
    data_name = path.split('\\')[-1][:-5]
    #read file
    data = gpd.read_file(path)
    data.name = data_name
    #add file to dictionary with respective filename as key
    data_list[data_name] = data

# Read area of interest
area_shape = gpd.read_file(path_area)

# Turn off interactive plotting 
plt.ioff()

###########################################################################################
# Prepare metafile 
meta_out = open((path_clustering + 'spatial_clustering_info.txt'), 'w')
meta_out.write('Spatial Clustering of Fire Hotspot Data - Parameter:\n')


### Iterate over all data and generate clusters
for date in data_list:
    data = data_list.get(date)
	# Collect Metainfo
    crs = data.crs
    dataname_split = date.split('_')
    area = dataname_split[0].upper()
    data_date = dataname_split[-1]	
    
    
    
    print('Processing: ', area, data_date)
	
	## Prepare data for clustering
	
	# Convert polygons to points: Extract centroids from polygons
	# New GeoDataFrame with columns from original data
    col = data.columns.tolist()
    data_centroids = gpd.GeoDataFrame(columns = col)
    data_centroids.crs = crs
	# Extract nodes and attribute values of dataframe to new one
    for index, row in data.iterrows():
        for pt in list(row['geometry'].centroid.coords):
            data_centroids = data_centroids.append({'h3index': row['h3index'], 'indexesFound': int(row['indexesFound']), 
                                                    'frpMWkm2Avg': float(row['frpMWkm2Avg']), 'tstart': row['tstart'],
                                                    'tend': row['tend'],'geometry': Point(pt) }, ignore_index=True)
	
	# Extraction of x y coordinates from geometry of type point
    x = pd.Series(data_centroids['geometry'].apply (lambda p: p.x))
    y = pd.Series(data_centroids['geometry'].apply (lambda p: p.y))

	# Stack with coordinates of all features of data 
    X = np.column_stack((x, y))   
    
	# Distance Matrix of all features (insight for selecting epsilon DBSCAN)
    # Coordinates as radians
    distance_matrix_centroids = compute_distance_matrix(np.radians(X))
    	
    # # Plotting histogram of distances
	# dist_array = np.asarray(distance_matrix_centroids)
	# plt.title('Histogram of distance matrix - {d}'.format(d = data_date), pad = 15)
    # sns.distplot(dist_array.ravel(), bins = 70, kde = False)
	# plt.show()

	####################################################################
	## Clustering
	# KMeans
	# Determine optimal number of clusters (Average silhouette method) 
    # Measures how well each object lies within its cluster - high average silhouette width indicates a good clustering
    
#    # Test number of clusters from range 2 to 15
#    silhouette = {}
#    
#    for n_cluster in range (2,15):
#        kmeans = KMeans(n_clusters = n_cluster, init = 'k-means++', random_state = 42)
#        y_kmeans = kmeans.fit_predict(X)
#        silhouette[n_cluster] = silhouette_score(X, y_kmeans)
#        
#    # Plot 
#    plt.plot(range(len(silhouette)), list(silhouette.values()),  'bx-')
#    plt.xticks(range(len(silhouette)), list(silhouette.keys()))
#    plt.title('silhouette average - {d}'.format(d = data_date))
#    plt.xlabel('number of clusters')
#    plt.ylabel('silhouette average')
#    plt.savefig((path_clustering + data_date + '_silhouette.png'), dpi = 500)
#    plt.show()
#    
#    #optimal number of cluster based on this method with highest silhouette score
#    best_cluster_n = max(silhouette, key = silhouette.get)	
#    number_cluster_kmeans = best_cluster_n
    
        # set number of clusters
    if len(X) >= 5:
        number_cluster_kmeans = 3
    else:
        number_cluster_kmeans = len(X)
    
	# init 'k-means++' for speeding up convergence; random state for deterministic randomness for centroid initialization
    kmeans = KMeans (n_clusters = number_cluster_kmeans, init = 'k-means++', random_state = 42, max_iter = 400)
    # Compute cluster centers and predict cluster index for each sample
    y_kmeans = kmeans.fit_predict(X)
	
	# Add cluster label to data
    columname_kmeans = 'spatial_cluster_kmeans_n_' + str(number_cluster_kmeans)
    cluster_k = pd.DataFrame(y_kmeans, columns = [columname_kmeans])
    data_centroids = data_centroids.join(cluster_k)
	
	# Plot result
	# Plot cluster with its center
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(1, figsize=(12,9))
    area_shape.plot(ax = ax, color = 'white', edgecolor='grey', linewidth = 1)
    data_centroids.plot(column = columname_kmeans, ax = ax, cmap = 'Set1', markersize = 7)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, marker='x', 
				c='#111111', label='Centroids')
    plt.suptitle('Detected Hotspots - {a} {y} '.format(a = area, y = data_date), fontsize = 20)
    plt.title('KMEANS Spatial Clustering ({n} Cluster)'.format(n = number_cluster_kmeans), fontsize = 15, pad=15)
    plt.legend()
    ax.set_axis_off()
    plt.savefig((path_clustering + data_date + columname_kmeans + '.png'), dpi = 500)
#    plt.show()
    plt.close(fig)
	
	####################################################################
	# DBSCAN
	# derive parameters from distance matrix
	# distances stats - compute mean and standard deviation
    dist_mean = np.mean(distance_matrix_centroids)
    dist_std = np.std(distance_matrix_centroids)
    dist_min = np.min(distance_matrix_centroids)
    dist_max = np.max(distance_matrix_centroids)
    
    print (data_date)
    print ('mean: ', dist_mean)
    print ('st: ', dist_std)
    print ('min: ', dist_min)
    print ('max ', dist_max )
    
    max_dist = dist_max

	# Spatial Clustering	
    # Maximum distance between two samples to be considered as neighbors
    epsilon = (max_dist)/6372.
    # Number of samples in neighborhood for a point to be considered as a core point
    if len(X) < 50:
        min_samp = 2
    else:
        min_samp = 7
	
    db = DBSCAN(eps = epsilon, min_samples = min_samp, algorithm = 'ball_tree', metric = 'haversine').fit(np.radians(X))
					
	# Add cluster label to dataframe 
    y_dbscan = db.labels_
    number_cluster_db = len(set(y_dbscan))
    columname_dbscan = 'spatial_cluster_dbscan_n_' + str(number_cluster_db)
    cluster_db = pd.DataFrame(y_dbscan, columns = [columname_dbscan])
    data_centroids = data_centroids.join(cluster_db)
	
	# Plot Cluster
	# Generate CMAP for all clusters
	# Cluster -1 (noise) will be set to white
    labels = np.unique (y_dbscan)
    label_col = []	
    for id in labels:        
        # label -1 indicates noise as identified by DBSCAN
        if id == -1: 
            label_col.append('#FFFFFF')
        # Generate random colors for all other labels
        else:
            hex_number = '#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)])    
            if not hex_number == '#FFFFFF':
                label_col.append(hex_number)
				
	# Plotting
    plt.rcParams['font.family'] = 'Arial'             
    fig, ax = plt.subplots(1, figsize=(12,9))       
    cmap = plt.matplotlib.colors.ListedColormap(label_col, N=number_cluster_db)                 
	#centroids.plot(ax = ax, color = 'grey')
    area_shape.plot(ax = ax, color = 'white', edgecolor='grey', linewidth = 1)
    data_centroids.plot(column = columname_dbscan, ax = ax, cmap = cmap, markersize = 5)
    plt.suptitle('Detected Hotspots - {a} {y} '.format(a = area, y = data_date), fontsize = 20)
    plt.title('DBSCAN Spatial Clustering ({n} Cluster - max. Distance: {d} km)'.format(n = number_cluster_db-1,
              d = round(max_dist, 2)), fontsize = 15, pad=15)
    ax.set_axis_off()
    plt.savefig((path_clustering + data_date + columname_dbscan + '.png'), dpi = 500)
#    plt.show()
    plt.close(fig)
	
	##########################################
    # Export generated point shapefile
    data_out_name = path_centroids + '{a}_spatial_clusters_{d}'.format(a = area.lower(), d = data_date) 
    data_centroids.to_file((data_out_name + '.gpkg'), driver = 'GPKG')

	# Print Metadata 
	# Data Year
	# Number of Features
	# Distance Matrix
	# KMeans: number of clusters, features per cluster
	# DBSCAN: epsilon, min_samples (fix), number of cluster, features per cluster, noise
    meta_out.write('\n{a}: {y}\n'.format(a = area, y = data_date) +
            'Number of Features: {n}\n'.format(n=len(data_centroids.index)))
          
    # Write info of clustered features
    # Kmeans
    meta_out.write('\n'+
                   'KMEANS Clustering\n')# +
#                   'Maximum of average silhouette: {s} (tested number of cluster: 2-15)\n'.format(s=best_cluster_n))
    write_cluster_info(meta_out, y_kmeans)
    # DBSCAN
    meta_out.write('\n'+
                   'DBSCAN Clustering\n' +
                   'Mean distance of features: {d}\n'.format(d = round(dist_mean,2))+
                   'Epsilon: {d} km bzw. {e} in radians\n'.format(d = round(max_dist,2), e = round(epsilon,5)))
    write_cluster_info(meta_out, y_dbscan)
            
    meta_out.write('\n'+
                   '--------------------------------\n')

print('Spatial Clustering done.')	