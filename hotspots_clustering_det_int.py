# -*- coding: utf-8 -*-
"""
Fire Hotspot Analysis

CLUSTERING ANALYSIS

Clustering data in sight of
    - number of detected hotspots per feature
    - average intensity (FRP) per feature

Used methods for processing data: 
    - outlier trimming of avg intensity
    - kmeans clustering with set number of clusters to 5 

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
from shapely.geometry import Point

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 

#from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

##############################################################################################
#set datapaths
#directory with all data
#path_data_orig = ('D:\johanna_roll\h3_hotspots\germany_1995_2019\\')
path_data_orig = os.path.dirname(os.path.realpath('hotspots_clustering_det_int.py')) + '\\'
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
    
path_clustering = path_out + ('clustering_det_int\\')
if not os.path.exists(path_clustering):
    os.makedirs(path_clustering)
	
path_centroids = path_clustering + ('centroids_shapefiles\\')
if not os.path.exists(path_centroids):
    os.makedirs(path_centroids)
    
##############################################################################################
# Function    
def set_cluster_color (y):
    '''
    Set color for each cluster
    For better visualization, the colors are assigned by descending number of features per cluster
    The smaller the number of features per cluster, the more intense the color
    
    Function right now limited to 5 clusters (for more, add more colors in 'label_col')
    
    Parameter: Clustered data with labels
    Returns: Dictionary with cluster label (key) and assigned color (value), sorted by number of features per cluster (descending) 
    '''

    label_col = ['#FFFFFF', '#c8c8c8', '#fbd280', '#ec846e', '#ca0020']                 
             
    # Get unique labels with min and max             
    labels = np.unique(y)
    label_min, label_max = int(labels.min()), int(labels.max())
    
    # Counting features per cluster
    counts_cluster = {}
    for i in range (label_min, label_max+1):
        count = len(y[y == i])        
        counts_cluster[i] = count
    
    # Sorting by features per cluster (descending)    
    counts_cluster_sorted = sorted(counts_cluster.items(), key = lambda x: x[1], reverse = True)
    
    # Assinging color for each cluster
    color_cluster = {}
    for i, c in zip(counts_cluster_sorted, label_col):
        color_cluster[i[0]] = c
        
    return color_cluster
     
    
def print_cluster_info (cluster_labels, df, colname):
    '''
    Prints information for each cluster in console: 
        - number of clustered features
        - number of clusters
        - mean of number of detected hotspots in each cluster
        - mean of average intensity (FRP) in each cluster
        
    Parameter: Data with clustered labels (y), dataframe of original data, columname in df with clusterlabels    
    '''
    # Number of labelled features
    n_features = len(cluster_labels)
    num_clusters = len(set(cluster_labels))

    print ('{f} features are assigned to {c} cluster\n'.format(f=n_features, c=num_clusters))   
    labels = np.unique(cluster_labels)
    label_min, label_max = int(labels.min()), int(labels.max())
    
    print ('Features per Cluster:')
    for i in range (label_min, label_max+1):
        count = len(cluster_labels[cluster_labels == i])
        
        # compute mean number of detections and avg intensity for each cluster
        df_label = df.loc[df[colname] == i]
        mean_n_det = df_label['indexesFound'].mean()
        mean_int = df_label['frpMWkm2Avg'].mean()
        
        print ('Cluster {l}: {c}'.format(l=i, c=count))
        print ('Mean of number of detected hotspots: ', round(mean_n_det, 1))
        print ('Mean of average intensity (FRP in MW/km2): ', round (mean_int, 10))
        

def write_cluster_info (output_file, cluster_labels, df, colname):
    '''
    Writes information for each cluster to file: 
        - number of clustered features
        - number of clusters
        - mean of number of detected hotspots in each cluster
        - mean of average intensity (FRP) in each cluster
        
    Parameter: Output file, data with clustered labels (y), dataframe of original data, columname in df with clusterlabels    
    '''
    
    # Number of labelled features
    n_features = len(cluster_labels)
    num_clusters = len(set(cluster_labels))

    output_file.write('{f} features are assigned to {c} cluster\n'.format(f=n_features, c=num_clusters))  
    
    labels = np.unique(cluster_labels)
    label_min, label_max = int(labels.min()), int(labels.max())
    
    output_file.write('Features per Cluster:\n')
    for i in range (label_min, label_max+1):
        count = len(cluster_labels[cluster_labels == i])
        output_file.write('Cluster {l}: {c}\n'.format(l=i, c=count)) 
        
        # Compute mean number of detections and avg intensity for each cluster
        df_label = df.loc[df[colname] == i]
        mean_n_det = df_label['indexesFound'].mean()
        mean_int = df_label['frpMWkm2Avg'].mean()
        
        output_file.write('Mean of number of detected hotspots: {d}\n'.format(d=round(mean_n_det, 1)))
        output_file.write('Mean of average intensity (FRP in MW/km2): {i}\n'.format(i = round (mean_int, 10)))
    
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
meta_out = open((path_clustering + 'clustering_info.txt'), 'w')
meta_out.write('Clustering of Fire Hotspot Data: Number of Detected Hotspots & Average Intensity (FRP in MW/km2)\n')

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
	
	# Convert polygons to points: Extract centroids from polygons for better visualization
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
	
    # Extract number of detected hotspots (indexesFound) and average intensity (frpWMkm2Av) for all features
    a = pd.DataFrame(data_centroids['indexesFound'])
    b = pd.DataFrame(data_centroids['frpMWkm2Avg'])
    
    # Preprocessing: detect and trimm outlier of avg intensity data to the 90th percentile
    # Skweness score of distribution of intensity values 
    skew_score = b['frpMWkm2Avg'].skew()
    # Get 90th percentile
    p90 = b.quantile(0.90)
    # All avg intensity values above the 90th percentile will be assigned with value of 90th percentile = trimmed
    b_new = np.where(b > p90, p90, b)
    b_trim = pd.DataFrame(b_new)
    skew_score_trim = b_trim[0].skew()
    
    print ('Detection of outlier in avg intensity data\n')
    print ('Skew score: from {s} to {st} after trimming of data'.format(s = round(skew_score,2), st = round(skew_score_trim,2)))
    
    # Stacking of all features 
    X = np.column_stack((a, b_trim))
    
    # Scaling of data
    X_transf = StandardScaler().fit_transform(X)
    # Other option for scaling: RobustScaler (more robust towards outliers in data)
    # However, the scaling effect of the number of detections data appears to be better with StandardScaler 
    # X = RobustScaler(quantile_range=(10, 90)).fit_transform(X_transf)
        	
    # # Plotting histogram of distances
	# dist_array = np.asarray(dist_matrix)
	# plt.title('Histogram of distance matrix - {d}'.format(d = data_date), pad = 15)
    # sns.distplot(dist_array.ravel(), bins = 70, kde = False)
	# plt.show()

	####################################################################
	## Clustering
	# KMeans
    # Determine optimal number of cluster (Elbow Method) - visual interpretation needed
#    wcss = []
#    for i in range (1,15):
#        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#        kmeans.fit(X)
#        wcss.append (kmeans.inertia_)
#        
#    plt.plot(range(1,15), wcss, 'bx-')
#    plt.title('elbow method')
#    plt.xlabel('number of clusters')
#    plt.ylabel('wcss')
#    plt.show()
    
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
##    # Plot 
##    plt.plot(range(len(silhouette)), list(silhouette.values()),  'bx-')
##    plt.xticks(range(len(silhouette)), list(silhouette.keys()))
##    plt.title('silhouette average - {d}'.format(d = data_date))
##    plt.xlabel('number of clusters')
##    plt.ylabel('silhouette average')
##    plt.savefig((path_clustering + data_date + '_silhouette.png'), dpi = 500)
##    plt.show()
#    
#    #optimal number of cluster based on this method with highest silhouette score
#    best_cluster_n = max(silhouette, key = silhouette.get)	
#    number_cluster_kmeans = best_cluster_n
    
    # set number of clusters
    if len(X) >= 5:
        number_cluster_kmeans = 5
    else:
        number_cluster_kmeans = len(X)
        
	# init 'k-means++' for speeding up convergence; random state for deterministic randomness for centroid initialization
    kmeans = KMeans (n_clusters = number_cluster_kmeans, init = 'k-means++', random_state = 42, max_iter = 400)
    # Compute cluster centers and predict cluster index for each sample
    y_kmeans = kmeans.fit_predict(X_transf)
	
	# Add cluster label to data
    columname_kmeans = 'cluster_kmeans_n_' + str(number_cluster_kmeans)
    cluster_k = pd.DataFrame(y_kmeans, columns = [columname_kmeans])
    data_centroids = data_centroids.join(cluster_k)
    
    print_cluster_info(y_kmeans, data_centroids, columname_kmeans)

    ##################################################################
	# Plot result
    # Dictionary with specific colors assigned to each cluster 
    color_cluster = set_cluster_color (y_kmeans)
    
    # Scatterplot 
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(1, figsize=(12,9))
    plt.suptitle('Scatterplot of Clustered Fire Hotspot Data', fontsize = 20)
    plt.title('{a} {y}'.format(a = area, y = data_date), fontsize = 15, pad = 15)
    plt.ylabel('Average Intensity (FRP in MW/km2)', fontsize = 13)
    plt.xlabel('Number of Detected Hotspots in Feature', fontsize =  13)    
    ax.set_facecolor('#DDDDDD')

    # Plotting each cluster indivudially
    for cluster in color_cluster:
        color = color_cluster.get(cluster)
        # Boolean array with "True" for respective cluster
        cond = y_kmeans == cluster
        # X where condition is met
        X_cluster = X[cond]    
        # Scatterplot of cluster with 
        ax.scatter(X_cluster[:, 0], X_cluster[:, 1], c= color, label='Cluster {l}'.format(l=cluster), edgecolors = '#FFFFFF')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        
    ax.legend()
    plt.savefig((path_clustering + 'scatterplot_clustering_{a}_{d}.png'.format(a = area.lower(), d = data_date)), dpi = 600)
#    plt.show()
    plt.close(fig)


    # Plot Map
    fig, ax = plt.subplots(1, figsize=(12,9))
    plt.rcParams['font.family'] = 'Arial'
    plt.suptitle('Detected Hotspots - {a} {y} '.format(a = area, y = data_date), fontsize = 20)
    plt.title('KMEANS Clustering Number of Detected Hotspots & Average Intensity (FRP)', fontsize = 15, pad=15)
    area_shape.plot(ax = ax, color = '#FFFFFF', edgecolor='#DDDDDD', linewidth = 1)
              
    # Plotting each cluster indivudially - from large to small cluster size    
    for cluster in color_cluster:
        color_cl = color_cluster.get(cluster)
        
        plot_cluster = 'plot_' + str(cluster)
        
        # Extraction of data for respective cluster
        data_cluster = data_centroids.loc[data_centroids[columname_kmeans] == cluster]
        data_cluster.plot(ax = ax, color = color_cl, label='Cluster {l}'.format(l=cluster), 
                          edgecolors = '#666666', linewidth = 0.25, markersize = 3)
    
    plt.legend(markerscale = 5)
    ax.set_axis_off()
    plt.savefig((path_clustering + data_date + columname_kmeans + '.png'), dpi = 500)
#    plt.show()
    plt.close(fig)
	
	##########################################
    # Export generated point shapefile
    data_out_name = path_centroids + '{a}_det_int_clusters_{d}'.format(a = area.lower(), d = data_date) 
    data_centroids.to_file((data_out_name + '.gpkg'), driver = 'GPKG')
    
	# Print Metadata 
	# Data Year
	# Number of Features
	# Distance Matrix
	# KMeans: number of clusters, features per cluster
    meta_out.write('\n{a}: {y}\n'.format(a = area, y = data_date) +
            'Number of Features: {n}\n'.format(n=len(data_centroids.index)))
          
    # Write info of clustered features
    # Kmeans
    meta_out.write('\n'+
                   'KMEANS Clustering\n')
    write_cluster_info(meta_out, y_kmeans, data_centroids, columname_kmeans)

            
    meta_out.write('\n'+
                   '--------------------------------\n')

print('Spatial Clustering done.')	