# -*- coding: utf-8 -*-
"""
Fire Hotspot Analysis

CLUSTERING ANALYSIS

Generating Scatterplots of data in sight of
    - number of detected hotspots per feature 
    - average intensity (FRP) per feature 
    - middle DOY between start and end of detection (month)

Used methods for processing data: 
    - outlier trimming of avg intensity
    - encoding of tstart & duration
    - multiple correspondence analysis (MCA) of encoded middle DOY
    - principle component analysis (PCA) of all three coponents - reduction to two dimensions
    - kmeans clustering with 5 clusters

Output
Scatterplots for each datafile (year/date) 
    - PCA based on number of detections, avg intensity, middle DOY
    - PC1 axis with insights on correlation between number of detections & avg intensity
    - PC2 axis with insights on time of the year (beginning (positive values) and end (negative values) of timeseries)

@author: Johanna Roll
"""
#import packages
import glob
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import prince

##############################################################################################
# Set datapaths
# Directory with all data
#path_data_orig = ('D:\johanna_roll\h3_hotspots\germany_1995_2019\\')
path_data_orig = os.path.dirname(os.path.realpath('hotspots_clustering_det_int_time.py')) + '\\'
path_data = path_data_orig + ('data_clipped\\')

# If no clipped data exists, original data is data
if not os.path.exists(path_data):
    path_data = path_data_orig

# Pet paths of all data
path_list = glob.glob(os.path.join(path_data, '*.gpkg'))

# Directory for outputs
path_out = path_data + ('output\\')
if not os.path.exists(path_out):
    os.makedirs(path_out)
    
path_clustering = path_out + ('clustering_det_int_time\\')
if not os.path.exists(path_clustering):
    os.makedirs(path_clustering)
    
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

    label_col = ['#111111', '#c8c8c8', '#fbd280', '#ec846e', '#ca0020']
                 
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

# Turn off interactive plotting 
plt.ioff()

###########################################################################################

doy_count_dict = {}

### Iterate over all data and generate clusters
for date in data_list:
    data = data_list.get(date)
	# Collect Metainfo
    dataname_split = date.split('_')
    area = dataname_split[0].upper()
    data_date = dataname_split[-1]	
    
    print('Processing: ', area, data_date)
	
	## Prepare data for clustering	
    # Extract number of detected hotspots (indexesFound) and average intensity (frpWMkm2Av) for all features
    a = pd.DataFrame(data['indexesFound'])
    b = pd.DataFrame(data['frpMWkm2Avg'])
    
    # Preprocessing: detect and trimm outlier of avg intensity data to the 90th percentile
    # Skweness score of distribution of intensity values 
    skew_score = b['frpMWkm2Avg'].skew()
    # Get 90th percentile
    p90 = b.quantile(0.90)
    # All avg intensity values above the 90th percentile will be assigned with value of 90th percentile = trimmed
    b_new = np.where(b > p90, p90, b)
    b_trim = pd.DataFrame(b_new)
    skew_score_trim = b_trim[0].skew()
    
#    print ('Detection of outlier in avg intensity data\n')
#    print ('Skew score: from {s} to {st} after trimming of data'.format(s = round(skew_score,2), st = round(skew_score_trim,2)))
#    
    # Stacking of features 
    det_int = np.column_stack((a, b_trim))
    
    #########
    ##### Extraction of information of temporal dimension of detected hotspots
    #temporal information for each year
    #takes the middle of tstart and tend as reference for temporal setting
    #this approach will proprably generete better results with monthly / daily data
    doy_mean_year = []
    
    for index, row in data.iterrows():
    
        #start and end of detection for each feature
        tstart_doy = pd.to_datetime(row['tstart']).dayofyear
        tend_doy = pd.to_datetime(row['tend']).dayofyear

        #calculate the median DOY
        doy_mean = int((tstart_doy + tend_doy)/2)
        doy_mean_year.append(doy_mean)
    
    doy_mean_year = np.asarray(doy_mean_year)
    doy_mean_year = doy_mean_year.reshape(-1,1)
      
    # Encoding mean DOY with detection
    enc = OneHotEncoder(handle_unknown = 'ignore')
    enc.fit(doy_mean_year)
    doy_mean_enc = enc.transform(doy_mean_year).toarray()
    
    # MCA of detected DOYs
    mca = prince.MCA()
    mca = mca.fit(doy_mean_enc)
    mca = mca.transform(doy_mean_enc)
    mca_doy = np.array(mca)


    ####################################################################
    ## Stacking of variables of each features: 
    # - number of detections
    # - avg intensity
    # - started month of detection    
    stack = np.column_stack((det_int, mca_doy))
    
    # Principal Component Analysis (PCA)
    # Reducing the dimensions of stack from four to two 
    pca = PCA(n_components = 2)
    principal_comp = pca.fit_transform(stack)
    pc_df = pd.DataFrame(data = principal_comp, columns = ['PC1', 'PC2'])
    
    # Assessing loss of information following PCA
    pca_quali = pca.explained_variance_ratio_
    
    # Converting Dataframe to Array
    X = np.asarray(pc_df)

	#############################################
	## Clustering
	# KMeans
    # set number of clusters
    if len(X) >= 5:
        number_cluster_kmeans = 5
    else:
        number_cluster_kmeans = len(X)

	# init 'k-means++' for speeding up convergence; random state for deterministic randomness for centroid initialization
    kmeans = KMeans (n_clusters = number_cluster_kmeans, init = 'k-means++', random_state = 42, max_iter = 400)
    # Compute cluster centers and predict cluster index for each sample
    y_kmeans = kmeans.fit_predict(X)
	
	# Add cluster label to data
    columname_kmeans = 'cluster_det_int_time' + str(number_cluster_kmeans)
    cluster_k = pd.DataFrame(y_kmeans, columns = [columname_kmeans])
    data = data.join(cluster_k)

    ##################################################################
	# Plot result
    # Dictionary with specific colors assigned to each cluster 
    color_cluster = set_cluster_color (y_kmeans)
    
    # Scatterplot 
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(1, figsize=(12,9))
    plt.title('Scatterplot of Clustered Fire Hotspot Data ({n} features): Number of Detection, Avg Intensity, Mean of Start and End of Detection'.format(n=len(X)+1), fontsize = 12, pad = 15)
    plt.suptitle('{a} {y}'.format(a = area, y = data_date), fontsize = 20)
    plt.ylabel('Principal Component 2 - temporal component', fontsize = 13)
    plt.xlabel('Principal Component 1 - thematic component of number of detections & avg intensity', fontsize =  13)    
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
        
    ax.legend()
    plt.savefig((path_clustering + 'scatterplot_det_int_time_{a}_{d}.png'.format(a = area.lower(), d = data_date)), dpi = 600)
#    plt.show()
    plt.close(fig)

print('Done.')	