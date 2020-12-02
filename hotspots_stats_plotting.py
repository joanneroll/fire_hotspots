# -*- coding: utf-8 -*-
"""
Fire Hotspot Analysis

STATS ANALYSIS--

Reading of data
Calculating relevant information
- sum of detected hotspots per year
- mean / min / max of avg intensity (FRP in MW/km2) per year
- months & day of years (DOY) with detected hotspots

Plotting stats analysis
Writing dataframes to file 
Plotting information on detected hotspots and avg intensity as map 
- coloring fit to statistical evaluation

@author: Johanna Roll
"""

#import packages 
import glob
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import jenkspy

####################################################################################
#set datapaths
#directory with all data
#path_data_orig = ('D:\johanna_roll\h3_hotspots\germany_1995_2019\\')
path_data_orig = os.path.dirname(os.path.realpath('hotspots_stats_plotting.py')) + '\\'

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

path_stats = path_out + ('statistics\\')
if not os.path.exists(path_stats):
    os.makedirs(path_stats)    

path_plotting = path_out + ('plotting_data\\')
if not os.path.exists(path_plotting):
    os.makedirs(path_plotting)

#####################################################################################
#setup data dictionary
data_list = {}

#read data
#GeoDataFrames
for path in path_list:
    #get dataname
    data_name = path.split('\\')[-1][:-5]
    #read file
    data = gpd.read_file(path)
    data.name = data_name
    #add file to dictionary with respective filename as key
    data_list[data_name] = data

#read area of interest
area_shape = gpd.read_file(path_area)

# Turn off interactive plotting 
plt.ioff()

##########################################################################################    
#analyse data    

#set up dictionary for collecting analysis    
data_analysis = {}
data_analysis_list = {}

det_total = []
int_total = []

doy_start_total = []
doy_end_total = []

for key in data_list:
    #call individual year
    data_year = data_list.get(key)
    
    #setup lists for analysis for each year ()
    det_year = []
    int_year = []
    
    #iterate over all features (rows) of year
    for index, row in data_year.iterrows():
        #get information
        #number of detections
        det = row['indexesFound']
        #intensity
        av_frp = row['frpMWkm2Avg']
            
        #year
        dataname_split = key.split('_')
        year = dataname_split[-1]	
        #area
        area = dataname_split[0].upper()

        
        #collect information to list
        det_year.append(det)
        int_year.append(av_frp)
        
        det_total.append(det)
        
        #start and end of detection for each feature
        tstart_doy = pd.to_datetime(row['tstart']).dayofyear
        tend_doy = pd.to_datetime(row['tend']).dayofyear
        #collect single doy values
        doy_start_total.append(tstart_doy)
        doy_end_total.append(tend_doy)
        
    #analyse information for each year
    det_sum = sum(det_year)
    
    #remove extreme outliers in avg intensity
    #trimming of data based on 90th percentil
    p90 = np.quantile(int_year, 0.90)
    int_year = np.where(int_year > p90, p90, int_year)
    
    int_total.extend(int_year)
        
    int_mean = sum(int_year)/len(int_year)
    int_min = min(int_year)
    int_max = max(int_year)
    
    #temporal information for each year
    #start and end of detection for each feature
    tstart_month = np.array(pd.DatetimeIndex(pd.to_datetime(data_year['tstart'])).month)
    tstart_day = np.array(pd.DatetimeIndex(pd.to_datetime(data_year['tstart'])).dayofyear)
    
    tend_month = np.array(pd.DatetimeIndex(pd.to_datetime(data_year['tend'])).month)
    tend_day = np.array(pd.DatetimeIndex(pd.to_datetime(data_year['tend'])).dayofyear)

    #collect analysis for year
    analysis_year = {'date' : year, 'detections_sum' : det_sum, 'intensity_mean' : int_mean, 
                     'intensity_min' : int_min, 'intensity_max' : int_max}
    
    analysis_year_lists = {'date' : int(year), 'tstart_month' : tstart_month, 'tstart_doy' : tstart_day,
                           'tend_month' : tend_month, 'tend_doy' : tend_day, 'n_det' : det_year}
    
    #save analysis for each year (key) in dictionary    
    data_analysis[key] = analysis_year
    data_analysis_list[key] = analysis_year_lists

#analyse tstart and tend for duration / timeframe of detection of hotspots for each feature
#duration will be summed up for each month, respectively DOY
#two versions: 
# - one collecting only the registered dates (=number of features)
# - one also considering the number of detected hotspots per feature on registered dates (=accumulated number of detections)
months_count_year = {}
months_count_acc_year = {}
doy_count_year = {}
doy_count_acc_year = {}


#call first and last registerend DOY of dataset for plotting
first_start_doy = min(doy_start_total)
last_end_doy = max(doy_end_total)

for stats_year in data_analysis_list:
    #loop through each year
    year = data_analysis_list[stats_year]['date']
    
    #get number of detected hotspots per sample 
    det = data_analysis_list[stats_year]['n_det']
    
    #get array of months in which the detection started and ended
    tstart_month = data_analysis_list[stats_year]['tstart_month']
    tend_month = data_analysis_list[stats_year]['tend_month']
    
    #create list with length of 12 -> months - Index 0 additional / ignored
    #index 1 corresponds with january, 2 with february, ect..
    months_count = [0] * 13
    months_count_acc = [0] * 13
    
    #in range of starting and ending month, count each month
    #list months_count collects all months, in which hotspot detections were registered
    for start, end, n_det in zip (tstart_month, tend_month, det):
        for month in range (start, end+1, 1):
            #counting months with detections
            months_count[month] += 1
            #counting number of detections of months with detections
            months_count_acc[month] += n_det
                        
    months_count_year[year] = months_count 
    months_count_acc_year[year] = months_count_acc
    
    
    #same procedure for more detailed temporal analysis with DOY   
    tstart_doy = data_analysis_list[stats_year]['tstart_doy']
    tend_doy = data_analysis_list[stats_year]['tend_doy']
    
    #length of list based on leap year - lists need to be of same lenght
    doy_count = [0] * (366+1)    
    doy_count_acc = [0] * (366+1)

    #collect counts for each DOY for every avalable year
    for start, end, n_det in zip (tstart_doy, tend_doy, det):
            for doy in range (start, end+1, 1):
                #counting DOYs with detections
                doy_count[doy] += 1
                #counting number of detections of DOYs with detections
                doy_count_acc[doy] += n_det
                            
    doy_count_year[year] = doy_count  
    doy_count_acc_year[year] = doy_count_acc


#print basic stats information in console
for key, value in data_analysis.items():
    print (key)
    for info in value:    
        if info == 'detections' or info == 'intensity':
            #skip arrays
            continue
        
        print(info, ' : ', value[info])   
    print ('\n')

#read dictionaries in dataframe
years_analysis_df = pd.DataFrame(data_analysis, dtype = float).transpose()
#convert column 'year' from float to int
years_analysis_df['date'] = years_analysis_df['date'].astype(int)

months_count_df = pd.DataFrame(months_count_year, dtype = int)
months_count_acc_df = pd.DataFrame(months_count_acc_year, dtype = int)
doy_count_df = pd.DataFrame(doy_count_year, dtype = int)
doy_count_acc_df = pd.DataFrame(doy_count_acc_year, dtype = int)


#save files
#export dataframes to excel
years_analysis_df.to_excel((path_stats + 'output_analysis.xlsx'), index=False, header=True)

months_count_df.to_excel((path_stats + 'months_count.xlsx'), index=True, header=True)
months_count_acc_df.to_excel((path_stats + 'months_count_acc.xlsx'), index=True, header=True)
doy_count_df.to_excel((path_stats + 'doy_count.xlsx'), index=True, header=True)
doy_count_acc_df.to_excel((path_stats + 'doy_count_acc.xlsx'), index=True, header=True)

###########################################################################################################
###########################################################################################################
###########################################################################################################
#plotting results 

#####STATS

#set font family globally
plt.rcParams['font.family'] = 'Arial'
#data_description = 'Data: Germany 1995-2019'
data_description = 'Data: {a} - {b}'.format(a = str(years_analysis_df['date'].min()) , 
                          b = str(years_analysis_df['date'].max()))

#detected hotspots
fig, ax = plt.subplots(1, figsize=(16,9))
graph = sns_plot_det = sns.barplot(x = 'date', y = 'detections_sum', data = years_analysis_df, 
                           color='#39CCCC')
plt.suptitle('Number of Detected Fire Hotspots', fontsize = 20)
plt.title(data_description, fontsize = 15, pad=15)
plt.xlabel('Date', fontsize = 15)
plt.xticks(rotation=45)
plt.ylabel('Detected Hotspots', fontsize = 15)
plt.savefig((path_stats + 'graph_number_detections.png'))
#plt.show()
plt.close(fig)


#intensity of hotspots
fig, ax = plt.subplots(1, figsize=(16,9))
sns.lineplot( x = 'date', y = 'intensity_max', data = years_analysis_df,
                             color='#FF4136', label = 'Maximum')
sns.lineplot( x = 'date', y = 'intensity_mean', data = years_analysis_df,
                            color='#39CCCC', label= 'Mean')
sns.lineplot( x = 'date', y = 'intensity_min', data = years_analysis_df,
                             color ='#0074D9', label = 'Minimum')

ax.set_yscale('log')
plt.suptitle('Average Intensity of Detected Hotspots - Fire Radiative Power (FRP)', fontsize = 20)
plt.title(data_description, fontsize = 15, pad=15)
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Mean of Average Intensity (FRP in MW/km2)', fontsize = 15)
plt.savefig((path_stats + 'graph_intensity.png'))
#plt.show()
plt.close(fig)


#intensity Mean
fig, ax = plt.subplots(1, figsize=(16,9))
sns.lineplot( x = 'date', y = 'intensity_mean', data = years_analysis_df,
                            color='#39CCCC')
plt.suptitle('Average Intensity of Detected Hotspots - Mean', fontsize = 20)
plt.title(data_description, fontsize = 15, pad=15)
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Mean of Average Intensity (FRP in MW/km2)', fontsize = 15)
plt.savefig((path_stats + 'graph_intensity_mean.png'))
#plt.show()
plt.close(fig)

#intensity Min
fig, ax = plt.subplots(1, figsize=(16,9))
sns.lineplot( x = 'date', y = 'intensity_min', data = years_analysis_df,
                             color ='#0074D9')
plt.suptitle('Average Intensity of Detected Hotspots: Minimum', fontsize = 20)
plt.title(data_description, fontsize = 15, pad=15)
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Mean of Average Intensity (FRP in MW/km2)', fontsize = 15)
plt.savefig((path_stats + 'graph_intensity_min.png'))
#plt.show()
plt.close(fig)


#intensity Max
fig, ax = plt.subplots(1, figsize=(16,9))
sns.lineplot( x = 'date', y = 'intensity_max', data = years_analysis_df,
                             color='#FF4136')
plt.suptitle('Average Intensity of Detected Hotspots: Maximum', fontsize = 20)
plt.title(data_description, fontsize = 15, pad=15)
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Mean of Average Intensity (FRP in MW/km2)', fontsize = 15)
plt.savefig((path_stats + 'graph_intensity_max.png'))
#plt.show()
plt.close(fig)

### temporal analysis
#months with registered detections
n_columns = len(months_count_df.columns)
fig, axes = plt.subplots(nrows=n_columns, ncols=1, figsize = (12,(n_columns*2.5)), sharey = True)
for ax in axes:
    ax.set_xlim((1,12))   
fig.tight_layout()
fig.subplots_adjust(top=0.96)
fig.suptitle('Months with Registered Detections of Fire Hotspots for each Available Date', fontsize = 20)
months_count_df.plot(subplots=True, ax=axes)
plt.savefig((path_stats + 'detections_per_month.png'))
#plt.show()
plt.close(fig)

#months of registered detections with ACCUMULATED number of detections
n_columns = len(months_count_acc_df.columns)
fig, axes = plt.subplots(nrows=n_columns, ncols=1, figsize = (12,(n_columns*2.5)), sharey = True)
for ax in axes:
    ax.set_xlim((1,12))   
fig.tight_layout()
fig.subplots_adjust(top=0.96)
fig.suptitle('Months with Registered and Accumulated Detections of Fire Hotspots for each Available Date', fontsize = 20)
months_count_acc_df.plot(subplots=True, ax=axes)
plt.savefig((path_stats + 'detections_per_month_acc.png'))
#plt.show()
plt.close(fig)


#DOYs with registered detections - for whole year
n_columns = len(doy_count_df.columns)
fig, axes = plt.subplots(nrows=n_columns, ncols=1, figsize = (12,(n_columns*2.5)), sharey = True)
fig.tight_layout()
fig.subplots_adjust(top=0.96)
fig.suptitle('Day Of Years (DOY) with Registered Detections of Fire Hotspots for each Available Date', fontsize = 20)
doy_count_df.plot(subplots=True, ax=axes)
plt.savefig((path_stats + 'detections_per_doy_total_year.png'))
#plt.show()
plt.close(fig)

#DOYs with registered detections with ACCUMULATED number of detections - for whole year
n_columns = len(doy_count_acc_df.columns)
fig, axes = plt.subplots(nrows=n_columns, ncols=1, figsize = (12,(n_columns*2.5)), sharey = True)
fig.tight_layout()
fig.subplots_adjust(top=0.96)
fig.suptitle('Day Of Years (DOY) with Registered and Accumulated Detections of Fire Hotspots for each Available Date', fontsize = 20)
doy_count_acc_df.plot(subplots=True, ax=axes)
plt.savefig((path_stats + 'detections_per_doy_acc_total_year.png'))
#plt.show()
plt.close(fig)


#DOYs with registered detections - only for available DOYs
n_columns = len(doy_count_df.columns)
fig, axes = plt.subplots(nrows=n_columns, ncols=1, figsize = (12,(n_columns*2.5)), sharey = True)

for ax in axes:
    #set range of x axis to registered doys only
    ax.set_xlim((first_start_doy, last_end_doy))
    #ensure xticks as integer for visuality
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

fig.tight_layout()
fig.subplots_adjust(top=0.96)
fig.suptitle('Day Of Years (DOY) with Registered Detections of Fire Hotspots for each Available Date', fontsize = 20)
doy_count_df.plot(subplots=True, ax=axes)
plt.savefig((path_stats + 'detections_per_doy.png'))
#plt.show()
plt.close(fig)

#DOYs with registered detections - only for available DOYs
n_columns = len(doy_count_acc_df.columns)
fig, axes = plt.subplots(nrows=n_columns, ncols=1, figsize = (12,(n_columns*2.5)), sharey = True)

for ax in axes:
    #set range of x axis to registered doys only
    ax.set_xlim((first_start_doy, last_end_doy))
    #ensure xticks as integer for visuality
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

fig.tight_layout()
fig.subplots_adjust(top=0.96)
fig.suptitle('Day Of Years (DOY) with Registered and Accumulated Detections of Fire Hotspots for each Available Date', fontsize = 20)
doy_count_acc_df.plot(subplots=True, ax=axes)
plt.savefig((path_stats + 'detections_per_doy_acc.png'))
#plt.show()
plt.close(fig)

print ('Statistic analysis and plotting done.')

###########################################################################################################
###### PLOTTING DATA - MAP
print ('Starting plotting of maps.')

#Prepare Colormaps respectively breaks based on distribution of data of available time series
#breaks number of detections
#get number of unique values
det_value = np.unique(det_total).tolist()

#if possible, define 5 natural breaks (jenksy)
if len(det_value) > 5:
    color_breaks_det = jenkspy.jenks_breaks(det_total, nb_class = 5)
else:
    color_breaks_det = det_value
    color_breaks_det.append(color_breaks_det[-1]+1)

#breaks intensity
#define 5 quantils
quantil_breaks = [0.2, 0.4, 0.6, 0.8]
quantils = np.quantile(int_total, quantil_breaks)
int_min_max = [min(int_total), max(int_total)]
#append quantil breaks with min and max
int_min_max = np.array(int_min_max)
color_breaks_int = sorted(np.append(quantils, int_min_max))


for key, stats_year in zip(data_list, data_analysis):
    data_year = data_year = data_list.get(key)
                
    #plotting
    #prepare information
    year = data_analysis_list[stats_year]['date']
    plotfilename_det = 'detections_' + str(year)
    plotfilename_int = 'intensity_' + str(year)  

    print('Plotting: ', year)
    
    #plotting number of detected hotspots
    fig, ax = plt.subplots(1, figsize=(12,10))
    cmap = colors.ListedColormap(['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c'])
    norm = colors.BoundaryNorm(boundaries = color_breaks_det, ncolors = 5)

    
    area_shape.plot(ax = ax, color = '#DDDDDD', edgecolor='grey', linewidth = 1)
    data_year.plot(ax = ax, color = 'grey')
    data_year.plot(ax = ax, column='indexesFound', cmap = cmap, norm = norm)
        
    fig.suptitle('Number of Detected Hotspots', fontsize = 18)
    ax.set_title('Date: {y}'.format(y=year), fontdict={'fontsize': 13})
    
    #axis colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size = '3 %', pad = 0.05)
    vmax=data_year.frpMWkm2Avg.max()
    
    #handeling case with only one unique value of number of detected hotspots
    mappable = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
    cbar = fig.colorbar(mappable, cax)

    ax.set_axis_off()
    plt.savefig((path_plotting + plotfilename_det + '.png'), dpi = 500)
    #plt.show()     
    plt.close(fig)



    #plotting intensity of hotspots
    fig, ax = plt.subplots(1, figsize=(12,10))
    cmap = colors.ListedColormap(['#fff5f0', '#fdbea5', '#fc7050', '#d42020', '#67000d'])
    norm = colors.BoundaryNorm(boundaries = color_breaks_int, ncolors = 5)
    
    area_shape.plot(ax = ax, color = '#DDDDDD', edgecolor='grey', linewidth = 1)
    data_year.plot(ax = ax, color = 'grey')
    data_year.plot(ax = ax, column='frpMWkm2Avg', cmap = cmap, norm = norm)
    fig.suptitle('Average Intensity of Detected Hotspots: {y}'.format(y=year), fontsize = 18)
    ax.set_title('Fire Radiative Power (FRP) in MW/km2', fontdict={'fontsize': 13})
    
    #axis colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size = '3 %', pad = 0.05)
    vmax=data_year.frpMWkm2Avg.max()
    mappable = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
    cbar = fig.colorbar(mappable, cax)
    
    ax.set_axis_off()
    plt.savefig((path_plotting + plotfilename_int + '.png'), dpi = 500)
    #plt.show() 
    plt.close(fig)
