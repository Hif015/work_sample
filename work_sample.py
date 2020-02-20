###############################################
# EQ Works Sample assignment
# Done by Hilda Faraji,  Febuary 2020

# Imports :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from tabulate import tabulate
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from config import *
from tools import *
from scipy import stats
import os
###############################################
# Data Analysis Module
#####################################################
fig_idx = 1    # index of the figures
# Reading the Input csv files:
df = pd.read_csv('./data/DataSample.csv')
df_POI = pd.read_csv('./data/POIList.csv')

# 1. cleanup the data
print('Stage 1: cleaning Data ...')
# A sample dataset of request logs is given in `data/DataSample.csv`.
# We consider records that have identical `geoinfo` and `timest` as suspicious.
# Please clean up the sample dataset by filtering out
# those suspicious request records.

# Two rows in POI csv file were identical, one removed
df_POI = df_POI.drop_duplicates(subset=['Latitude', 'Longitude'], keep='first')
# Removing Duplicates in DAta as requested in the question
df = df.drop_duplicates(subset=['TimeSt', 'Latitude', 'Longitude'], keep=False)

# Some more analysis of data to make sure the longitudes and Latitudes
# are in the acceptable range for Canada:
# If not I consider them as wrong measures and remove them
# from further analysis

# List the counties in the table:
country_list = df.Country.unique()
# Result: All data in Canada

# We are only looking into Canadian locations ,
# therefore check the valid range for both latitude and longitude:
# I notices wrong values for both longitudes and latitudes:
# These lists are not empty:
geo_range = np.array(canada_georange)

longitude_invalid = df[~df['Longitude'].between(geo_range[0, 0],
                                                geo_range[0, 1])]['Longitude']
latitude_invalid = df[~df['Latitude'].between(geo_range[1, 0],
                                              geo_range[1, 1])]['Latitude']

# Keep only valid geo range for the data
df = df[df['Longitude'].between(geo_range[0, 0], geo_range[0, 1]) &
        df['Latitude'].between(geo_range[1, 0], geo_range[1, 1])]

########################################################################

# 2. Label
# Assign each _request_ (from `data/DataSample.csv`)
# to the closest (i.e. minimum distance) _POI_
# (from `data/POIList.csv`).
# Note: a _POI_ is a geographical Point of Interest.
########################################################################

print('Stage 2: Labeling Data ...')


# Therefore we will keep the right correspondence between min dist
# indexes and their Poi_Id
# row index not necessarily the same as POIID , therefore I keep
# the list of indices
POI_ID_vec = np.array(df_POI['POIID'])
POI_ids = ([int(idd[-1]) for idd in POI_ID_vec])

# For plotting purposes only I calculate the distance
# in terms of degrees.
results = find_min_dist(np.array([df_POI['Latitude'], df_POI['Longitude']]),
                        np.array([df['Latitude'], df['Longitude']]))

# Distance estimation in Km using Haversine Formula
results_haversine = find_min_dist_haversine(np.array([df_POI['Latitude'],
                                                      df_POI['Longitude']]).T,
                                            np.array([df['Latitude'],
                                                      df['Longitude']]).T)
df = df.assign(Poi_Id=results_haversine[:, 1],
               POI_min_dist=results_haversine[:, 0],
               POI_min_dist_degrees=results[:, 0])

df.info()
# an example:
# print(df['Poi_Id'].head(20))
###############################################################################
# 3. Analysis and plots:
# 1 For each _POI_, calculate the average and standard deviation
# of the distance between the _POI_ to each of its assigned _requests_.
# 2 At each _POI_, draw a circle (with the center at the POI)
# that includes all of its assigned _requests_.
# Calculate the radius and density (requests/area) for each _POI_.
###############################################################################

print('Stage 3: Statistics and Analysis ....')

# Initilizing the statistical measures
mean_data = np.zeros(len(POI_ids))
median_data = np.zeros(len(POI_ids))
std_data = np.zeros(len(POI_ids))
mode_data = np.zeros(len(POI_ids))
radius_dist = np.zeros(len(POI_ids))
density = np.zeros(len(POI_ids))

# Visualization of the clusters
fig3, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
fig_idx += 1

for ind in range(len(POI_ids)):
    # find all the requests assigned to the POI[id]
    data_POI_dists = np.array(df[df['Poi_Id'] == ind]['POI_min_dist'])

    mean_data[ind] = np.mean(data_POI_dists)
    median_data[ind] = np.median(data_POI_dists)
    std_data[ind] = np.std(data_POI_dists)
    mode_data[ind] = stats.mode(data_POI_dists)[0]
    radius_dist[ind] = np.max(data_POI_dists)
    # radius of the circle= distance of
    # farthest point to the POI in the cluster
    density[ind] = len(data_POI_dists)/(np.pi * (radius_dist[ind] ** 2))
    # Note on Normal assumptions: The three distribution failed to be normal
    # (as expected due to the nature of data)
    if normality_check_flag:
        # normality tests: Shapiro test, Histogram plots, Q-Q plots
        stat, p = shapiro(data_POI_dists)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p < ALPHA_SHAPIRO:
            print(POI_ID_vec[ind]+':'+'Not a Gaussian Distribution')
        else:
            print(POI_ID_vec[ind]+':'+'Seems like a Gaussian Distribution')

        qqplot(data_POI_dists, line='s')
        plt.title('Q-Q Plot for Normality Check for '+POI_ID_vec[ind])

    # if needed for mode estimation
    plt.subplot(1, 3, ind+1)
    plt.hist(data_POI_dists, bins=30, density=True)
    plt.title(POI_ID_vec[ind]+', Number of Requests = '
                              ''+str(len(data_POI_dists)))
    plt.xlabel('Distance to '+POI_ID_vec[ind])
    plt.ylabel('Pmf')

plt.suptitle('Histogram of the distances')

# Plot with GeoPandas
gdf_POI = gpd.GeoDataFrame(
    df_POI, geometry=gpd.points_from_xy(df_POI.Longitude, df_POI.Latitude))
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# shrinking map to Canada
country = world[world.name == 'Canada']
colors = ['green', 'blue', 'cyan', 'purple']

gdf_POI = gpd.GeoDataFrame(
    df_POI, geometry=gpd.points_from_xy(df_POI.Longitude, df_POI.Latitude))
fig_idx += 1
ax = country.plot(color='white', edgecolor='black', figsize=(20, 10))

density_degree = np.zeros(len(POI_ids))
for ind in range(len(POI_ids)):
    df_per_POI = df[df['Poi_Id'] == ind]
    data_POI_dists_degree = np.array(df_per_POI['POI_min_dist_degrees'])
    # find all the rows that have this POI id
    data_POI_dists = np.array(df_per_POI['POI_min_dist'])
    gdf = gpd.GeoDataFrame(
        df_per_POI, geometry=gpd.points_from_xy(df_per_POI.Longitude,
                                                df_per_POI.Latitude))
    gdf.plot(ax=ax, color=colors[ind], markersize=5, marker='x')
    radius_degree = np.max(data_POI_dists_degree)
    density_degree[ind] = len(data_POI_dists)/(np.pi * (radius_degree ** 2))
    center_POI = [np.array(df_POI['Latitude'])[ind],
                  np.array(df_POI['Longitude'])[ind]]
    circ1 = plt.Circle((center_POI[1], center_POI[0]), radius_degree,
                       color=colors[ind], alpha=.2)
    ax.add_artist(circ1)
    ax.set_aspect(1)
gdf_POI.plot(ax=ax, color='red', markersize=20, marker='o')
plt.ylabel('Latitude')
plt.xlabel('Logitude')
plt.title('Clustering of Geodetic Data around the POIs')

#######################################################################

# 4a. Modeling the Data

# 1. To visualize the popularity of each _POI_,
# they need to be mapped to a scale that ranges from -10 to 10.
# Please provide a mathematical model to implement this,#
# taking into consideration of extreme cases and outliers.
# Aim to be more sensitive around the average and provide
# as much visual differentiability as possible.

# 2.**Bonus**: Try to come up with some reasonable hypotheses
# regarding _POIs_, state all assumptions,
# testing steps and conclusions.
# Include this as a text file (with a name `bonus`)
# in your final submission.
######################################################################
# Outlier Removal:
ax2 = country.plot(color='white', edgecolor='black', figsize=(20, 10))
fig_idx += 1
post_density = np.zeros(len(POI_ids))
requests_num = np.zeros(len(POI_ids))
for ind in range(len(POI_ids)):
    df_per_POI = df[df['Poi_Id'] == ind]
    data_POI_dists_degree = np.array(df_per_POI['POI_min_dist_degrees'])
    # Two stage outlier removal
    # find all the rows that have this POI id
    data_POI_dists = np.array(df_per_POI['POI_min_dist'])
    # Step 1: removing the extreme outliers using quantiles
    q1 = np.quantile(sorted(data_POI_dists), 0.25)
    q3 = np.quantile(sorted(data_POI_dists), 0.75)
    IQR = q3 - q1
    upper_dist = q3 + iqr_thresh_extreme * IQR
    # first refinement
    df_per_POI = df_per_POI[df_per_POI['POI_min_dist'] < upper_dist]

    median_val = np.median(df_per_POI['POI_min_dist'])
    mean_val = np.mean(df_per_POI['POI_min_dist'])
    std_val = np.std(df_per_POI['POI_min_dist'])
    std_val_med = std_median(df_per_POI['POI_min_dist'], median_val)
    # Step 2 : removing the outliers from the remaining data using quantiles
    # or  mean+ constant * std
    q1 = np.quantile(sorted(data_POI_dists), 0.25)
    q3 = np.quantile(sorted(data_POI_dists), 0.75)
    IQR = q3 - q1
    upper_dist_iqr = q3 + iqr_thresh * IQR
    # if Gaussian assumption
    upper_dist_Gaussian_mean = mean_val+std_thresh*std_val
    upper_dist_Gaussian_med = median_val+std_thresh*std_val_med
    # Second refinement : three ways, mean, median or IQR
    df_per_POI = df_per_POI[df_per_POI['POI_min_dist'] < upper_dist_iqr]
    # df_per_POI= df_per_POI[df_per_POI['POI_min_dist']
    # < upper_dist_Gaussian_mean]
    # df_per_POI= df_per_POI[df_per_POI['POI_min_dist']
    # < upper_dist_Gaussian_med]
    df[df['Poi_Id'] == ind] = df_per_POI

    data_POI_dists = np.array(df_per_POI['POI_min_dist'])
    requests_num[ind] = len(data_POI_dists)
    lat_data = np.hstack((np.array(df_POI['Latitude'])[ind],
                          np.array(df_per_POI['Latitude'])))
    long_data = np.hstack((np.array(df_POI['Longitude'])[ind],
                           np.array(df_per_POI['Longitude'])))
    center_POI = [long_data[0], lat_data[0]]
    radius_dist_val = np.max(data_POI_dists)
    post_density[ind] = (len(data_POI_dists)/(np.pi * (radius_dist_val ** 2)))
    # plots:
    plt.plot(long_data, lat_data, 'x', color=colors[ind])
    plt.plot(center_POI[0], center_POI[1], marker='o', markersize=10,
             color='r')
    radius_degree = np.max(df_per_POI['POI_min_dist_degrees'])
    circ1 = plt.Circle((center_POI[0], center_POI[1]), radius_degree,
                       color=colors[ind], alpha=.2)
    ax2.add_artist(circ1)
    ax2.set_aspect(1)


plt.ylabel('Latitude')
plt.xlabel('Logitude')
plt.title('Clustering of Geodetic Data around POIs After Outlier Removal')

# post_histogram plots:

fig3, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
fig_idx += 1

for ind in range(len(POI_ids)):
    # find all the requests assigned to the POI[id]
    data_POI_dists = np.array(df[df['Poi_Id'] == ind]['POI_min_dist'])

    plt.subplot(1, 3, ind+1)
    plt.hist(data_POI_dists, bins=30, density=True)
    plt.title(POI_ID_vec[ind]+', Number of Requests = '
                              ''+str(len(data_POI_dists)))
    plt.xlabel('Distance to '+POI_ID_vec[ind])
    plt.ylabel('Pmf')

plt.suptitle('Histogram of the distances after outlier removal')


# Popularity measures and Scaling:
# Assumption on popularity: The PPOI with the highest number of requests
# Math model for scaling data:
# scaled_data = (b-a)*(X-x.min)/(X.max-X.min)+a
# where [a,b] is the new scale range and X is the data

scaled_popularity_crowd = data_rescale(requests_num, scale_vec)
scaled_popularity_density = data_rescale(post_density, scale_vec)

##########################################################################
# Final results: Figures, Table and savings


# Statistics Printing:

print_stats = list()
print_popularity = list()
for ind in range(len(POI_ids)):
    print_stats.append([POI_ID_vec[ind], mean_data[ind], median_data[ind],
                        std_data[ind], mode_data[ind], radius_dist[ind],
                        density[ind], density_degree[ind], post_density[ind]])
    print_popularity.append([POI_ID_vec[ind], scaled_popularity_crowd[ind],
                             scaled_popularity_density[ind]])
print(tabulate(print_stats, headers=['Cluster_Center', 'Mean_Distance',
                                     'Median_Distance', 'STD_Dist',
                                     'Mode_Dist', 'Cluster_Radius_Km',
                                     'Pre_Density_in_Km',
                                     'Pre_Density_in_Degree',
                                     'Post_Density_in_Km']))
print('\n\n\n')
print(tabulate(print_popularity, headers=['Cluster_Center',
                                          'Scaled_Popularity_Crowd',
                                          'Scaled_Popularity_Density']))

# Save the above results in a CSV file
csv_filename = './output/POI_Statistics.csv'
headers = ['Geodetic_Cluster', 'Mean_Distance', 'Median_Distance', 'STD_Dist',
           'Mode_Dist', 'Cluster_Radius_Km',
           'Pre_Density_in_Km', 'Pre_Density_in_Degree', 'Post_Density_in_Km']
print_data_df = pd.DataFrame(print_stats, columns=headers)
if not os.path.exists('output'):
    os.makedirs('output')
print_data_df.to_csv(csv_filename, index=False)


if plot_flag:
    plt.show()
