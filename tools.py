#################
# Functions definitions
##################

# Imports##################################
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
###################################################


def find_min_dist(v_poi, v_data):
    # Some dimension manipulation and broadcasting
    # Find distance to all POIS and get the index of closest POI
    temp_x = (v_data[0, :].reshape(v_data.shape[1], 1)-v_poi[0, :])**2
    temp_y = (v_data[1, :].reshape(v_data.shape[1], 1)-v_poi[1, :])**2
    dist_vec = np.sqrt(temp_x+temp_y)
    return np.hstack((np.min(dist_vec, axis=1).reshape(v_data.shape[1], 1),
                      np.argmin(dist_vec, axis=1).reshape(v_data.shape[1], 1)))


def find_min_dist_haversine(v_poi, v_data):
    # convert to radians
    v_data = np.radians(v_data)
    v_poi = np.radians(v_poi)
    dist_vec = haversine_distances(v_data, v_poi)
    # multiply by Earth radius to get kilometers
    dist_vec = dist_vec * 6371000 / 1000
    return np.hstack((np.min(dist_vec, axis=1).reshape(len(v_data), 1),
                      np.argmin(dist_vec, axis=1).reshape(len(v_data), 1)))


def data_rescale(input_data, scale_vec):
    # Math model for scaling data:
    # scaled_data = (b-a)*(X-x.min)/(X.max-X.min)+a
    # where [a,b] is the new scale range and X is the data
    # scaled_data = scale_vec[0]+(scale_vec[1]-scale_vec[0])
    # *(input_data-np.min(input_data))/(np.max(input_data)-np.min(input_data))
    # I have used the numpy functions results are the same as above code
    scaled_data = np.interp(input_data, (input_data.min(), input_data.max()),
                            (scale_vec[0], scale_vec[1]))
    return scaled_data


def std_median(data, med_val):
    return np.sqrt((sum(data-med_val)**2)/(len(data)-1))

######################################################
