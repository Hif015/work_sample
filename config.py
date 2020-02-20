###########################
# Parameters Definitions:
###########################
canada_georange = [[-141, -50], [42, 83]]  # Longitude and latitude range
scale_vec = [-10, 10]   # defined in the question to rescale POI
ALPHA_SHAPIRO = 0.05  # to compare the p value used a Normality
# check for distribution
iqr_thresh_extreme = 3  # for removing Extreme thresholds.
# The lower the value the more aggressive the outlier removal
iqr_thresh = 1.5  # for removing thresholds.
std_thresh = 1  # if normal distribution is assumed
normality_check_flag = False  # To check if the distribution is similar
# to a normal distribution
plot_flag = False  # If True the figures are shown
z_thresh = 2.5  # Threshold for z score
# Different methods to tackle outliers:
outlier_removal_method = {'qantile': True,
                          'mean_std': False,
                          'median_stdmed': False}
