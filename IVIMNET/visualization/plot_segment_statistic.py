import numpy as np
import matplotlib.pyplot as plt


def segment_statistic(param_map, segmented_image, n_labels, show_plot=False):
    """
    This function take param map image such as D/D*/Fp and calculate the statisctics on every segment.
    :param map: is full path with file name and extention
    :param segmented_image : ITK SNAP segmented file
    return std_series, mean_series: statistic of every ROI (region of interest)

    """
    if show_plot:
        plt.figure()
        plt.imshow(param_map, cmap = 'gray')

    labels_vector = np.unique(segmented_image)

    std_series = np.zeros((n_labels + 1, 1))  # plus one because label 0 is the background
    mean_series = np.zeros((n_labels + 1, 1))  # plus one because label 0 is the background
    coefficient_of_variation = np.zeros((n_labels + 1, 1))

    for label in labels_vector:
        if label == 0:
            continue

        bool_map = segmented_image == label
        segmented_voxels = param_map * bool_map

        if show_plot:
            plt.figure()
            plt.imshow(bool_map[0,])
            plt.figure()
            plt.imshow(segmented_voxels[0,])

        std_series[label, 0] = param_map[bool_map[0,]].std()
        mean_series[label, 0] = param_map[bool_map[0,]].mean()
        coefficient_of_variation[label, 0] = std_series[label] / mean_series[label]

    return std_series, mean_series, labels_vector, coefficient_of_variation
