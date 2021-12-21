from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from pydicom import dcmread


def dicom_to_numpy(path, show_plot=False):
    """
    Depreciated. Replaced with loading the NPZ files
    """

    dicom_paths = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    for sx in dicom_paths:
        ds = dcmread(sx)
        numpy_arr = ds.pixel_array

        if show_plot:
            plt.imshow(numpy_arr, cmap="gray")
            plt.show()
    return numpy_arr
