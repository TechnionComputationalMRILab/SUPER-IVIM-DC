import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def vtk_read(file_path, b_value, slice_num=0, plot_flag=0, save_plot_flag=0, save_plot_name="python plot images"):
    """
    :param file_path:        DW_MRI images files path, that we taken in different b-values. Type: List
    :param b_value:          The b_values of the images. (Vector)
    :param slice_num:        Optional. If image is 3D, chose a specific 2D slice. Default: 0
    :param plot_flag:        Optional. If 1 than the function will plot the results. Default: 0
    :param save_plot_flag:   Optional. If 1 than the function will save the a plot of the the results. Default: 0
    :param: save_plot_name:  Optional. The name of the file in case of save_plot_flag is 1.
    :return: 2D numpy array: dim: (b_num, y_wid, x_yid)
    """
    # Images to list
    mr_slice = []
    for i in range(len(file_path)):
        print(file_path[i])
        reader = sitk.ImageFileReader()
        reader.SetFileName(file_path[i])
        image = reader.Execute()
        nda = sitk.GetArrayFromImage(image)
        if nda.ndim == 2:
            mr_slice.append(nda)
        elif nda.ndim == 3:
            mr_slice.append(nda[slice_num, :, :])
        else:
            assert (nda.ndim == 2), "Error in image dimensions. Dimensions of image in file must be 2 or 3."


    # plot
    if plot_flag == 1 or save_plot_flag == 1:
        max_value = np.max(mr_slice)  # find max value (for plot clim)
        b_num = len(file_path)
        fig, ax = plt.subplots(2, 4, figsize=(20, 20))
        b_id = 0
        _, b_rem = divmod(b_num, 2)
        for i in range(2):
            for j in range(4):
                if b_rem == 1 and b_id == b_num:
                    ax[i, j].axis('off')
                    break
                ax[i, j].imshow(mr_slice[b_id], cmap='gray', clim=(0, max_value))  # TODO: maybe clim=(0, 1600)
                ax[i, j].set_title("b-value = " + str(b_value[b_id]))
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                b_id += 1
    if save_plot_flag == 1:
        plt.savefig(save_plot_name, transparent=True)
    if plot_flag == 1:
        plt.subplots_adjust(hspace=-0.6)
        plt.show(block=False)

    # Images list to ndarray
    mr_slice = np.array(mr_slice)  # 1st dim: b num , 2st dim: y-axis, 3st dim: x-axis

    return mr_slice