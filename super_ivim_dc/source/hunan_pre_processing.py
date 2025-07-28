# -*- coding: utf-8 -*-

import numpy as np
import SimpleITK as sitk
import glob
from scipy.stats import gmean
import os
from directories import *
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # ================= Pre Processing DW-MRI Data : =================

    processed_folder = PROCESSED_DATA_DIRECTORY
    
    files_list = glob.glob(f'{processed_folder}b_*.nii.gz')
    files_list = np.array([file.replace('\\','/') for file in files_list ])
    bvals = np.array([float(file[len(f'{processed_folder}b_'):-7]) for file in files_list])

    bvals_order = np.argsort(bvals)
    bvals = bvals[bvals_order]
    files_list = files_list[bvals_order]

    seg_file_name =  f'{processed_folder}seg.nii.gz'
    sitk_seg_image = sitk.ReadImage(seg_file_name)
    seg_image = sitk.GetArrayFromImage(sitk_seg_image)

    labels = np.unique(seg_image)
    labels = labels[1:]

    signals = np.zeros((len(labels), len(bvals)))

    for label in labels:
        bval_signals = np.zeros(bvals.shape)
        for i, bval_filename in enumerate(files_list):
            sitk_bval_image = sitk.ReadImage(bval_filename)
            bval_image = sitk.GetArrayFromImage(sitk_bval_image)
            si = bval_image[seg_image==label].mean()
            bval_signals[i] = si

        signals[label-1,:] = bval_signals
        signals_csv = np.asarray(signals)

        np.savetxt(os.path.join(PROCESSED_DATA_DIRECTORY, 'segmented_human_data.csv'), signals_csv, delimiter=",")
