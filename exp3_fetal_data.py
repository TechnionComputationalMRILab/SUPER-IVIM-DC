# -*- coding: utf-8 -*-


#todo: remove unused packages and functions

import os
import json
import numpy as np
import pandas as pd
from source.plot import *
import IVIMNET.deep as deep
import matplotlib.pyplot as plt
from source.train_model import train_model
from source.utiles import * #create_working_folder, read_data_sitk
from directories import *
from source.hyperparams import hyperparams as hp
from source.Classsic_ivim_fit import IVIM_fit_sls_trf


if __name__ == "__main__":

    key = 'fetal'

    minor_motion = False #True # set to False to get all cases parameters estimation

    cases = os.path.join(DATA_DIRECTORY, "cases")
    segs = os.path.join(DATA_DIRECTORY, "segs")

    output_directory = os.path.join(WORKING_DIRECTORY, 'exp3_fetal') 
    work_dir = create_working_folder(output_directory)

    # ====================== training ========================

    arg = hp(key)
    arg = deep.checkarg(arg)
    bvalues = arg.sim.bvalues #np.array([0,50,100,200,400,600])
    SNR = arg.sim.SNR[0]
    arg.sim.num_samples_eval = 256*256

    for mode in ['IVIMNET', 'SUPER-IVIM-DC']:
        matNN = train_model(key, arg, mode, 1, work_dir)

    # ================== load data ====================

    if (minor_motion):
        cases_idx = np.array([647, 693, 697, 710, 756, 794, 798, 800, 801, 807, 817, 819, 828, 844, 856, 860, 877, 887]) # minor motion cases
        GA = np.genfromtxt(os.path.join(SIGNALS_DIRECTORY, 'Fetal', 'GA_minor_motion.csv'), delimiter=',') 
    else:
        cases_idx = [filename for filename in enumerate(os.listdir(cases))]
        cases_idx = np.flip(np.array(cases_idx)[:,1])
        cases_idx = np.array([np.char.replace(np.char.replace(s,'case',''),'.nii', '') for s in cases_idx])
        GA = np.genfromtxt(os.path.join(SIGNALS_DIRECTORY, 'Fetal', 'GA_all_cases.csv'), delimiter=',') 

    num_cases = len(cases_idx)

    fetal_data = np.genfromtxt(os.path.join(DATA_DIRECTORY, 'fetal', f'fetal_mean_signals_{num_cases}.csv'), delimiter=',')
    fetal_S0 = fetal_data[:,0][:, np.newaxis]
    fetal_data_norm = fetal_data/fetal_S0 # normalize data by S0

    # ================== LS (sls-trf) fit ====================
    params_sls_trf = np.zeros((num_cases,4))

    N = 1
    bounds = [[0.0003, 0.009, 0.001, 0.99],[0.01, 0.04,0.5, 1]]  # d,d*,f,so
    p0 = [((bounds[0][0]+bounds[1][0])/2) ,(bounds[0][1]+bounds[1][1])/2, (bounds[0][2]+bounds[1][2])/2 ,
          ((bounds[0][3]+bounds[1][3])/2)]
    for i in range(num_cases):
        D_sls_trf, DStar_sls_trf, f_sls_trf, s0_sls_trf, _ , del_index = IVIM_fit_sls_trf(1, fetal_data_norm[i,:][:, np.newaxis], bvalues, bounds, p0, min_bval_high=200)
        params_sls_trf[i, 0], params_sls_trf[i, 1], params_sls_trf[i, 2], params_sls_trf[i, 3] = D_sls_trf, f_sls_trf, DStar_sls_trf, s0_sls_trf

        #save params
        save = 0
        if (save):
            np.savetxt(os.path.join(SIGNALS_DIRECTORY, 'Fetal', f'exp3_LS_params_{num_cases}_{bounds}.csv'), np.asarray(params_sls_trf), delimiter=",")

    # ======================= DL fit ========================

    IVIMNET_params, DC_params = np.zeros((num_cases, 4)), np.zeros((num_cases, 4))

    # ================== IVIMNET prediction ==================
    ivimnet_pathSNR10 = f'{output_directory}/20220221-173402/IVIMNET_10.pt'

    recon_error_net, Dp_net, Dt_net, Fp_net, S0_net = deep.infer_clinical_supervised_IVIM(fetal_data_norm, bvalues, ivimnet_pathSNR10, arg)
    IVIMNET_params = (Dt_net), (Fp_net), (Dp_net), (S0_net)

    # ================== SUPER-IVIM-DC prediction ==================

    ivim_DC_pathSNR10 = f'{output_directory}/20220221-173402/SUPER-IVIM-DC_10.pt'

    recon_error_comb, Dp_DC, Dt_DC, Fp_DC, S0_DC = deep.infer_clinical_supervised_IVIM(fetal_data_norm, bvalues, ivim_DC_pathSNR10, arg)
    DC_params = (Dt_DC), (Fp_DC), (Dp_DC), (S0_DC)

    # ================== plot signal ==================

    for i in range(num_cases):
        plt.figure()
        plot_IVIM_signal((S0_net[i]), (Dt_net[i]), (Fp_net[i]), (Dp_net[i]), cases_idx[i], bvalues, fetal_data_norm[i,:], 'IVIMNET')
        plot_IVIM_signal((S0_DC[i]), (Dt_DC[i]), (Fp_DC[i]), (Dp_DC[i]), cases_idx[i], bvalues, fetal_data_norm[i,:], 'SUPER-IVIM-DC')
        plot_IVIM_signal(params_sls_trf[i, 3], params_sls_trf[i, 0], params_sls_trf[i, 1], params_sls_trf[i, 2], cases_idx[i], bvalues, fetal_data_norm[i,:], 'LS')
        plt.show()

    IVIMNET_params_csv, DC_params_csv = np.asarray(IVIMNET_params), np.asarray(DC_params)

    np.savetxt(os.path.join(SIGNALS_DIRECTORY, "Fetal", f'IVIMNET_params_{num_cases}.csv'), IVIMNET_params_csv, delimiter=",")
    np.savetxt(os.path.join(SIGNALS_DIRECTORY, "Fetal", f'DC_params_{num_cases}.csv'), IVIMNET_params_csv, delimiter=",")

    # ================== plot correlations ==================

    F_IVIMNET, F_SUPER_IVIM_DC, F_LS = IVIMNET_params[1], DC_params[1], params_sls_trf[:,1] #IVIMNET_params[:,1], DC_params[:,1], params_sls_trf[:,1]

    # all GA

    fig = plt.figure(figsize=(18, 4))
    plt.subplot(131)
    plot_corr(GA, F_LS, 'LS', two_stage = False)
    plt.subplot(132)
    plot_corr(GA, F_IVIMNET, 'IVIMNET', two_stage = False)
    plt.subplot(133)
    plot_corr(GA, F_SUPER_IVIM_DC, 'SUPER_IVIM_DC', two_stage = False)
    plt.show()

    plt.savefig(os.path.join(FIGURE_DIRECTORY, 'exp3_fetal', f'all_corr_{num_cases}.eps'), format='eps', dpi=300, bbox_inches = 'tight')

    # By division to GA

    fig = plt.figure(figsize=(18, 4))
    plt.subplot(131)
    plot_corr(GA, F_LS, 'LS', two_stage = True)
    plt.subplot(132)
    plot_corr(GA, F_IVIMNET, 'IVIMNET', two_stage = True)
    plt.subplot(133)# create a subplot of certain size
    plot_corr(GA, F_SUPER_IVIM_DC, 'SUPER_IVIM_DC', two_stage = True)
    plt.show()

    plt.savefig(os.path.join(FIGURE_DIRECTORY, 'exp3_fetal', f'two_part_corr_{num_cases}.eps'), format='eps', dpi=300, bbox_inches = 'tight')



