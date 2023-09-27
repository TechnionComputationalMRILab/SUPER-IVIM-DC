#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:05:54 2022

@author: noam.korngut@bm.technion.ac.il
"""

import os
import json
import numpy as np
import pandas as pd
import IVIMNET.deep as deep
import matplotlib.pyplot as plt
from source.train_model import train_model
from source.utiles import create_working_folder
from source.hyperparams import hyperparams as hp
from directories import *
from source.Classsic_ivim_fit import IVIM_fit_sls_trf, IVIM_model


if __name__ == "__main__":

    key = 'clinic'

    output_directory = os.path.join(WORKING_DIRECTORY, 'exp2_helthy_vol_study') 
    work_dir = create_working_folder(output_directory)

    # ======================= training =======================

    # for mode in ['IVIMNET', 'SUPER-IVIM-DC']:
    for mode in ['SUPER-IVIM-DC', 'IVIMNET']:
        for sf in range(1,7):
            arg = hp(key)
            arg = deep.checkarg(arg)
            SNR = arg.sim.SNR[0] #50
            bvalues_full = arg.sim.bvalues
            arg.sim.num_samples_eval = 256*256

            # arg.sim.bvalues = np.array(np.concatenate((bvalues_full[0:13:sf],np.array([200]) ,bvalues_full[14:21:j], np.array([1000]))))
            arg.sim.bvalues = np.array(np.concatenate((bvalues_full[0:13:sf],np.array([200]) ,bvalues_full[14:21:sf], np.array([1000]))))

            matNN = train_model(key, arg, mode, sf, work_dir)

    # ======================= evaluate model on clinic data =======================

    RMSE_table_DC, RMSE_table_Net, RMSE_table_lsq = np.zeros((6,3)), np.zeros((6,3)), np.zeros((6,3))
    NRMSE_table_DC, NRMSE_table_Net, NRMSE_table_lsq = np.zeros((6,3)), np.zeros((6,3)), np.zeros((6,3))

    clinic_signal = np.genfromtxt(os.path.join(PROCESSED_DATA_DIRECTORY, 'segmented_human_data.csv'), delimiter=',')

    #normalize signal
    clinic_signal_S0 = clinic_signal[:,0][:, np.newaxis]
    clinic_signal_norm = clinic_signal/clinic_signal_S0

    #ROI = ['kidney_R', 'kidney_L', 'kidney_R2', 'spleen', 'liver1', 'liver2' ]

    params_Net = np.zeros((clinic_signal.shape[0],4,6)) #Label, params, sf
    params_DC = np.zeros((clinic_signal.shape[0],4,6))
    params_lsq = np.zeros((clinic_signal.shape[0],4,6))

    for sf in range(1,7):
        sf_sec = np.array([1,1,2,2,3,3])
        j = sf_sec[sf-1]
        arg.sim.bvalues = np.array(np.concatenate((bvalues_full[0:13:sf],np.array([200]) ,bvalues_full[14:21:j], np.array([1000]))))
        sf_sig = np.array(np.concatenate((np.concatenate((clinic_signal[:,0:13:sf],(clinic_signal[:,13]).reshape(clinic_signal.shape[0],1)),1), np.concatenate((clinic_signal[:,14:21:j], (clinic_signal[:,-1]).reshape(clinic_signal.shape[0],1)),1)),1))

        # ================== LS (sls-trf) fit ====================
        print("**classic fitting - simple map**")
        bounds = [[0.0003, 0.009, 0.001, 0.99],[0.01, 0.04,0.5, 1]] # d,d*,f
        p0 = [((bounds[0][0]+bounds[1][0])/2) ,(bounds[0][1]+bounds[1][1])/2, (bounds[0][2]+bounds[1][2])/2 ,
                                                  ((bounds[0][3]+bounds[1][3])/2)]
        N = 1
        for k in range(sf_sig.shape[0]):
            sig_sls = sf_sig[k,:]/sf_sig[k,0]
            D_sls_trf, DStar_sls_trf, f_sls_trf, s0_sls_trf, _, del_index = IVIM_fit_sls_trf(N, sig_sls[:, np.newaxis], arg.sim.bvalues, bounds, p0, min_bval_high=200)
            params_lsq[k,0, sf-1], params_lsq[k,1, sf-1], params_lsq[k,2, sf-1], params_lsq[k,3,sf-1] = D_sls_trf, f_sls_trf, DStar_sls_trf, s0_sls_trf

        # ================== IVIMNET prediction ==================
        ivimnet_pathSNR50 =  f'{output_directory}/20220224-174831/IVIMNET_50_sf_{sf}.pt' 
        recon_error_net, Dp_net, Dt_net, Fp_net, S0_net = deep.infer_clinical_supervised_IVIM(sf_sig, arg.sim.bvalues, ivimnet_pathSNR50, arg)
        params_Net[:,0,sf-1], params_Net[:,1,sf-1], params_Net[:,2,sf-1], params_Net[:,3,sf-1] = Dt_net, Fp_net, Dp_net, S0_net


        # ================== SUPER-IVIM-DC predication ==================
        ivim_DC_pathSNR50 = f'{output_directory}/20220224-174831/SUPER-IVIM-DC_50_sf_{sf}.pt'
        recon_error_comb, Dp_DC, Dt_DC, Fp_DC, S0_DC = deep.infer_clinical_supervised_IVIM(sf_sig, arg.sim.bvalues, ivim_DC_pathSNR50, arg)
        params_DC[:,0,sf-1], params_DC[:,1,sf-1], params_DC[:,2,sf-1], params_DC[:,3,sf-1] = Dt_DC, Fp_DC, Dp_DC, S0_DC


        # ================== plot signal ==================
        b_vector = np.arange(0,bvalues_full.max())
        for l in range(clinic_signal.shape[0]):

            D_lsq, f_lsq, DStar_lsq, s0_lsq = params_lsq[l,0,0], params_lsq[l,1,0], params_lsq[l,2,0], params_lsq[l,3,0]
            D_DC, f_DC, DStar_DC, s0_DC = params_DC[l,0,sf-1], params_DC[l,1,sf-1], params_DC[l,2,sf-1], params_DC[l,3,sf-1]
            D_Net, f_Net, DStar_Net, s0_Net = params_Net[l,0,sf-1], params_Net[l,1,sf-1], params_Net[l,2,sf-1], params_Net[l,3,sf-1]

            si_fit_lsq = IVIM_model(b_vector.reshape(-1, 1), D_lsq, DStar_lsq, f_lsq, s0_lsq)
            si_fit_DC = IVIM_model(b_vector.reshape(-1, 1), D_DC, DStar_DC, f_DC, s0_DC)
            si_fit_Net = IVIM_model(b_vector.reshape(-1, 1), D_Net, DStar_Net, f_Net, s0_Net)

            plt.figure()
            plt.plot(bvalues,np.log(sf_sig[l,:]/sf_sig[l,0]), 'k.')
            plt.plot(b_vector,np.log(si_fit_lsq), 'b--', label = 'lsq')
            plt.plot(b_vector,np.log(si_fit_DC), 'r--', label = 'IVIM-DC')
            plt.plot(b_vector,np.log(si_fit_Net), 'g--', label = 'IVIM-Net')
            plt.title ('sf '+str(sf)+' '+str([l]))
            plt.legend()


        # ================ NMRSE calculation =======================
        #todo:add nrmse function

        RMSE_table_lsq[sf-1,:] = np.sqrt(np.square(np.subtract(params_lsq[:,0,sf-1] , params_lsq[:,0,0]).mean())), np.sqrt(np.square(np.subtract(params_lsq[:,1,sf-1] , params_lsq[:,1,0]).mean())), np.sqrt(np.square(np.subtract(params_lsq[:,2,sf-1] , params_lsq[:,2,0]).mean())) #np.sqrt(np.sum((params_DC[:,1] - params_lsq[:,1])**2)/len(ROI)), np.sqrt(np.sum((params_DC[:,2,sf-1] - params_lsq[:,2])**2)/len(ROI))
        nRMSE_table_lsq = RMSE_table_lsq[:,0]/np.mean(params_lsq[:,0,0]), RMSE_table_lsq[:,1]/np.mean(params_lsq[:,1,0]), RMSE_table_lsq[:,2]/np.mean(params_lsq[:,2,0])

        RMSE_table_DC[sf-1,:] = np.sqrt(np.square(np.subtract(params_DC[:,0,sf-1] , params_lsq[:,0,0]).mean())), np.sqrt(np.square(np.subtract(params_DC[:,1,sf-1] , params_lsq[:,1,0]).mean())), np.sqrt(np.square(np.subtract(params_DC[:,2,sf-1] , params_lsq[:,2,0]).mean()))#np.sqrt(np.sum((params_DC[:,1] - params_lsq[:,1])**2)/len(ROI)), np.sqrt(np.sum((params_DC[:,2,sf-1] - params_lsq[:,2])**2)/len(ROI))
        RMSE_table_Net[sf-1,:] = np.sqrt(np.square(np.subtract(params_Net[:,0,sf-1] , params_lsq[:,0,0]).mean())), np.sqrt(np.square(np.subtract(params_Net[:,1,sf-1] , params_lsq[:,1,0]).mean())), np.sqrt(np.square(np.subtract(params_Net[:,2,sf-1] , params_lsq[:,2,0]).mean())) #np.sqrt(np.sum((params_Net[:,0,sf-1] - params_lsq[:,0])**2)/len(ROI)), np.sqrt(np.sum((params_Net[:,1] - params_lsq[:,1])**2)/len(ROI)), np.sqrt(np.sum((params_Net[:,2,sf-1] - params_lsq[:,2])**2)/len(ROI))
        RMSE_DC = pd.DataFrame(data=RMSE_table_DC, index=['sf_1', 'sf_2', 'sf_3', 'sf_4', 'sf_5', 'sf_6'], columns=['D', 'F', 'D*'])
        RMSE_Net = pd.DataFrame(data=RMSE_table_Net, index=['sf_1', 'sf_2', 'sf_3', 'sf_4', 'sf_5', 'sf_6'], columns=['D', 'F', 'D*'])
        
        nRMSE_table_DC= RMSE_table_DC[:,0]/np.mean(params_lsq[:,0,0]), RMSE_table_DC[:,1]/np.mean(params_lsq[:,1,0]), RMSE_table_DC[:,2]/np.mean(params_lsq[:,2,0])
        nRMSE_table_Net = RMSE_table_Net[:,0]/np.mean(params_lsq[:,0,0]), RMSE_table_Net[:,1]/np.mean(params_lsq[:,1,0]), RMSE_table_Net[:,2]/np.mean(params_lsq[:,2,0])

        save = 0
        if (save):
            np.savetxt(os.path.join(SIGNALS_DIRECTORY, 'healthy_vol_study', 'exp2_IVIMNET_nrmse_table.csv'), np.asarray(nRMSE_table_Net), delimiter=",")
            np.savetxt(os.path.join(SIGNALS_DIRECTORY, 'healthy_vol_study', 'exp2_SUPER-IVIM-DC_nrmse_table.csv'), np.asarray(nRMSE_table_DC), delimiter=",")

    # plot results
    labels = ['D', 'f', 'D*']
    for i,label in enumerate(['D', 'f', 'D*']):
        plt.figure()
        plt.plot(range(1,7), nRMSE_table_DC[i], label = 'SUPER-IVIM-DC')
        plt.plot(range(1,7), nRMSE_table_Net[i], label = 'IVIMNET')
        plt.scatter(range(1,7), nRMSE_table_Net[i], color='orange')
        plt.scatter(range(1,7), nRMSE_table_DC[i], marker='*', color='blue', s=100)
        plt.title(label, fontdict={'fontsize': 25})
        plt.xlabel("sampling factor", fontdict={'fontsize': 22})
        plt.ylabel("nrmse", fontdict={'fontsize': 22})
        plt.legend(prop={'size': 13})
 
        plt.savefig(os.path.join(FIGURE_DIRECTORY, f'exp2_{label}_lsq_nrmse.eps'), format='eps', dpi=300, bbox_inches = 'tight')

  
       

