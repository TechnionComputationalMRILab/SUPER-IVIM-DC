#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Feb 13 17:01:47 2022

@author: noam.korngut@bm.technion.ac.il
"""

import os
import json
import time
import numpy as np
import pandas as pd
import IVIMNET.deep as deep
import matplotlib.pyplot as plt
from source.train_model import train_model
from source.utiles import create_working_folder
from directories import *
from source.hyperparams import hyperparams as hp


if __name__ == "__main__":

    key = 'sim'

    output_directory = os.path.join(WORKING_DIRECTORY, "exp1_simulations") 
    work_dir = create_working_folder(output_directory)

    # ======================= Training =======================
    for mode in ['IVIMNET', 'SUPER-IVIM-DC']:
        for sf in range(1,7):
            arg = hp(key)
            arg = deep.checkarg(arg)
            SNR = arg.sim.SNR[0]
            bvalues = arg.sim.bvalues
            arg.sim.num_samples_eval = 256*256
            arg.sim.bvalues = np.array(np.concatenate((bvalues[0:12:sf], np.array([200,400,600,800]))))

            # ======================= Save init =======================
            init_settings = dict(range = arg.sim.range, cons_max = arg.net_pars.cons_max, cons_min = arg.net_pars.cons_min, bvalues = arg.sim.bvalues, loss_coef = arg.loss_coef_ivim, snr= SNR)
            with open(f'{work_dir}/init/{mode}_sf_{sf}.json', 'w') as fp:
                json.dump(init_settings, fp, default=str, indent=4, sort_keys=True)

            matNN = train_model(key, arg, mode, sf, work_dir)

            matNN = np.asarray(matNN)
            np.savetxt(f'{work_dir}/exp1_{mode}_NRMSE_snr_{SNR}_sf_{sf}.csv', matNN, delimiter=",")

    # ================== plot NRMSE tables ====================

    NRMSE_table = np.zeros((2,6,3)) # mode (IVIMNET, SUPER_IVIM_DC), sf (1-6), param (D,f,D*)

    for sf in range (1,7):
        for i, mode in enumerate(['IVIMNET', 'SUPER-IVIM-DC']):

            nrmse = np.genfromtxt(f'{work_dir}/exp1_{mode}_NRMSE_snr_{SNR}_sf_{sf}.csv', delimiter=',')
            NRMSE_table[i,sf-1,:] = nrmse[:,3]

    for i, label in enumerate(['D', 'f', 'D*']):
        plt.figure()
        plt.plot(range(1,7), NRMSE_table[1,:,i] ,label = 'SUPER-IVIM-DC')
        plt.plot(range(1,7), NRMSE_table[0,:,i] ,label = 'IVIMNET')
        plt.scatter(range(1,7), NRMSE_table[0,:,i], color='orange')
        plt.scatter(range(1,7), NRMSE_table[1,:,i], marker='*', color='blue', s=100)
        plt.title(f'{label}', fontdict={'fontsize': 25})
        plt.xlabel('sf', fontdict={'fontsize': 22})
        plt.ylabel('nrmse', fontdict={'fontsize': 22})
        plt.legend(prop={'size': 16})
        plt.show()

        plt.savefig(os.path.join(FIGURE_DIRECTORY, 'exp1_simulations', f'exp1_{label}.eps'), format='eps', dpi=300)
