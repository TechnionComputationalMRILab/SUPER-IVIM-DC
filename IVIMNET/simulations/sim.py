import numpy as np
import time
import scipy.stats as scipy
import torch
import matplotlib.pyplot as plt

from IVIMNET.utils.checkarg import checkarg
from IVIMNET.simulations.sim_signal import sim_signal
from IVIMNET.deep.learn import learn_IVIM, learn_supervised_IVIM
from IVIMNET.deep.predict import predict_IVIM, predict_supervised_IVIM
from IVIMNET.fitting_algorithms import fit_dats


def sim(SNR, arg, supervised=False):
    """ This function defines how well the different fit approaches perform on simulated data. Data is simulated by
    randomly selecting a value of D, f and D* from within the predefined range. The script calculates the random,
    systematic, root-mean-squared error (RMSE) and Spearman Rank correlation coefficient for each of the IVIM parameters.
    Furthermore, it calculates the stability of the neural network (when trained multiple times).

    input:
    :param SNR: SNR of the simulated data. If SNR is set to 0, no noise is added
    :param arg: an object with simulation options. hyperparams.py gives most details on the object (and defines it),
    Relevant attributes are:
    arg.sim.sims = number of simulations to be performed (need a large amount for training)
    arg.sim.num_samples_eval = number of samples to evaluate (save time for lsq fitting)
    arg.sim.repeats = number of times to repeat the training and evaluation of the network (to assess stability)
    arg.sim.bvalues: 1D Array of b-values used
    arg.fit contains the parameters regarding lsq fitting
    arg.train_pars and arg.net_pars contain the parameters regarding the neural network
    arg.sim.range gives the simulated range of D, f and D* in a 2D array

    :return matlsq: 2D array containing the performance of the lsq fit (if enabled). The rows indicate D, f (Fp), D*
    (Dp), whereas the colums give the mean input value, the random error and the systematic error
    :return matNN: 2D array containing the performance of the NN. The rows indicate D, f (Fp), D*
    (Dp), whereas the colums give the mean input value, the random error and the systematic error
    :return stability: a 1D array with the stability of D, f and D* as a fraction of their mean value.
    Stability is only relevant for neural networks and is calculated from the repeated network training.
    """
    arg = checkarg(arg)
    # this simulated the signal
    IVIM_signal_noisy, D, f, Dp = sim_signal(SNR, arg.sim.bvalues, sims=arg.sim.sims, Dmin=arg.sim.range[0][0],
                                             Dmax=arg.sim.range[1][0], fmin=arg.sim.range[0][1],
                                             fmax=arg.sim.range[1][1], Dsmin=arg.sim.range[0][2],
                                             Dsmax=arg.sim.range[1][2], rician=arg.sim.rician, key=arg.key)

    # prepare a larger array in case we repeat training
    if arg.sim.repeats > 1:
        paramsNN = np.zeros([arg.sim.repeats, 4, arg.sim.num_samples_eval])
    else:
        paramsNN = np.zeros([4, arg.sim.num_samples_eval])

    # if we are not skipping the network for evaluation
    if not arg.train_pars.skip_net:
        # loop over repeats
        for aa in range(arg.sim.repeats):
            start_time = time.time()
            # train network
            print('\nRepeat: {repeat}\n'.format(repeat=aa))
            # supervised addition
            if supervised:
                # combine the ouput and the IVIM parameters as labels here
                print('Supervised Training')
                labels = np.stack((D, f, Dp), axis=1).squeeze()
                net = learn_supervised_IVIM(IVIM_signal_noisy, labels, arg.sim.bvalues,
                                                 arg)  # add the labels to this function
            else:
                net = learn_IVIM(IVIM_signal_noisy, arg.sim.bvalues, arg)
            elapsed_time = time.time() - start_time
            print('\ntime elapsed for training: {}\n'.format(elapsed_time))
            start_time = time.time()
            # predict parameters
            if arg.sim.repeats > 1:
                paramsNN[aa] = predict_IVIM(IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.sim.bvalues, net,
                                                 arg)  # size?
            else:
                if supervised:
                    print('Supervised Prediiction')
                    paramsNN = predict_supervised_IVIM(IVIM_signal_noisy[:arg.sim.num_samples_eval, :],
                                                       labels[:arg.sim.num_samples_eval, :], arg.sim.bvalues, net,
                                                       arg)
                else:
                    paramsNN = predict_IVIM(IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.sim.bvalues, net,
                                            arg)
            elapsed_time = time.time() - start_time
            print('\ntime elapsed for inference: {}\n'.format(elapsed_time))
            # remove network to save memory

            # del net # I need the net after training

            if arg.train_pars.use_cuda:
                torch.cuda.empty_cache()
        print('results for NN')
        # if we repeat training, then evaluate stability
        # only remember the D, Dp and f needed for evaluation
        D_eval = D[:arg.sim.num_samples_eval]
        Dp_eval = Dp[:arg.sim.num_samples_eval]
        f_eval = f[:arg.sim.num_samples_eval]
        if arg.sim.repeats > 1:
            matNN = np.zeros([arg.sim.repeats, 3, 3])
            for aa in range(arg.sim.repeats):
                # determine errors and Spearman Rank
                matNN[aa] = print_errors(np.squeeze(D_eval), np.squeeze(f_eval), np.squeeze(Dp_eval), paramsNN[aa])
            matNN = np.mean(matNN, axis=0)
            # calculate Stability Factor
            stability = np.sqrt(np.mean(np.square(np.std(paramsNN, axis=0)), axis=1))
            stability = stability[[0, 1, 2]] / [np.mean(D_eval), np.mean(f_eval), np.mean(Dp_eval)]
            # set paramsNN for the plots
            paramsNN_0 = paramsNN[0]
        else:
            matNN = print_errors(np.squeeze(D_eval), np.squeeze(f_eval), np.squeeze(Dp_eval), paramsNN)
            stability = np.zeros(3)
            paramsNN_0 = paramsNN
        # del paramsNN
        # show figures if requested
        plots(arg, D_eval, Dp_eval, f_eval, paramsNN)
    else:
        # if network is skipped
        stability = np.zeros(3)
        matNN = np.zeros([3, 5])
    if arg.fit.do_fit:
        start_time = time.time()
        # all fitting is done in the fit.fit_dats for the other fitting algorithms (lsq, segmented and Baysesian)
        paramsf = fit_dats(arg.sim.bvalues, IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.fit)
        elapsed_time = time.time() - start_time
        print('\ntime elapsed for fit: {}\n'.format(elapsed_time))
        print('results for fit')
        # determine errors and Spearman Rank
        matlsq = print_errors(np.squeeze(D_eval), np.squeeze(f_eval), np.squeeze(Dp_eval), paramsf)
        # del paramsf, IVIM_signal_noisy
        # show figures if requested
        plots(arg, D_eval, Dp_eval, f_eval, paramsf)
        return matlsq, matNN, stability
    else:
        # if lsq fit is skipped, don't export lsq results
        return matNN, stability


def plots(arg,D,Dp,f,params):
    if arg.fig:
        dummy = np.array(params)
        # plot correlations
        plt.figure()
        plt.plot(D[:1000], Dp[:1000], 'rx', markersize=5)
        plt.xlim(0, 0.005)
        plt.ylim(0, 0.3)
        plt.xlabel('Dt')
        plt.ylabel('Dp')
        plt.gcf()
        # plt.savefig('plots/inputDtDp.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(f[:1000], Dp[:1000], 'rx', markersize=5)
        plt.xlim(0, 0.6)
        plt.ylim(0, 0.3)
        plt.xlabel('f')
        plt.ylabel('Dp')
        plt.gcf()
        # plt.savefig('plots/inputfDp.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(D[:1000], f[:1000], 'rx', markersize=5)
        plt.xlim(0, 0.005)
        plt.ylim(0, 0.6)
        plt.xlabel('Dt')
        plt.ylabel('f')
        plt.gcf()
        # plt.savefig('plots/inputDtf.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(dummy[0, :1000], dummy[2, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.005)
        plt.ylim(0, 0.3)
        plt.xlabel('Dt')
        plt.ylabel('Dp')
        plt.gcf()
        # plt.savefig('plots/DtDp.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(dummy[1, :1000], dummy[2, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.6)
        plt.ylim(0, 0.3)
        plt.xlabel('f')
        plt.ylabel('Dp')
        plt.gcf()
        # plt.savefig('plots/fDp.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(dummy[0, :1000], dummy[1, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.005)
        plt.ylim(0, 0.6)
        plt.xlabel('Dt')
        plt.ylabel('f')
        plt.gcf()
        # plt.savefig('plots/Dtf.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(Dp[:1000], dummy[2, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.3)
        plt.ylim(0, 0.3)
        plt.ylabel('DpNN')
        plt.xlabel('Dpin')
        plt.gcf()
        # plt.savefig('plots/DpoutDpin.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(D[:1000], dummy[0, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.005)
        plt.ylim(0, 0.005)
        plt.ylabel('DtNN')
        plt.xlabel('Dtin')
        plt.gcf()
        # plt.savefig('plots/DtoutDtin.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(f[:1000], dummy[1, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.6)
        plt.ylim(0, 0.6)
        plt.ylabel('fNN')
        plt.xlabel('fin')
        plt.gcf()
        # plt.savefig('plots/foutfin.png')
        plt.ion()
        plt.show()
        #plt.close('all') # Keep the plot open/close them


def print_errors(D, f, Dp, params):
    # this function calculates and prints the random, systematic, root-mean-squared (RMSE)
    # errors and Spearman Rank correlation coefficient

    rmse_D = np.sqrt(np.square(np.subtract(D, params[0])).mean())
    rmse_f = np.sqrt(np.square(np.subtract(f, params[1])).mean())
    rmse_Dp = np.sqrt(np.square(np.subtract(Dp, params[2])).mean())

    # initialise Spearman Rank matrix
    Spearman = np.zeros([3, 2])
    # calculate Spearman Rank correlation coefficient and p-value
    Spearman[0, 0], Spearman[0, 1] = scipy.spearmanr(params[0], params[2])  # DvDp
    Spearman[1, 0], Spearman[1, 1] = scipy.spearmanr(params[0], params[1])  # Dvf
    Spearman[2, 0], Spearman[2, 1] = scipy.spearmanr(params[1], params[2])  # fvDp
    # If spearman is nan, set as 1 (because of constant estimated IVIM parameters)
    Spearman[np.isnan(Spearman)] = 1
    # take absolute Spearman
    Spearman = np.absolute(Spearman)
    del params

    normD_lsq = np.mean(D)
    normf_lsq = np.mean(f)
    normDp_lsq = np.mean(Dp)

    print('\nresults from NN: columns show themean, the SD/mean, the systematic error/mean, '
          'the RMSE/mean and the Spearman coef [DvDp,Dvf,fvDp] \n'
          'the rows show D, f and D*\n')
    print([normD_lsq, '  ', rmse_D / normD_lsq, ' ', Spearman[0, 0]])
    print([normf_lsq, '  ', rmse_f / normf_lsq, ' ', Spearman[1, 0]])
    print([normDp_lsq, '  ', rmse_Dp / normDp_lsq,' ', Spearman[2, 0]])

    mats = [[normD_lsq, rmse_D / normD_lsq, Spearman[0, 0]],
            [normf_lsq, rmse_f / normf_lsq, Spearman[1, 0]],
            [normDp_lsq, rmse_Dp / normDp_lsq, Spearman[2, 0]]]

    return mats
