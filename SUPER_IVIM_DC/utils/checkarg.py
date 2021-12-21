import torch
import warnings
from SUPER_IVIM_DC.utils.parameters import net_pars, train_pars, sim, lsqfit


def checkarg(arg):
    if not hasattr(arg, 'fig'):
        arg.fig = False
        warnings.warn('arg.fig not defined. Using default of False')
    if not hasattr(arg, 'save_name'):
        warnings.warn('arg.save_name not defined. Using default of ''default''')
        arg.save_name = 'default'
    if not hasattr(arg, 'net_pars'):
        warnings.warn('arg no net_pars. Using default initialisation')
        arg.net_pars = net_pars()
    if not hasattr(arg, 'train_pars'):
        warnings.warn('arg no train_pars. Using default initialisation')
        arg.train_pars = train_pars()
    if not hasattr(arg, 'sim'):
        warnings.warn('arg no sim. Using default initialisation')
        arg.sim = sim()
    if not hasattr(arg, 'fit'):
        warnings.warn('arg no lsq. Using default initialisation')
        arg.fit = lsqfit()

    arg.net_pars = checkarg_net_pars(arg.net_pars)
    arg.train_pars = checkarg_train_pars(arg.train_pars)
    arg.sim = checkarg_sim(arg.sim)
    arg.fit = checkarg_lsq(arg.fit)

    return arg


def checkarg_lsq(arg):
    if not hasattr(arg, 'method'):
        warnings.warn('arg.fit.method not defined. Using default of ''lsq''')
        arg.method = 'lsq'
    if not hasattr(arg, 'do_fit'):
        warnings.warn('arg.fit.do_fit not defined. Using default of True')
        arg.do_fit = True
    if not hasattr(arg, 'load_lsq'):
        warnings.warn('arg.fit.load_lsq not defined. Using default of False')
        arg.load_lsq = False
    if not hasattr(arg, 'fitS0'):
        warnings.warn('arg.fit.fitS0 not defined. Using default of False')
        arg.fitS0 = False
    if not hasattr(arg, 'jobs'):
        warnings.warn('arg.fit.jobs not defined. Using default of 4')
        arg.jobs = 4
    if not hasattr(arg, 'bounds'):
        warnings.warn('arg.fit.bounds not defined. Using default.')
        arg.bounds = ([0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3])  # Dt, Fp, Ds, S0
    return arg


def checkarg_sim(arg):
    if not hasattr(arg, 'bvalues'):
        warnings.warn('arg.sim.bvalues not defined. Using default value')
        arg.bvalues = [0, 5, 10, 20, 30, 40, 60, 150, 300, 500, 700]
    if not hasattr(arg, 'repeats'):
        warnings.warn('arg.sim.repeats not defined. Using default value of 1')
        arg.repeats = 1  # this is the number of repeats for simulations
    if not hasattr(arg, 'rician'):
        warnings.warn('arg.sim.rician not defined. Using default of False')
        arg.rician = False
    if not hasattr(arg, 'SNR'):
        warnings.warn('arg.sim.SNR not defined. Using default of [20]')
        arg.SNR = [20]
    if not hasattr(arg, 'sims'):
        warnings.warn('arg.sim.sims not defined. Using default of 100000')
        arg.sims = 100000
    if not hasattr(arg, 'num_samples_eval'):
        warnings.warn('arg.sim.num_samples_eval not defined. Using default of 100000')
        arg.num_samples_eval = 100000
    if not hasattr(arg, 'range'):
        warnings.warn('arg.sim.range not defined. Using default.')
        arg.range = ([0.0005, 0.05, 0.01],
                     [0.003, 0.4, 0.1])
    return arg


def checkarg_net_pars(arg):
    """ see parameters.net_pars for the attribute descriptions """
    if not hasattr(arg, 'dropout'):
        warnings.warn('arg.net_pars.dropout not defined. Using default value of 0.1')
        arg.dropout = 0.1
    if not hasattr(arg, 'batch_norm'):
        warnings.warn('arg.net_pars.batch_norm not defined. Using default of True')
        arg.batch_norm = True
    if not hasattr(arg, 'parallel'):
        warnings.warn('arg.net_pars.parallel not defined. Using default of True')
        arg.parallel = True
    if not hasattr(arg, 'con'):
        warnings.warn('arg.net_pars.con not defined. Using default of ''sigmoid''')
        arg.con = 'sigmoid'
    if not hasattr(arg, 'cons_min'):
        warnings.warn('arg.net_pars.cons_min not defined. Using default values of  [-0.0001, -0.05, -0.05, 0.7]')
        arg.cons_min = [-0.0001, -0.05, -0.05, 0.7, -0.05, 0.06]
    if not hasattr(arg, 'cons_max'):
        warnings.warn('arg.net_pars.cons_max not defined. Using default values of  [-0.0001, -0.05, -0.05, 0.7]')
        arg.cons_max = [0.005, 0.7, 0.3, 1.3, 0.3, 0.3]
    if not hasattr(arg, 'fitS0'):
        warnings.warn('arg.net_pars.parallel not defined. Using default of False')
        arg.fitS0 = False
    if not hasattr(arg, 'depth'):
        warnings.warn('arg.net_pars.depth not defined. Using default value of 4')
        arg.depth = 4
    if not hasattr(arg, 'width'):
        warnings.warn('arg.net_pars.width not defined. Using default of number of b-values')
        arg.width = 0
    return arg


def checkarg_train_pars(arg):
    if not hasattr(arg, 'optim'):
        warnings.warn('arg.train.optim not defined. Using default ''adam''')
        arg.optim = 'adam'
    if not hasattr(arg, 'lr'):
        warnings.warn('arg.train.lr not defined. Using default value 0.0001')
        arg.lr = 0.0001
    if not hasattr(arg, 'patience'):
        warnings.warn('arg.train.patience not defined. Using default value 10')
        arg.patience = 10
    if not hasattr(arg, 'batch_size'):
        warnings.warn('arg.train.batch_size not defined. Using default value 128')
        arg.batch_size = 128
    if not hasattr(arg, 'maxit'):
        warnings.warn('arg.train.maxit not defined. Using default value 500')
        arg.maxit = 500
    if not hasattr(arg, 'split'):
        warnings.warn('arg.train.split not defined. Using default value 0.9')
        arg.split = 0.9
    if not hasattr(arg, 'load_nn'):
        warnings.warn('arg.train.load_nn not defined. Using default of False')
        arg.load_nn = False
    if not hasattr(arg, 'loss_fun'):
        warnings.warn('arg.train.loss_fun not defined. Using default of ''rms''')
        arg.loss_fun = 'rms'
    if not hasattr(arg, 'skip_net'):
        warnings.warn('arg.train.skip_net not defined. Using default of False')
        arg.skip_net = False
    if not hasattr(arg, 'use_cuda'):
        arg.use_cuda = torch.cuda.is_available()
    if not hasattr(arg, 'device'):
        arg.device = torch.device("cuda:0" if arg.use_cuda else "cpu")
    return arg
