import numpy as np
import torch


class sim:
    """
    Attributes
    ----------
    bvalues : np.array
        array of b-values
    SNR : list
        the SNRs to simulate
    sims : int
        number of simulations to run
    num_samples_eval : int
        number of simulations to evaluate. This can be lower than the number run. Particularly to save time when
        fitting. More simulations help with generating sufficient data for the neural network
    repeats : int
        number of repeats for simulations
    rician : bool
        add Rician noise to simulations; if false, gaussian noise is added instead
    range : tuple
    """
    def __init__(self):
        self.bvalues = np.array([0, 5, 10, 20, 30, 40, 60, 150, 300, 500, 700])
        self.SNR = [10]
        self.sims = 1000000
        self.num_samples_eval = 1000
        self.repeats = 1
        self.rician = True
        self.range = ([0.0005, 0.05, 0.01],
                      [0.003, 0.55, 0.1])


class sim_clinic:
    """
    See attributes for class sim
    """
    def __init__(self):
        self.bvalues = np.array([0, 50, 100, 200, 400, 600, 800])
        self.SNR = [10]
        self.sims = 1000000
        self.num_samples_eval = 1000
        self.repeats = 1
        self.rician = True
        self.range = ([0.0005, 0.05, 0.01],
                      [0.003, 0.55, 0.1])

    def verbose_message(self):
        return f'Clinic model with {self.SNR} SNR, {self.sims} samples noise rician {self.rician}'


class sim_phantom:
    """
    See attributes for class sim
    """
    def __init__(self):
        self.bvalues = np.array([100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500])
        self.SNR = [10]
        self.sims = 1000000
        self.num_samples_eval = 1000
        self.repeats = 1
        self.rician = True
        self.range = ([0.0005, 0.05, 0.01],
                      [0.003, 0.55, 0.1])

    def verbose_message(self):
        return f'phantom model with {self.SNR} SNR'


class lsqfit:
    """
    Attributes
    --------

    method: str
        "seg", "bayes" or "lsq"
    do_fit: bool
        skip lsq fitting
    load_lsq : bool
        load the last results for lsq fit
    fitS0 : bool
        indicates whether to fit S0 (True) or fix it to 1 in the least squares fit.
    jobs : int
        number of parallel jobs. If set to 1, no parallel computing is used
    bounds: tuple of lists
        Dt, Fp, Ds, S0
    """
    def __init__(self):
        self.method = 'bayes'
        self.do_fit = True
        self.load_lsq = False
        self.fitS0 = True
        self.jobs = 4
        self.bounds = ([0, 0, 0.005, 0.7],[0.005, 0.7, 0.3, 1.3])


class net_pars:
    """
    Used to select the parameters for the network

    ...

    Attributes
    ----------
    dropout : float: 0.0/0.1
         chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also
         states that smaller networks might desire smaller amount of dropout
    batch_norm : bool
        turns on batch normalization
    parallel : bool
        defines whether the network estimates each parameter separately (each parameter has its own network)
        or whether 1 shared network is used instead
    con : str: 'sigmoid', 'abs', 'none'
        defines the constraint function;
        'sigmoid' gives a sigmoid function giving the max/min;
        'abs' gives the absolute of the output,
        'none' does not constrain the output
    cons_min, cons_max : list
        Min/max values for Dt, Fp, Ds, S0
        Required for con = 'sigmoid'
    fitS0 : bool
        indicates whether to fit S0 (True) or fix it to 1 (for normalised signals);
        I prefer fitting S0 as it takes along the potential error is S0.
    depth : int
        number of layers
    width : int
        new option that determines network width. Setting it to 0 makes it as wide as the number of b-values
    """

    def __init__(self, nets='optim'):
        """
        Parameters
        ----------
        nets : str
            Used to select a network
            "optim" or 'optim_adsig' : optimized network settings
            "orig" : as summarized in Table 1 from the main article for the original network
        """
        if (nets == 'optim') or (nets == 'optim_adsig'):
            # the optimized network settings
            self.dropout = 0.1
            self.batch_norm = True
            self.parallel = True
            self.con = 'sigmoid'
            self.cons_min = [-0.0001, -0.05, -0.05, 0.7]  # Dt, Fp, Ds, S0
            self.cons_max = [0.005, 0.7, 0.3, 1.3]  # Dt, Fp, Ds, S0
            self.fitS0=True
            self.depth = 4
            self.width = 0

        elif nets == 'orig':
            # as summarized in Table 1 from the main article for the original network
            self.dropout = 0.0
            self.batch_norm = False
            self.parallel = False
            self.con = 'abs'
            self.cons_min = [-0.0001, -0.05, -0.05, 0.7]  # Dt, Fp, Ds, S0
            self.cons_max = [0.005, 0.7, 0.3, 1.3]  # Dt, Fp, Ds, S0
            self.fitS0 = False
            self.depth = 3
            self.width = 0

        else:
            # chose wisely :)
            self.dropout = 0.3
            self.batch_norm = True
            self.parallel = True
            self.con = 'sigmoid'
            self.cons_min = [-0.0001, -0.05, -0.05, 0.7]  # Dt, Fp, Ds, S0
            self.cons_max = [0.005, 0.7, 0.3, 1.3]  # Dt, Fp, Ds, S0
            self.fitS0 = False
            self.depth = 4
            self.width = 500


class train_pars:
    """
    most of these are options from the article and explained in the M&M.

    Attributes
    ----------
    optim : str
        Implemented optimizers
        Choices are: 'sgd'; 'sgdr'; 'adagrad' 'adam'
    lr : float
        Learning rate
    patience : int
        this is the number of epochs without improvement that the network waits until determining it found its optimum
    batch_size: int
        number of datasets taken along per iteration
    maxit : int
        max iterations per epoch
    split : float
        split of test and validation data
    load_nn : bool
        load the neural network instead of retraining
    loss_fun : str
        what is the loss used for the model.
        rms is root mean square (linear regression-like);
        L1 is L1 normalisation (less focus on outliers)
    skip_net : bool
        skip the network training and evaluation
    scheduler : bool
        as discussed in the article, LR is important. This approach allows to reduce the LR iteratively
        when there is no improvement throughout an 5 consecutive epochs
    use_cuda, device
        uses GPU if available
    select_best : bool
        ???
    ivim_combine : bool
        ???
    """
    def __init__(self, nets='optim', verbose=1):
        self.optim = 'adam'
        if nets == 'optim':
            self.lr = 0.0001
        elif nets == 'orig':
            self.lr = 0.001
        else:
            self.lr = 0.0001
        self.patience = 10
        self.batch_size = 128
        self.maxit = 500
        self.split = 0.9
        self.load_nn= False
        self.loss_fun = 'rms'
        self.skip_net = False
        self.scheduler = False

        # use GPU if available
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

        self.select_best = False
        self.ivim_combine = True

        print(f'ivim combine value: {self.ivim_combine} \n' if verbose > 0 else "", end='')
