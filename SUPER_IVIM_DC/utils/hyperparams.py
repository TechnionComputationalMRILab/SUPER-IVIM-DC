import torch
from SUPER_IVIM_DC.utils.parameters import net_pars, train_pars, lsqfit, sim, sim_clinic, sim_phantom


class hyperparams:
    """
    Attributes
    ----------
    fig : bool
        plot results and intermediate steps
    save_name : str
        'optim', 'orig', or "optim_adsig" for in vivo
    net_pars : ???
        ???
    train_pars : ???
        ???
    fit : function
        Perform one of three fitting methods: segmented least square, bayes or lsq
    sim : ???
        ???
    loss_coef_ivim : torch.FloatTensor
        change to 0.1 for regular test 0.4 for clinic
    key : str
    """
    def __init__(self, key='sim', verbose=1):
        self.fig = False
        self.save_name = 'optim'
        self.net_pars = net_pars(self.save_name)
        self.train_pars = train_pars(self.save_name, verbose=verbose)
        self.fit = lsqfit()

        if key == 'clinic':
            self.sim = sim_clinic()
            self.loss_coef_ivim = torch.FloatTensor([0.4])
            self.key = 'clinic'
            print(f'{self.sim.verbose_message()} \n' if verbose > 0 else "", end='')

        elif key == 'sim':
            self.sim = sim()
            self.loss_coef_ivim = torch.FloatTensor([0.1])
            self.key = 'sim'

        elif key == 'phantom':
            self.sim = sim_phantom()
            self.loss_coef_ivim = torch.FloatTensor([0.1])
            self.key = 'phantom'
            print(f'{self.sim.verbose_message()} \n' if verbose > 0 else "", end='')

        print(f'Hyperparameter class: {key} \n' if verbose > 0 else "", end='')

        self.loss_coef_Dp = torch.FloatTensor([200])
        self.loss_coef_Dt = torch.FloatTensor([20000])  # In the thesis I wrote 20000
        self.loss_coef_Fp = torch.FloatTensor([80])  # all comparison done with Fp coefficiant = 80


