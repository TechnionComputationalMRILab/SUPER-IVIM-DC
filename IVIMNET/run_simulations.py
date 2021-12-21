from IVIMNET.utils.checkarg import checkarg
from IVIMNET.simulations.sim_signal import sim_signal

if __name__ == "__main__":
    from hyperparams import hyperparams as hp_example

    SNR = 10
    arg = hp_example()
    arg = checkarg(arg)
    bvalues = arg.sim.bvalues
    IVIM_signal_noisy, D, f, Dp = sim_signal(SNR, bvalues, 15, Dmin=arg.sim.range[0][0],
                                             Dmax=arg.sim.range[1][0], fmin=arg.sim.range[0][1],
                                             fmax=arg.sim.range[1][1], Dsmin=arg.sim.range[0][2],
                                             Dsmax=arg.sim.range[1][2], rician=arg.sim.rician)
