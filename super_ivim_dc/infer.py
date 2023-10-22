from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .IVIMNET import deep
from .IVIMNET import simulations
from .source.hyperparams import hyperparams as hp

DEFAULT_BVALUE_LIST = np.array([0,15,30,45,60,75,90,105,120,135,150,175,200,400,600,800])
# SAMPLE_SIZE = np.array([10, 50, 100, 250, 500, 1000, 2000])

def infer_entry():
    import argparse

    parser = argparse.ArgumentParser(
        description="SUPER-IVIM-DC Simulation Inference", 
        epilog='Developed by the Technion Computational MRI lab: https://tcml-bme.github.io/'
    )

    parser.add_argument("--snr", "-snr", default=10, help="SNR value")

    parser.add_argument("--bval", "-b", default=-1, help="b-value as a comma separated list (without spaces)")

    parser.add_argument("--sample-size", "-ss", default=100, help="Sample size")

    parser.add_argument("--ivimnet", default=None, help="IVIMNET model path")

    parser.add_argument("--super-ivim-dc", default=None, help="SUPER-IVIM-DC model path")

    parser.add_argument("--fig", "-f", default="./output", help="Save figure path")

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")

    args = parser.parse_args()

    if args.bval == -1:
        bvalues = DEFAULT_BVALUE_LIST
    else:
        bvalues = np.array(args.bval.split(","), dtype=int)

    if args.super_ivim_dc is None and args.ivimnet is None:
        print("Please provide at least one model path")
        return

    if args.verbose:
        print(f"SNR: {args.snr}")
        print(f"b-values: {bvalues}")
        print(f"Sample size: {args.sample_size}")
        print(f"IVIMNET model path: {args.ivimnet}")
        print(f"SUPER-IVIM-DC model path: {args.super_ivim_dc}")

    infer(
        SNR=args.snr, 
        bvalues=bvalues, 
        sample_size=args.sample_size, 
        ivimnet_path=args.ivimnet, 
        super_ivim_dc_path=args.super_ivim_dc, 
        save_figure_to=args.fig
    )

def infer(
        SNR=10, 
        bvalues=DEFAULT_BVALUE_LIST, 
        sample_size=100,
        ivimnet_path=None,
        super_ivim_dc_path=None,
        save_figure_to="./output"
    ):

    arg = hp()
    arg = deep.checkarg(arg)

    IVIM_signal_noisy, D, f, Dp = simulations.sim_signal(
        SNR, 
        bvalues, 
        sample_size, 
        Dmin=arg.sim.range[0][0],
        Dmax=arg.sim.range[1][0], 
        fmin=arg.sim.range[0][1],
        fmax=arg.sim.range[1][1], 
        Dsmin=arg.sim.range[0][2],
        Dsmax=arg.sim.range[1][2], 
        rician=arg.sim.rician
    )

    labels = np.stack((D, f, Dp), axis=1).squeeze()  
    
    if ivimnet_path is not None:
        DtNET_error, FpNET_error, DpNET_error, S0NET_error = deep.infer_supervised_IVIM(
            IVIM_signal_noisy, 
            labels, 
            bvalues, 
            ivimnet_path, 
            arg
        )

        with open(Path(ivimnet_path).parent / "IVIMNET_infer.txt", "w") as ff:
            ff.write(
                f"SNR: {SNR}\n" +
                f"b-values: {bvalues}\n" +
                f"DtNET parameters error: {DtNET_error}\n" +
                f"FpNET parameters error: {FpNET_error}\n" +
                f"DpNET parameters error: {DpNET_error}\n" +
                f"S0NET parameters error: {S0NET_error}\n"
            )

    if super_ivim_dc_path is not None:
        DtSUPER_error, FpSUPER_error, DpSUPER_error, S0SUPER_error = deep.infer_supervised_IVIM(
            IVIM_signal_noisy, 
            labels, 
            bvalues, 
            super_ivim_dc_path, 
            arg
        )

        with open(Path(super_ivim_dc_path).parent / "IVIMSUPER_infer.txt", "w") as ff:
            ff.write(
                f"SNR: {SNR}\n" +
                f"b-values: {bvalues}\n" +
                f"DtSUPER parameters error: {DtSUPER_error}\n" +
                f"FpSUPER parameters error: {FpSUPER_error}\n" +
                f"DpSUPER parameters error: {DpSUPER_error}\n" +
                f"S0SUPER parameters error: {S0SUPER_error}\n"
            )

    if super_ivim_dc_path is not None and ivimnet_path is not None:
        errors_np_array = np.stack([
            DpNET_error,  
            DpSUPER_error, 
            DtNET_error, 
            DtSUPER_error,
            FpNET_error,
            FpSUPER_error,
        ], axis=1)
        bp_title = "IVIMNET VS IVIMSUPER parameters error SNR=10"
        
        deep.boxplot_ivim(errors_np_array, bp_title, save_figure_to)


def infer_signal(
        signal,
        labels,
        bvalues,
        model_path
):
    
    arg = hp()
    arg = deep.checkarg(arg)
    DtNET_error, FpNET_error, DpNET_error, S0NET_error = deep.infer_supervised_IVIM(
        signal, 
        labels, 
        bvalues, 
        model_path, 
        arg
    )

    return DtNET_error, FpNET_error, DpNET_error, S0NET_error
