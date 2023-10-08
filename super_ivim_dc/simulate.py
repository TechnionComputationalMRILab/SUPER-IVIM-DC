from .source.train_model import train_model
from .source.hyperparams import hyperparams as hp
from .IVIMNET import deep
import numpy as np
from pathlib import Path

DEFAULT_BVALUE_LIST = np.array([0,15,30,45,60,75,90,105,120,135,150,175,200,400,600,800])


def simulate_entry():
    import argparse

    parser = argparse.ArgumentParser(
        description="SUPER-IVIM-DC Simulation", 
        epilog='Developed by the Technion Computational MRI lab: https://tcml-bme.github.io/'
    )

    parser.add_argument("--snr", "-snr", default=10, help="SNR value")

    parser.add_argument("--bval", "-b", default=-1, help="b-value as a comma separated list (without spaces)")

    parser.add_argument("--mode", "-m", default="both", help="Simulation mode (can be SUPER-IVIM-DC, IVIMNET, or both)")

    parser.add_argument("--sf", "-sf", default=1, help="Sampling factor (1-6)")

    parser.add_argument("--output", "-o", default="./output", help="Working directory")

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")

    args = parser.parse_args()

    if args.bval == -1:
        bvalues = DEFAULT_BVALUE_LIST
    else:
        bvalues = np.array(args.bval.split(","), dtype=int)

    if args.verbose:
        print(f"SNR: {args.snr}")
        print(f"b-values: {bvalues}")
        print(f"Mode: {args.mode}")
        print(f"Sampling factor: {args.sf}")
        print(f"Output directory: {args.output}")    

    simulate(args.snr, bvalues, args.mode, args.sf, args.output)

def simulate(
        SNR=10, 
        bvalues=DEFAULT_BVALUE_LIST, 
        mode="both", 
        sf=1, 
        work_dir="./output"
    ):
    """
    Simulate using SUPER-IVIM-DC or IVIMNET.

    Args:
        SNR (_type_): _description_
        bvalues (_type_): _description_
        mode (str, optional): _description_. Can be either "SUPER-IVIM-DC, "IVIMNET" or "both". Defaults to "both".
        sf (int, optional): Sampling factor. Defaults to 1.
        work_dir (str, optional): Output directory of .pt and .json files. Defaults to "./output".
    """
    # set up arguments
    arg = hp('sim')
    arg = deep.checkarg(arg)
    arg.sim.SNR = [SNR]
    arg.sim.bvalues = bvalues
    arg.fig = False

    # create the work directory + directory for initial values if it doesn't exist 
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    (Path(work_dir) / "init").mkdir(parents=True, exist_ok=True)

    if mode == "both":
        matNN_superivimdc = train_model('sim', arg, 'SUPER-IVIM-DC', sf, work_dir)
        np.savetxt(f'{work_dir}/exp1_SUPER-IVIM-DC_NRMSE_snr_{SNR}_sf_{sf}.csv', np.asarray(matNN_superivimdc), delimiter=",")

        matNN_ivimnet = train_model('sim', arg, 'IVIMNET', sf, work_dir)
        np.savetxt(f'{work_dir}/exp1_IVIMNET_NRMSE_snr_{SNR}_sf_{sf}.csv', np.asarray(matNN_ivimnet), delimiter=",")

    else:
        mode = mode.upper()
        matNN = train_model('sim', arg, mode, sf, work_dir)
        np.savetxt(f'{work_dir}/exp1_{mode}_NRMSE_snr_{SNR}_sf_{sf}.csv', np.asarray(matNN), delimiter=",")
