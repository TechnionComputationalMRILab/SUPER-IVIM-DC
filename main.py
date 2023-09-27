from source.train_model import train_model
from source.hyperparams import hyperparams as hp
import IVIMNET.deep as deep
import numpy as np
from pathlib import Path

bvalues = np.array([0,15,30,45,60,75,90,105,120,135,150,175,200,400,600,800])
SNR = 10

# mode: str
# can be either
#       SUPER-IVIM-DC
#       IVIMNET
mode = "SUPER-IVIM-DC"

# sf - sampling factor
sf = 1

# directory for the .json and .pt files
work_dir = "./test"

# -----------------------------------------------------------------------------
# set up arguments
arg = hp('sim')
arg = deep.checkarg(arg)
arg.sim.SNR = [SNR]
arg.sim.bvalues = bvalues

# create the work directory + directory for initial values if it doesn't exist 
Path(work_dir).mkdir(parents=True, exist_ok=True)
(Path(work_dir) / "init").mkdir(parents=True, exist_ok=True)

matNN = train_model('sim', arg, mode, sf, work_dir)
print(matNN)
