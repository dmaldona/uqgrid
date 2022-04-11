import sys
sys.path.append("..")
from uqgrid.psysdef import Psystem
from uqgrid.parse import load_matpower, load_psse
from uqgrid.pflow import runpf
import matplotlib.pyplot as plt

psys = load_psse(raw_filename="../data/ACTIVSg200.raw")
psys.plot_network()
plt.show()
