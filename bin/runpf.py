import sys
sys.path.append("..")
from uqgrid.psysdef import Psystem
from uqgrid.parse import load_matpower, load_psse
from uqgrid.pflow import runpf

#psys = load_psse(raw_filename="../data/ieee9_v33_mod1.raw")
psys = load_matpower(mat_file="../data/case118.mat")
psys.createYbusComplex()
v, Sinj = runpf(psys, verbose=True)
