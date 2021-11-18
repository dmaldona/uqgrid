# TEST CASE 001: Two-bus system

from uqgrid.psysdef import Psystem, GenGENROU, ExcESDC1A, GovIEESGO, MotCIM5
from uqgrid.dynamics import integrate_system
from uqgrid.parse import load_psse, add_dyr
from uqgrid.pflow import runpf

import unittest
import numpy as np

# Build system

class TestCase(unittest.TestCase):

    def test2bus(self):

        zfault = 1.0

        h = 1.0/120.0 # integration step in seconds
        nsteps = 2000
 
        psys = load_psse(raw_filename="data/2bus_33.raw")
        psys.createYbusComplex()
        psys.add_busfault(1, zfault, 1.0)
    
        psys.add_gen_dynamics(psys.gens[0],
            GenGENROU(0, 1.575, 1.512, 0.291, 0.39, 0.1733,
            0.0787, 3.38, 0.0, 6.1, 1.0, 0.05, 0.15))
    
        res = integrate_system(psys, comp_sens=False, tend=10.0)

        psse = np.loadtxt('data/2bus_GENROU.csv', delimiter=',')

        # retrieve PSSE values. delete negative time steps and switching events
        time_p = np.delete(psse[0, :], [0, 1, 33, 52])
        volt1_p = np.delete(psse[1, :], [0, 1, 33, 52])
        volt2_p = np.delete(psse[3, :], [0, 1, 33, 52])
        eq_p = np.delete(psse[5, :], [0, 1, 33, 52])
        speed = np.delete(psse[9, :], [0, 1, 33, 52])
        
        history = res["history"]

        #errors
        error_volt1 = np.linalg.norm(np.abs((volt1_p - history[10,:])/history[10,:]))
        error_volt2 = np.linalg.norm(np.abs((volt2_p - history[12,:])/history[12,:]))
        error_eqp = np.linalg.norm(np.abs((eq_p - history[0,:])/history[0,:]))
        error_speed = np.linalg.norm(np.abs((speed - history[4,:])/(history[4,:] + 1)))

        EPS = 0.01
        self.assertTrue(error_volt1 < EPS, 'trajectory differs')
        self.assertTrue(error_volt2 < EPS, 'trajectory differs')
        self.assertTrue(error_eqp < EPS, 'trajectory differs')
        self.assertTrue(error_speed < EPS, 'trajectory differs')
    
    def test2bus_gov(self):

        return 0
        zfault = 1.0

        h = 1.0/120.0 # integration step in seconds
        nsteps = 2000
 
        psys = load_psse(raw_filename="data/2bus_33.raw")
        psys.createYbusComplex()
        psys.add_busfault(1, zfault, 1.0)
    
        add_dyr(psys, "data/2bus_IEESGO.dyr")

        res = integrate_system(psys, comp_sens=False, tend=10.0)

        psse = np.loadtxt('data/2bus_IEESGO.csv', delimiter=',')
        
        history = res["history"]

        # retrieve PSSE values. delete negative time steps and switching events
        time_p = np.delete(psse[0, :], [0, 1, 33, 52])
        volt1_p = np.delete(psse[1, :], [0, 1, 33, 52])
        volt2_p = np.delete(psse[3, :], [0, 1, 33, 52])
        eq_p = np.delete(psse[5, :], [0, 1, 33, 52])
        speed = np.delete(psse[9, :], [0, 1, 33, 52])
        
        #errors
        error_eqp = np.linalg.norm(np.abs((eq_p - history[0,:])/history[0,:]))
        error_speed = np.linalg.norm(np.abs((speed - history[4,:])/(history[4,:] + 1)))

        EPS = 0.01
        self.assertTrue(error_eqp < EPS, 'trajectory differs')
        self.assertTrue(error_speed < EPS, 'trajectory differs')
