# TEST NETWORK: 

from uqgrid.psysdef import Psystem
from uqgrid.network import createYbusComplex
from uqgrid.parse import load_matpower

import unittest
import numpy as np
import scipy.io as sio

EPS = 1e-10

print("TESTING NETWORK MODULE")

class TestCase(unittest.TestCase):

    def testninebus(self):
        psys = Psystem()

        psys.add_bus(1, bus_type = 3)
        psys.add_bus(2, bus_type = 2)
        psys.add_bus(3, bus_type = 2)
        psys.add_bus(4, bus_type = 2)
        psys.add_bus(5, bus_type = 2)
        psys.add_bus(6, bus_type = 2)
        psys.add_bus(7, bus_type = 2)
        psys.add_bus(8, bus_type = 2)
        psys.add_bus(9, bus_type = 2)

        psys.buses[0].set_vinit(1.04000, (np.pi/180.0)*0.0)
        psys.buses[1].set_vinit(1.02500, (np.pi/180.0)*9.6926)
        psys.buses[2].set_vinit(1.02500, (np.pi/180.0)*4.8812)
        psys.buses[3].set_vinit(0.99574, (np.pi/180.0)*-2.3060)
        psys.buses[4].set_vinit(0.95068, (np.pi/180.0)*-4.1382)
        psys.buses[5].set_vinit(0.96621, (np.pi/180.0)*-3.7372)
        psys.buses[6].set_vinit(0.99740, (np.pi/180.0)*3.9736)
        psys.buses[7].set_vinit(0.97915, (np.pi/180.0)*0.8364)
        psys.buses[8].set_vinit(1.00414, (np.pi/180.0)*2.1073)

        psys.add_branch(0, 3, 0.0000, 0.0576)
        psys.add_branch(1, 6, 0.0000, 0.0625)
        psys.add_branch(2, 8, 0.0000, 0.0586)
        psys.add_branch(3, 4, 0.0100, 0.0850, sh = 0.176)
        psys.add_branch(3, 5, 0.0170, 0.0920, sh = 0.158)
        psys.add_branch(4, 6, 0.0320, 0.1610, sh = 0.306)
        psys.add_branch(5, 8, 0.0390, 0.1700, sh = 0.358)
        psys.add_branch(6, 7, 0.0085, 0.0720, sh = 0.149)
        psys.add_branch(7, 8, 0.0119, 0.1008, sh = 0.209)

        ybus = createYbusComplex(psys)
        matpower_data = sio.loadmat('data/ymat9bus_matpower.mat')
        ybus_mat = matpower_data['ymat']

        for i in range(psys.nbuses):
            for j in range(psys.nbuses):
                    test_flag = np.abs(ybus[i, j] - ybus_mat[i, j]) < EPS
                    self.assertTrue(test_flag, 'Ymat entry (%d, %d) differs from test.' % (i, j))

    def testninebus_frommatpower(self):

        psys = load_matpower(mat_file="data/case14.mat")
        
        ybus = createYbusComplex(psys)
        matpower_data = sio.loadmat('data/ymat14bus_matpower.mat')
        ybus_mat = matpower_data['ymat']

        for i in range(psys.nbuses):
            for j in range(psys.nbuses):
                    test_flag = np.abs(ybus[i, j] - ybus_mat[i, j]) < EPS
                    self.assertTrue(test_flag, 'Ymat entry (%d, %d) differs from test.' % (i, j))


