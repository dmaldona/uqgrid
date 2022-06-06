import uqgrid
from uqgrid.parse import load_psse, add_dyr
from uqgrid.pflow import runpf
from uqgrid.dynamics import integrate_system

import unittest
import numpy as np

class TestCase(unittest.TestCase):

    def test_forward_sensitivities(self):
        h = 1.0/120.0 # integration step in seconds
        nsteps = 1000
        eps = 1e-4

        # create system
        psys = load_psse(raw_filename="data/ieee9_v33.raw")
        psys.add_busfault(1, 1.0, 0.1)
        psys.createYbusComplex()
        add_dyr(psys, "data/ieee9bus.dyr")

        alpha = 0.5
        alpha2 = 0.5 + eps
        alpha3 = 0.5 - eps

        var1 = 4
        var2 = 5
        var3 = 10

        # integrate nominal trajectory
        psys.loads[0].set_alpha(alpha)
        psys.loads[1].set_alpha(alpha)
        psys.loads[2].set_alpha(alpha)
        res = integrate_system(psys, verbose=False, comp_sens=True, tend=10.0, dt=h)

        history = res["history"]
        tvec = res["tvec"]
        history_u = res["history_u"]
        history_v = res["history_v"]
        history_m = res["history_m"]

        # TEST FIRST-ORDER SENSITIVITIES

        for load_idx in range(3):
            psys.loads[load_idx].set_alpha(alpha2)
            res2 = integrate_system(psys, verbose = False, comp_sens=False, tend=10.0, dt=h)
            history2 = res2["history"]

            fd = (history2[var1, :] - history[var1, :]) / eps
            analytic = history_u[var1, load_idx, :]
            result = np.allclose(fd, analytic)
            self.assertTrue(result)
            
            fd = (history2[var2, :] - history[var2, :]) / eps
            analytic = history_u[var2, load_idx, :]
            result = np.allclose(fd, analytic, atol=1.e-5)
            self.assertTrue(result)
            
            fd = (history2[var3, :] - history[var3, :]) / eps
            analytic = history_u[var3, load_idx, :]
            result = np.allclose(fd, analytic)
            self.assertTrue(result)
            
            psys.loads[load_idx].set_alpha(alpha)

        # SECOND-ORDER, SELF
        for load_idx in range(3):
            psys.loads[0].set_alpha(alpha)
            psys.loads[1].set_alpha(alpha)
            psys.loads[2].set_alpha(alpha)


            # Integrate perturbed trajectories
            psys.loads[load_idx].set_alpha(alpha2)
            res2 = integrate_system(psys, verbose = False, comp_sens=False, tend=10.0, dt=h)
            history2 = res2["history"]

            # Integrate perturbed trajectories
            psys.loads[load_idx].set_alpha(alpha3)
            res3 = integrate_system(psys, verbose = False, comp_sens=False, tend=10.0, dt=h)
            history3 = res3["history"]

            fd = (history2[var1, :] - 2*history[var1, :] + history3[var1, :]) / (eps**2.0)
            analytic = history_v[var1, load_idx, :]
            result = np.allclose(fd, analytic, rtol=1.e-4, atol=1.e-5)
            self.assertTrue(result)    

            fd = (history2[var2, :] - 2*history[var2, :] + history3[var2, :]) / (eps**2.0)
            analytic = history_v[var2, load_idx, :]
            result = np.allclose(fd, analytic, rtol=1.e-4, atol=1.e-4)
            self.assertTrue(result)

            fd = (history2[var3, :] - 2*history[var3, :] + history3[var3, :]) / (eps**2.0)
            analytic = history_v[var3, load_idx, :]
            result = np.allclose(fd, analytic, rtol=1.e-4, atol=1.e-5)
            self.assertTrue(result) 

            psys.loads[0].set_alpha(alpha)
            psys.loads[1].set_alpha(alpha)
            psys.loads[2].set_alpha(alpha)

        # first mixed sensitivities

        idx_i = 0
        idx_j = 1

        # Integrate perturbed trajectories
        psys.loads[0].set_alpha(alpha2)
        psys.loads[1].set_alpha(alpha2)
        psys.loads[2].set_alpha(alpha)
        resa = integrate_system(psys, verbose = False, comp_sens=False, tend=10.0, dt=h)
        hisA = resa["history"]

        # Integrate perturbed trajectories
        psys.loads[0].set_alpha(alpha2)
        psys.loads[1].set_alpha(alpha3)
        psys.loads[2].set_alpha(alpha)
        resb = integrate_system(psys, verbose = False, comp_sens=False, tend=10.0, dt=h)
        hisB = resb["history"]

        # Integrate perturbed trajectories
        psys.loads[0].set_alpha(alpha3)
        psys.loads[1].set_alpha(alpha2)
        psys.loads[2].set_alpha(alpha)
        resc = integrate_system(psys, verbose = False, comp_sens=False, tend=10.0, dt=h)
        hisC = resc["history"]

        # Integrate perturbed trajectories
        psys.loads[0].set_alpha(alpha3)
        psys.loads[1].set_alpha(alpha3)
        psys.loads[2].set_alpha(alpha)
        resd = integrate_system(psys, verbose = False, comp_sens=False, tend=10.0, dt=h)
        hisD = resd["history"]

        fd = (hisA[var1, :] - hisB[var1, :] - hisC[var1, :] + hisD[var1, :]) / (4*eps**2.0)
        analytic = history_m[var1, 0, :]
        result = np.allclose(fd, analytic)
        self.assertTrue(result)

        fd = (hisA[var2, :] - hisB[var2, :] - hisC[var2, :] + hisD[var2, :]) / (4*eps**2.0)
        analytic = history_m[var2, 0, :]
        result = np.allclose(fd, analytic, rtol=1.e-4, atol=1.e-4)
        self.assertTrue(result)

        fd = (hisA[var3, :] - hisB[var3, :] - hisC[var3, :] + hisD[var3, :]) / (4*eps**2.0)
        analytic = history_m[var3, 0, :]
        result = np.allclose(fd, analytic)
        self.assertTrue(result)