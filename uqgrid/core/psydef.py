import numpy as np
from itertools import count
import cmath
from numpy.linalg import inv, det
from scipy import optimize
from scipy.sparse import csr_matrix
from uqgrid.utils.tools import csr_add_row, csr_set_row
import numba
from numba import jit
import networkx as nx

# Import base classes from the new file
from uqgrid.core.base_models import DeviceModel, DynamicGenerator, Exciter, Governor, Motor, ws

# IMPORT DEVICE IMPLEMENTATIONS
from uqgrid.models.cim5_imp import residualFinit_cim5
from uqgrid.models.load_imp import cinj_load, jac_load

class Bus(object):
    """ Generic bus class.

        Attributes:
            n (int): bus number
            type (int): bus type
    """

    _ids = count(0)

    def __init__(self, id_tag, bus_type):
        self.id = id_tag  # This id is for external reference
        self.i = next(
            self._ids
        )  # This id is sequentially created, for internal numbering
        self.type = bus_type  # 1: PQ, 2:PV, 3:slack

        # these can be set at a later time
        self.baseKV = -1
        self.dummy = False

        # registers
        self.loads = []

    def set_vinit(self, v0m, v0a):
        self.v0m = v0m
        self.v0a = v0a

    def set_alpha(self, alpha):
        # This is a stupid way to deal with the alpha issue.
        # (TODO): remove this typecode and refactor with something that makes sense.
        self.alpha = alpha


class Branch(object):
    """ Generic branch class """

    def __init__(self, i, j, r, x, sh=0.0, tap=0.0, shift=0.0):
        self.fr = i
        self.to = j
        self.r = r  # resistance (p.u)
        self.x = x  # reactance (p.u)
        self.sh = sh  # shunt reactance (p.u)
        self.tap = tap
        self.shift = shift


class Load(DeviceModel):
    """ Class for load model. """
    # (NOTE) This class is a bit inconsistent wit the rest. The load is created
    # at the static level (power flow) but, by default (as in PSSE), it is a ZIP
    # load model that plays a role in the dynamics.
    # I have decided to make this load a "DeviceModel" that will share interface
    # with the rest of the models and also participate in the "theta" vector of
    # parameters. Perhaps in the future, I will turn Load to be a vanilla object
    # and automatically generate a ZIPLoad device (similar to what we do when we 
    # add dynamics to a generator)

    def __init__(self, bus, tag, pload, qload, basemva):
        DeviceModel.__init__(self, 0, 0, 7, tag, 'ZIPLoad')
        self.bus = bus
        self.pload = pload/basemva
        self.qload = qload/basemva

        # By default this will be a pure impedance load
        self.alpha = 1.0

        # Load weight (if multiple loads, weight < 1.0)
        self.weight = 1.0

        # By default this load is type static
        self.dynamic = 0

        # We initialize v0 to -1 to indicate there has not been a power-flow
        self.v0 = -1.0

        # theta = [pload, qload, alpha, weight]
        self.initialized = False

    def set_alpha(self, alpha):
        assert alpha <= 1.0
        assert alpha >= 0.0
        self.alpha = alpha

    def initialize(self, vm, va, p, q, x, y, psys):
        # set base voltage
        self.v0 = vm
        self.initialized = True
        pass

    def initialize_theta(self, theta):
        idx = self.par_ptr

        theta[idx] = self.pload
        theta[idx + 1] = self.qload
        theta[idx + 2] = self.alpha
        theta[idx + 3] = self.weight
        theta[idx + 4] = self.v0
        
        yload = (self.pload + 1j*self.qload)/(self.v0**2.0)
        
        theta[idx + 5] = yload.real
        theta[idx + 6] = yload.imag

    def preallocate_jacobian(self, idxs, psys, power_injection):
        return []

    def preallocate_hessian(self, h_nnz, idxs, psys):
        pass

    def residual_diff(self, F, z, v, theta, idxs, ctrl_idx,
                      ctrl_var, power_injection):
        pass

    def residual_pinj(self, F, z, v, theta, idxs):

        vm = v[2*self.bus]

        pl = self.pload
        ql = self.qload
        v0 = self.v0
        alpha = self.alpha

        F[2*self.bus] += -alpha*pl*(vm/v0)**2.0 - (1 - alpha)*pl
        F[2*self.bus + 1] += alpha*ql*(vm/v0)**2.0 + (1 - alpha)*ql

    def residual_cinj(self, F, z, v, theta, idx):
        cinj_load(F, z, v, theta, idx)

    def residual_jac(self, J, z, v, theta, idxs, ctrl_idx, ctrl_var,
            power_injection):
        jac_load(z, v, theta, idxs, ctrl_idx, ctrl_var, J.data, J.indptr,
                   J.indices, power_injection)

    def residual_hess(self, H, z, v, theta, idxs, ctrl_idx, ctrl_var):
        # (TODO) Need to refactor and fit residual_hes into residual_hess. But I remember
        # i needed to call the hessian of the load before the rest of the objects to avoid
        # an issue? Make sure it is correct
        pass

    def residual_hes(self, H, z, v, theta, dev):

        HP = H[dev + 2*self.bus]
        HQ = H[dev + 2*self.bus + 1]

        Pl = self.pload
        Ql = self.qload
        v0 = self.v0
        vm = v[2*self.bus]
        alpha = self.alpha

        col = np.zeros(2)
        val = np.zeros(2)

        row = dev + 2*self.bus
        col[0] = dev + 2*self.bus
        val[0] = -alpha*2.0*Pl*(1.0/v0)**2.0
        csr_add_row(HP.data, HP.indptr, HP.indices, 1, row, col, val)

        # second row
        row = dev + 2*self.bus
        col[0] = dev + 2*self.bus
        val[0] = alpha*(2.0*Ql*(1.0/v0)**2.0)
        csr_add_row(HQ.data, HQ.indptr, HQ.indices, 1, row, col, val)

    def gradient_alpha(self, G, z, v, theta, dev, power_injection):

        if power_injection:
            vm = v[2*self.bus]
            va = v[2*self.bus + 1]
        else:
            vr = v[2*self.bus]
            vi = v[2*self.bus + 1]
            vm = np.sqrt(vr**2.0 + vi**2.0)
            va = np.arctan2(vi, vr)

        Pl = self.pload
        Ql = self.qload
        v0 = self.v0
        alpha = self.alpha

        if power_injection:
            G[2*self.bus] += -Pl*(vm/v0)**2.0 + Pl
            G[2*self.bus + 1] += Ql*(vm/v0)**2.0 - Ql
        else:
            dylda = (Pl + 1j*Ql)/(v0**2.0)
            vm2 = vr*vr + vi*vi
            vm2_tld = 0.2

            G[2*self.bus] -= vr*dylda.real - vi*dylda.imag
            G[2*self.bus + 1] -= vr*dylda.imag + vi*dylda.real

            if vm2 > vm2_tld:
                G[2*self.bus] += (Pl*vr - Ql*vi)/vm2
                G[2*self.bus + 1] += (Ql*vr + Pl*vi)/vm2
            else:
                G[2*self.bus] += (Pl*vr - Ql*vi)/vm2_tld
                G[2*self.bus + 1] += (Ql*vr + Pl*vi)/vm2_tld

    def gradient_pp_alpha(self, GX, z, v, theta, dev):

        vm = v[2*self.bus]

        Pl = self.pload
        Ql = self.qload
        v0 = self.v0
        alpha = self.alpha

        vm_idx = dev + 2*self.bus

        GX[vm_idx, vm_idx] = -2.0*Pl*(vm/v0)**2.0/vm
        GX[vm_idx + 1, vm_idx] = 2.0*Ql*(vm/v0)**2.0/vm

class Generator(object):
    def __init__(self, bus, psch, qsch, basemva, mbase):
        self.bus = bus
        self.psch = psch/basemva
        self.qsch = qsch/basemva

        if mbase > 0:
            self.mbase = mbase
        else:
            self.mbase = -1

class Shunt(object):
    def __init__(self, bus, gsh, bsh, basemva):
        self.bus = bus
        self.gsh = gsh/basemva
        self.bsh = bsh/basemva

class BusFault(object):
    def __init__(self, bus, rfault, time):

        self.bus = bus
        self.rfault = rfault
        self.time = time
        self.active = False

    def apply(self):
        self.active = True

    def remove(self):
        self.active = False

    def residual_pinj(self, F, v):
        vm = v[2*self.bus]
        F[2*self.bus] -= vm*vm*(1.0/self.rfault)

    def residual_cinj(self, F, v):
        vr = v[2*self.bus]
        vi = v[2*self.bus + 1]
        yfault = 1/self.rfault
        
        F[2*self.bus] -= yfault*vr
        F[2*self.bus + 1] -= yfault*vi

    def residual_jac(self, J, z, v, theta, dev, power_injection):

        vm = v[2*self.bus]
        col = np.zeros(2)
        val = np.zeros(2)
        yfault = 1/self.rfault

        if power_injection:
            # first row
            row = dev + 2*self.bus
            col[0] = dev + 2*self.bus
            val[0] = -2*(1.0/self.rfault)*vm
            csr_add_row(J.data, J.indptr, J.indices, 1, row, col, val)
        else:
            row = dev + 2*self.bus
            col[0] = dev + 2*self.bus
            val[0] = -yfault
            csr_add_row(J.data, J.indptr, J.indices, 1, row, col, val)
            
            row = dev + 2*self.bus + 1
            col[0] = dev + 2*self.bus + 1
            val[0] = -yfault
            csr_add_row(J.data, J.indptr, J.indices, 1, row, col, val)

    def residual_hes(self, HESS, z, v, theta, dev):

        col = np.zeros(2)
        val = np.zeros(2)

        H2 = HESS[dev + 2*self.bus]
        row = dev + 2*self.bus
        col[0] = dev + 2*self.bus
        val[0] = -2*(1.0/self.rfault)
        csr_add_row(H2.data, H2.indptr, H2.indices, 1, row, col, val)

class COI(DeviceModel):
    """
        Simple COI model. Rather than computing it a posteriori, we include it as a state variable.
        This is useful to account for it in transient stability indexes. 

        NOTE: the indexes and parameters of the generators are retrieved in the initialization.
        This means that if a new generator is added after adding the COI model, it will not be
        accounted for. 
    """
    def __init__(self):
        DeviceModel.__init__(self, 0, 1, 0, "COI1", 'COI')

    def initialize(self, vm, va, p, q, x, y, psys):
        self.w_idx = np.array(psys.genspeed_idx_set())
        self.H = np.array([gen.H for gen in psys.gendyn])

    def initialize_theta(self, theta):
        pass

    def preallocate_jacobian(self, idxs, psys, power_injection):
        coord = []
        ap = idxs[1]
        w_idxs = self.w_idx
        row = ap
        cols = w_idxs
        coord.append([row, cols])
        return coord
    
    def residual_diff(self, F, z, v, theta, idxs, ctrl_idx, ctrl_var, power_injection):
        ap = idxs[1]
        wsum = np.dot(self.H, z[self.w_idx])
        hsum = np.sum(self.H)
        F[ap] = z[ap] - (wsum/hsum)
        return None

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        pass
    
    def residual_jac(self, J, z, v, theta, idxs, ctrl_idx, ctrl_var,
            power_injection):
        
        ap = idxs[1]
        ngens = self.H.shape[0]
        hsum = np.sum(self.H)

        row = ap
        col = np.zeros(ngens + 1)
        val = np.zeros(ngens + 1)
        col[:ngens] = self.w_idx
        col[ngens] = ap
        val[:ngens] = -(1.0/hsum)*self.H
        val[ngens] = 1.0
        
        csr_add_row(J.data, J.indptr, J.indices, ngens + 1, row, col, val)

# System class
class Psystem:
    def __init__(self, basemva=100.0):

        self.basemva = basemva

        self.nbuses = 0
        self.nbranches = 0
        self.nloads = 0
        self.ngens = 0
        self.nshunts = 0
        self.nevents = 0

        self.events = []
        self.buses = []
        self.branches = []
        self.loads = []
        self.shunts = []
        self.gens = []
        self.COI = []

        self.fault_events = []

        # Dynamic devices
        self.gendyn = []
        self.exc = []
        self.gov = []
        self.mot = []

        # Devices are those elements external
        # to the admittance matrix
        self.devices = []
        self.num_devices = 0
        self.num_dof_alg = 0
        self.num_dof_dif = 0
        self.num_pars = 0

        # power flow variables
        self.nslack = 0
        self.npv = 0
        self.npq = 0

        # flags
        self.assembled = -1
        self.init_flag = False
        self.geo_flag = False
        self.power_injection = True

        # numerical integration flags.
        # perhaps this should not be here?
        self.first_jacobian_evaluation = True

        # Store direct function references to avoid method lookups
        self.rhs_funcs = []  # Will hold direct references to residual_* functions
        self.jac_funcs = []  # Will hold direct references to jac_* functions
        self.cinj_funcs = []  # Will hold direct references to cinj_* functions

    def __str__(self):
        return (
            "Power system instance composed of:\n" +
            "\tNumber of buses %d. Number of branches %d\n" %
            (self.nbuses, self.nbranches) + "\tNumber of generators: %d.\n" %
            (len(self.gens)) + "\tNumber of exciters: %d.\n" %
            (len(self.exc)) + "\tNumber of governors: %d.\n" %
            (len(self.gov)) + "\tNumerical information: \n" +
            "\t\tSize of dynamic state vector: %d\n" %
            (self.num_dof_dif) + "\t\tSize of algebraic state vector: %d\n" %
            (self.num_dof_alg))

    def add_device(self, device):
        """ This function must be called after adding each device. Should 
            be general to every dynamic device. It will update the global
            degrees of freedom and assign a pointer to the device in the
            global vector.
        """

        # register numba function reference
        #self.rhs_funcs.append(device.residual_diff_numba)

        self.devices.append(device)
        self.devices[-1].set_pointers(self.num_dof_dif, self.num_dof_alg,
                                      self.num_pars, self.num_devices)
        dif, alg, pars = self.devices[-1].getdim()
        self.num_devices += 1
        self.num_dof_alg += alg
        self.num_dof_dif += dif
        self.num_pars += pars

    def add_bus(self, n, bus_type):
        self.buses.append(Bus(n, bus_type))
        self.nbuses += 1

        if bus_type == 3:
            self.npq += 1
        elif bus_type == 2:
            self.npv += 1
        elif bus_type == 1:
            self.nslack += 1
        else:
            raise ("Incorrect bus type found.")

    def add_load(self, bus, tag, pload, qload):
        self.loads.append(Load(bus, tag, pload, qload, self.basemva))
        self.add_device(self.loads[-1])
        self.nloads += 1

    def add_shunt(self, bus, gsh, bsh):
        self.shunts.append(Shunt(bus, gsh, bsh, self.basemva))
        self.nshunts += 1

    def add_branch(self, i, j, r, x, sh=0.0, tap=0.0, shift=0.0):
        self.branches.append(Branch(i, j, r, x, sh=sh, tap=tap, shift=shift))
        self.nbranches += 1

    def add_gen(self, bus, psch, qsch, mbase=-1):
        self.gens.append(Generator(bus, psch, qsch, self.basemva, mbase=mbase))
        self.ngens += 1

    def add_busfault(self, bus, rfault, time):
        self.fault_events.append(BusFault(bus, rfault, time))

    def add_gen_dynamics(self, gen, gendynamics):
        assert isinstance(gen, Generator)
        assert isinstance(gendynamics, DynamicGenerator)
        self.gendyn.append(gendynamics)
        self.add_device(self.gendyn[-1])
        gendynamics.set_bus(gen.bus)
        if gen.mbase > 0:
            ratio = gen.mbase/self.basemva
            gendynamics.set_ratio(ratio)

    def add_load_dynamics(self, load, loaddynamics):
        assert isinstance(load, Load)
        assert isinstance(loaddynamics, Motor)
        self.mot.append(loaddynamics)
        self.add_device(self.mot[-1])
        load.dynamic = 1
        loaddynamics.set_weight(load.weight)
        loaddynamics.set_bus(load.bus)

    def set_load_weights(self, bus, new_weight):
        """ Re-sets load weights at specified node. """

        # implemented for case where we have two loads only.
        assert len(self.buses[bus].loads) == 2
        assert self.init_flag == False

        ptot = self.buses[bus].loads[0].pload + self.buses[bus].loads[1].pload
        qtot = self.buses[bus].loads[0].qload + self.buses[bus].loads[1].qload

        self.buses[bus].loads[0].weight = new_weight
        self.buses[bus].loads[0].pload = ptot*new_weight
        self.buses[bus].loads[0].qload = qtot*new_weight

        self.buses[bus].loads[1].weight = 1 - new_weight
        self.buses[bus].loads[1].pload = ptot*(1 - new_weight)
        self.buses[bus].loads[1].qload = qtot*(1 - new_weight)

        # do this BEFORE loading dynamic file
        assert self.buses[bus].loads[0].dynamic == 0
        assert self.buses[bus].loads[1].dynamic == 0

    def assemble(self):
        """ creates essential data structures """
        graph = [[] for i in range(self.nbuses)]

        for branch in self.branches:
            graph[branch.fr].append(branch.to)
            graph[branch.to].append(branch.fr)

        # create networkx graph
        edgelist = [(branch.fr, branch.to) for branch in self.branches]
        G = nx.Graph()
        G.add_edges_from(edgelist)
        self.graph = G

        # lazy way to get unique elements (due to parallel lines).
        # Might be OK because in general, connectivity in psys is sparse.
        for i in range(len(graph)):
            graph[i] = list(set(graph[i]))

        self.graph_list = graph
        """ These additional data structures are a memory waste
            but should work much faster and I can work with them
            in Numba.
        """

        # find node with the maximum connections
        max_con = max(map(len, graph))
        self.max_con = max_con

        graph_mat = -1*np.ones((self.nbuses, 1 + max_con), dtype=np.int64)
        for i in range(len(graph)):
            graph_mat[i, 0] = len(graph[i])
            for j in range(graph_mat[i, 0]):
                graph_mat[i, 1 + j] = graph[i][j]
        self.graph_mat = graph_mat
        """ register loads and generators connected to buses """
        for load in self.loads:
            bus = load.bus
            self.buses[i].loads.append(load)
        """ for each bus with multiple loads, calculate weights """
        for bus in self.buses:
            if len(bus.loads) <= 1:
                pass
            tot_load = 0.0
            for load in bus.loads:
                tot_load += load.pload
            for load in bus.loads:
                load.weight = load.pload/tot_load

        self.assembled = 1

    def createYbusComplex(self):
        from uqgrid.models.network import createYbusComplex
        self.ybus_spa = createYbusComplex(self)


        """ Bizarre wasteful numpy matrix"""
        ybus_mat = np.zeros(
             (self.nbuses, self.max_con + 1), dtype=np.complex128)

        for i in range(self.nbuses):
            ybus_mat[i, 0] = self.ybus_spa[i, i]
            for j in range(self.graph_mat[i, 0]):
                to_bus = self.graph_mat[i, 1 + j]
                ybus_mat[i, j + 1] = self.ybus_spa[i, to_bus]

        self.ybus_mat = ybus_mat

    def ybus_complex2real(self):
        from uqgrid.models.network import realify_ybus
        self.rybus = realify_ybus(self)

    # For exciters and governors, these are always associated to a generator.
    # Associated generator must be provided.
    # Note: since every exciter/governor must be preceded by a generator in the
    # dynamic elements list, we can initialize those independently using the results
    # of the initialized generator.

    def add_exc(self, gen, exc):
        assert isinstance(gen, DynamicGenerator)
        self.exc.append(exc)
        gen.attach_exciter(exc)
        self.add_device(self.exc[-1])
        exc.set_bus(gen.bus)

    def add_gov(self, gen, gov):
        assert isinstance(gen, DynamicGenerator)
        self.gov.append(gov)
        gen.attach_governor(gov)
        self.add_device(self.gov[-1])
        gov.set_bus(gen.bus)

    def add_mot(self, load, mot):
        assert isinstance(load, Load)
        self.mot.append(mot)
        self.add_device(self.mot[-1])
        mot.set_bus(load.bus)

    def add_COI(self):
        coi_obj = COI()
        self.COI.append(coi_obj)
        self.add_device(self.COI[-1])

    def initialize(self):
        # survey generators and assign indexes
        for gen in self.gendyn:
            if gen.exciter:
                exc = gen.exciter
                gen.efd_idx = self.device_to_global(exc, exc.efd_idx)
                gen.set_efd_idx(self.device_to_global(exc, exc.efd_idx))

            if gen.governor:
                gov = gen.governor
                gen.pm_idx = self.device_to_global(gov,
                                                   gov.state_list.index('p_m'))
                gen.set_pm_idx(
                    self.device_to_global(gov, gov.state_list.index('p_m')))
                gov.w_idx = self.device_to_global(gen,
                                                  gen.state_list.index('w'))

        self.init_flag = True
        self.first_jacobian_evaluation = True

    # Tools

    def device_to_global(self, dev, dev_idx):
        assert isinstance(dev, DeviceModel)
        if dev_idx < dev.dif_dim:
            return dev.dif_ptr + dev_idx
        else:
            return self.num_dof_dif + dev.alg_ptr + (dev_idx - dev.dif_dim)

    def survey_dynamic_models(self):
        for model in self.devices:
            print("Model %d. Bus: %d. Type: %s. diff_ptr: %d. alg_ptr: %d" %
                  (model.ndev, model.bus, model.model_type, model.dif_ptr,
                   model.alg_ptr))

    def add_ext2int(self, dictionary):
        assert len(dictionary) == self.nbuses
        self.ext2int = dictionary

    def get_loadvec(self):
        """ returns vector of size 2*nbus with total load consumption """
        pload = np.zeros(2*self.nbuses)
        for load in self.loads:
            pload[2*load.bus] -= load.pload
            pload[2*load.bus + 1] += load.qload

        return pload

    def set_load_parameters(self, par_vec):
        assert par_vec.shape[0] == self.nloads
        for i in range(self.nloads):
            self.loads[i].set_alpha(par_vec[i])

    def network_distance(self, bus_fr, bus_to, distance="shortest_path"):

        if distance == "shortest_path":
            # do I really need to do this to import it?
            from .network import distance_graph
            return distance_graph(self.graph, bus_fr, bus_to)
        elif distance == "resistance":
            from .network import distance_resistance
            return distance_resistance(self.graph, bus_fr, bus_to)
        else:
            raise ("Network distance not implemented.")
    # IO

    def busmag_idx_set(self):
        """ Returns list of bus magnitude indexes """
        ptr = self.num_dof_alg + self.num_dof_dif
        return [2*i + ptr for i in range(self.nbuses)]

    def busang_idx_set(self):
        """ Returns list of bus angle indexes """
        ptr = self.num_dof_alg + self.num_dof_dif
        return [2*i + 1 + ptr for i in range(self.nbuses)]

    def genspeed_idx_set(self):
        """ Returns list of generator speed deviations """
        return [gen.dif_ptr + 4 for gen in self.gendyn]

    def coi_idx(self):
        if not self.COI:
            return None
        else:
            return self.num_dof_dif + self.COI[0].alg_ptr

    def idx_to_description(self, idx_num):

        dif_size = self.num_dof_dif
        alg_size = self.num_dof_alg
        pow_size = 2*self.nbuses

        assert idx_num < alg_size + dif_size + pow_size

        if idx_num < dif_size:
            for model in self.devices:
                dev_ptr = model.dif_ptr
                dev_ptr_end = model.dif_ptr + model.dif_dim
                if dev_ptr <= idx_num <= dev_ptr_end:
                    print(
                        "Index %g pertains to a %s in bus %d. Dynamic state number: %d."
                        % (idx_num, model.model_type, model.bus,
                           idx_num - dev_ptr))
        elif idx_num > alg_size + dif_size:
            print("Voltage variable.")
        else:
            for model in self.devices:
                dev_ptr = model.alg_ptr
                dev_ptr_end = model.alg_ptr + model.alg_dim
                if dev_ptr <= idx_num <= dev_ptr_end:
                    print(
                        "Index %g pertains to a %s in bus %d. Algebraic state number: %d."
                        % (idx_num, model.model_type, model.bus,
                           idx_num - dev_ptr))

    # network plot

    def plot_network(self):
        if self.geo_flag == True:
            pos = {i:self.substations[self.bus2sub[i]] for i in range(self.nbuses)}
            nx.draw(self.graph, pos=pos)
        else:
            nx.draw(self.graph)

    def add_geo(self, substations, bus2sub):
        self.substations = substations
        self.bus2sub = bus2sub
        self.geo_flag = True


#####################################################
# IMPLEMENTATION OF MODEL EQUATIONS                ##
# THIS MIGHT BE MOVED TO DIFFERENT PYTHON MODULE   ##
#####################################################

class ExcESDC1A(Exciter):
    def __init__(self, id_tag, Ka, Ta, Kf, Tf, Ke, Te, Tr, Ae, Be):

        self.Ka = Ka
        self.Ta = Ta
        self.Kf = Kf
        self.Tf = Tf
        self.Ke = Ke
        self.Te = Te
        self.Tr = Tr
        self.Ae = Ae
        self.Be = Be

        # control variables
        self.vref = None
        self.efd_idx = 2

        parameter_list = ['Ka', 'Ta', 'Kf', 'Tf', 'Ke', 'Te', 'Tr', 'Ae', 'Be']
        state_list = ['vr1', 'vr2', 'e_fd']

        Exciter.__init__(self, id_tag, 3, 3, 0, len(parameter_list), state_list)

    def residualFinit(self, x, v, theta, p0, q0):

        F = np.zeros(self.initdim)

        # parameters
        Ka = self.Ka
        Ta = self.Ta
        Kf = self.Kf
        Tf = self.Tf
        Ke = self.Ke
        Te = self.Te
        Tr = self.Tr
        Ae = self.Ae
        Be = self.Be
        e_fd = self.e_fd0

        vr1 = x[0]
        vr2 = x[1]
        vref = x[2]

        F[0] = (Ka*(vref - v - vr2 - (Kf/Tf)*e_fd) - vr1)/Ta
        F[1] = -((Kf/Tf)*e_fd + vr2)/Tf
        F[2] = -(e_fd*(Ke + Ae*np.exp(Be*v)) - vr1)/Te

        return F

    def initialize(self, vm, va, p, q, x, y, psys):

        x0 = np.ones(self.initdim)
        sol = optimize.root(
            self.residualFinit,
            x0,
            args=(vm, va, p, q),
            method='krylov',
            options={
                'xtol': 1e-8,
                'disp': False
            })

        self.initialized = True
        x[self.dif_ptr:self.dif_ptr + 2] = sol.x[0:2]
        x[self.dif_ptr + 2] = self.e_fd0
        self.vref = sol.x[2]
        return None

    def initialize_theta(self, theta):

        idx = self.par_ptr

        theta[idx] = self.Ka
        theta[idx + 1] = self.Ta
        theta[idx + 2] = self.Kf
        theta[idx + 3] = self.Tf
        theta[idx + 4] = self.Ke
        theta[idx + 5] = self.Te
        theta[idx + 6] = self.Tr
        theta[idx + 7] = self.Ae
        theta[idx + 8] = self.Be

    def residual_diff(self, F, z, v, theta, idxs, ctrl_idx, ctrl_var):

        dp = idxs[0]
        ap = idxs[1]

        # parameters
        Ka = self.Ka
        Ta = self.Ta
        Kf = self.Kf
        Tf = self.Tf
        Ke = self.Ke
        Te = self.Te
        Tr = self.Tr
        Ae = self.Ae
        Be = self.Be

        # states
        vr1 = z[dp]
        vr2 = z[dp + 1]
        e_fd = z[dp + 2]

        # setpoint (to be implemented in external uref vector)
        vref = self.vref

        vm = v[2*self.bus]
        va = v[2*self.bus + 1]

        F[dp] = (Ka*(vref - vm - vr2 - (Kf/Tf)*e_fd) - vr1)/Ta
        F[dp + 1] = -((Kf/Tf)*e_fd + vr2)/Tf
        F[dp + 2] = -(e_fd*(Ke + Ae*np.exp(Be*vm)) - vr1)/Te

        return None

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def preallocate_jacobian(self, idxs, psys):

        coord = []

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]

        # these are INDEXES
        vr1 = dp
        vr2 = dp + 1
        e_fd = dp + 2

        vm = dev + 2*self.bus
        va = dev + 2*self.bus + 1

        # first row
        row = dp
        cols = [vr1, vr2, e_fd, vm]
        coord.append([row, cols])

        # second row
        row = dp + 1
        cols = [vr2, e_fd]
        coord.append([row, cols])

        # third row
        row = dp + 2
        cols = [vr1, e_fd, vm]
        coord.append([row, cols])

        return coord

    def residual_jac(self, J, z, v, theta, idxs, ctrl_idx, ctrl_var):
        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]

        # parameters
        Ka = self.Ka
        Ta = self.Ta
        Kf = self.Kf
        Tf = self.Tf
        Ke = self.Ke
        Te = self.Te
        Tr = self.Tr
        Ae = self.Ae
        Be = self.Be

        # states
        vr1 = z[dp]
        vr2 = z[dp + 1]
        e_fd = z[dp + 2]

        # setpoint (to be implemented in external uref vector)
        vref = self.vref

        vm = v[2*self.bus]
        va = v[2*self.bus + 1]

        # indexes
        vr1_idx = dp
        vr2_idx = dp + 1
        e_fd_idx = dp + 2
        vm_idx = dev + 2*self.bus
        va_idx = dev + 2*self.bus + 1

        col = np.zeros(10)
        val = np.zeros(10)

        # first row
        row = dp
        col[0] = vr1_idx
        val[0] = -1/Ta
        col[1] = vr2_idx
        val[1] = -Ka/Ta
        col[2] = e_fd_idx
        val[2] = -Ka*Kf/(Ta*Tf)
        col[3] = vm_idx
        val[3] = -Ka/Ta
        csr_set_row(J.data, J.indptr, J.indices, 4, row, col, val)

        # second row
        row = dp + 1
        col[0] = vr2_idx
        val[0] = -1/Tf
        col[1] = e_fd_idx
        val[1] = -Kf/Tf**2
        csr_set_row(J.data, J.indptr, J.indices, 2, row, col, val)

        # third row
        row = dp + 2
        col[0] = vr1_idx
        val[0] = 1/Te
        col[1] = e_fd_idx
        val[1] = -(Ke + Ae*np.exp(Be*vm))/Te
        col[2] = vm_idx
        val[2] = -e_fd*(Ae*Be*np.exp(Be*vm))/Te
        csr_set_row(J.data, J.indptr, J.indices, 3, row, col, val)

    def preallocate_hessian(self, h_nnz, idxs, psys):

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]

        # these are INDEXES
        vr1 = dp
        vr2 = dp + 1
        e_fd = dp + 2

        vm = dev + 2*self.bus
        va = dev + 2*self.bus + 1

        # F0
        h_nnz[dp + 2]['rows'].append(e_fd)
        h_nnz[dp + 2]['cols'].append([vm])

        h_nnz[dp + 2]['rows'].append(vm)
        h_nnz[dp + 2]['cols'].append([e_fd, vm])

    def residual_hess(self, HESS, z, v, theta, idxs, ctrl_idx, ctrl_var):

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]
        pp = idxs[3]
        bus = idxs[4]

        H0 = HESS[dp]
        H1 = HESS[dp + 1]
        H2 = HESS[dp + 2]

        # parameters
        Ka = self.Ka
        Ta = self.Ta
        Kf = self.Kf
        Tf = self.Tf
        Ke = self.Ke
        Te = self.Te
        Tr = self.Tr
        Ae = self.Ae
        Be = self.Be

        # states
        vr1 = z[dp]
        vr2 = z[dp + 1]
        e_fd = z[dp + 2]

        # setpoint (to be implemented in external uref vector)
        vref = self.vref

        vm = v[2*self.bus]
        va = v[2*self.bus + 1]

        # indexes
        vr1_idx = dp
        vr2_idx = dp + 1
        e_fd_idx = dp + 2
        vm_idx = dev + 2*self.bus
        va_idx = dev + 2*self.bus + 1

        # column and value vectors
        col = np.zeros(10)
        val = np.zeros(10)

        ### HESSIAN OF F0 ###

        row = e_fd_idx
        col[0] = vm_idx
        val[0] = -Ae*Be*np.exp(Be*vm)/Te
        csr_set_row(H2.data, H2.indptr, H2.indices, 1, row, col, val)

        row = vm_idx
        col[0] = e_fd_idx
        val[0] = -Ae*Be*np.exp(Be*vm)/Te
        col[1] = vm_idx
        val[1] = -Ae*Be**2*e_fd*np.exp(Be*vm)/Te
        csr_set_row(H2.data, H2.indptr, H2.indices, 2, row, col, val)


class GovIEESGO(Governor):
    def __init__(self, id_tag, T1, T2, T3, T4, T5, T6, K1, K2, K3):

        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.T4 = T4
        self.T5 = T5
        self.T6 = T6
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3

        # control variable
        self.pref = None

        parameter_list = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'K1', 'K2', 'K3']
        state_list = ['PF0', 'PLL', 'TP1', 'TP2', 'TP3', 'p_m']

        Governor.__init__(self, id_tag, 6, 5, 1, 9, state_list)

    def residualFinit(self, x, v, theta, p0, q0, w):

        F = np.zeros(self.initdim)

        PF0 = x[0]
        PLL = x[1]
        TP1 = x[2]
        TP2 = x[3]
        TP3 = x[4]
        pref = x[5]

        T1 = self.T1
        T2 = self.T2
        T3 = self.T3
        T4 = self.T4
        T5 = self.T5
        T6 = self.T6
        K1 = self.K1
        K2 = self.K2
        K3 = self.K3

        F[0] = (1.0/T1)*(K1*w - PF0)
        F[1] = (1/T3)*((1.0 - (T2/T3))*PF0 - PLL)

        SatP = pref - (T2/T3)*PF0 - PLL

        F[2] = (1/T4)*(SatP - TP1)
        F[3] = (1/T5)*(K2*TP1 - TP2)
        F[4] = (1/T6)*(K3*TP2 - TP3)
        F[5] = TP1*(1 - K2) + TP2*(1 - K3) + TP3 - self.p_m0

        return F

    def initialize(self, vm, va, p, q, x, y, psys):

        w = x[self.w_idx]

        x0 = np.ones(self.initdim)
        sol = optimize.root(
            self.residualFinit,
            x0,
            args=(vm, va, p, q, w),
            method='krylov',
            options={
                'xtol': 1e-8,
                'disp': False
            })

        self.initialized = True
        x[self.dif_ptr:self.dif_ptr + 5] = sol.x[0:5]
        y[self.alg_ptr:self.alg_ptr + 1] = self.p_m0
        self.pref = sol.x[5]

        return None

    def initialize_theta(self, theta):

        idx = self.par_ptr

        theta[idx] = self.T1
        theta[idx + 1] = self.T2
        theta[idx + 2] = self.T3
        theta[idx + 3] = self.T4
        theta[idx + 4] = self.T5
        theta[idx + 5] = self.T6
        theta[idx + 6] = self.K1
        theta[idx + 7] = self.K2
        theta[idx + 8] = self.K3

    def residual_diff(self, F, z, v, theta, idxs, ctrl_idx, ctrl_var):

        dp = idxs[0]
        ap = idxs[1]

        # parameters
        T1 = self.T1
        T2 = self.T2
        T3 = self.T3
        T4 = self.T4
        T5 = self.T5
        T6 = self.T6
        K1 = self.K1
        K2 = self.K2
        K3 = self.K3

        # states
        PF0 = z[dp]
        PLL = z[dp + 1]
        TP1 = z[dp + 2]
        TP2 = z[dp + 3]
        TP3 = z[dp + 4]
        p_m = z[ap]

        w = z[self.w_idx]
        pref = self.pref

        # resfun
        F[dp] = (1.0/T1)*(K1*w - PF0)
        F[dp + 1] = (1.0/T3)*((1.0 - (T2/T3))*PF0 - PLL)

        SatP = pref - (T2/T3)*PF0 - PLL

        F[dp + 2] = (1.0/T4)*(SatP - TP1)
        F[dp + 3] = (1.0/T5)*(K2*TP1 - TP2)
        F[dp + 4] = (1.0/T6)*(K3*TP2 - TP3)

        F[ap] = TP1*(1 - K2) + TP2*(1 - K3) + TP3 - p_m

        return None

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def preallocate_jacobian(self, idxs, psys, power_injection):

        coord = []

        dp = idxs[0]
        ap = idxs[1]

        # these are INDEXES
        PF0 = dp
        PLL = dp + 1
        TP1 = dp + 2
        TP2 = dp + 3
        TP3 = dp + 4
        p_m = ap

        w = self.w_idx

        # first row
        row = dp
        cols = [w, PF0]
        coord.append([row, cols])

        # second row
        row = dp + 1
        cols = [PF0, PLL]
        coord.append([row, cols])

        # third row
        row = dp + 2
        cols = [PF0, PLL, TP1]
        coord.append([row, cols])

        # fourth row
        row = dp + 3
        cols = [TP1, TP2]
        coord.append([row, cols])

        # fifth row
        row = dp + 4
        cols = [TP2, TP3]
        coord.append([row, cols])

        row = ap
        cols = [TP1, TP2, TP3, p_m]
        coord.append([row, cols])

        return coord

    def preallocate_hessian(self, h_nnz, idxs, psys):
        # Function is linear
        pass

    def residual_hess(self, HESS, z, v, theta, idxs, ctrl_idx, ctrl_var):
        pass

    def residual_jac(self, J, z, v, theta, idxs, ctrl_idx, ctrl_var,
            power_injection):
        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]

        # parameters
        T1 = self.T1
        T2 = self.T2
        T3 = self.T3
        T4 = self.T4
        T5 = self.T5
        T6 = self.T6
        K1 = self.K1
        K2 = self.K2
        K3 = self.K3

        # states
        PF0 = z[dp]
        PLL = z[dp + 1]
        TP1 = z[dp + 2]
        TP2 = z[dp + 3]
        TP3 = z[dp + 4]
        p_m = z[ap]

        w = z[self.w_idx]
        pref = self.pref

        # indeces
        PF0_idx = dp
        PLL_idx = dp + 1
        TP1_idx = dp + 2
        TP2_idx = dp + 3
        TP3_idx = dp + 4
        p_m_idx = ap

        w_idx = self.w_idx

        # column and value vectors
        col = np.zeros(10)
        val = np.zeros(10)

        # first row
        row = dp
        col[0] = w_idx
        val[0] = 1.0*K1/T1
        col[1] = PF0_idx
        val[1] = -1.0/T1
        csr_set_row(J.data, J.indptr, J.indices, 2, row, col, val)

        # second  row
        row = dp + 1
        col[0] = PF0_idx
        val[0] = 1.0*(-T2/T3 + 1.0)/T3
        col[1] = PLL_idx
        val[1] = -1.0/T3
        csr_set_row(J.data, J.indptr, J.indices, 2, row, col, val)

        # third  row
        row = dp + 2
        col[0] = PF0_idx
        val[0] = -1.0*T2/(T3*T4)
        col[1] = PLL_idx
        val[1] = -1.0/T4
        col[2] = TP1_idx
        val[2] = -1.0/T4
        csr_set_row(J.data, J.indptr, J.indices, 3, row, col, val)

        row = dp + 3
        col[0] = TP1_idx
        val[0] = 1.0*K2/T5
        col[1] = TP2_idx
        val[1] = -1.0/T5
        csr_set_row(J.data, J.indptr, J.indices, 2, row, col, val)

        row = dp + 4
        col[0] = TP2_idx
        val[0] = 1.0*K3/T6
        col[1] = TP3_idx
        val[1] = -1.0/T6
        csr_set_row(J.data, J.indptr, J.indices, 2, row, col, val)

        row = ap
        col[0] = TP1_idx
        val[0] = -K2 + 1
        col[1] = TP2_idx
        val[1] = -K3 + 1
        col[2] = TP3_idx
        val[2] = 1.0
        col[3] = p_m_idx
        val[3] = -1.0
        csr_set_row(J.data, J.indptr, J.indices, 4, row, col, val)


class MotCIM5(Motor):
    def __init__(self, id_tag, ra, xa, xm, r1, x1, H, D):

        self.ra = ra
        self.xa = xa
        self.xm = xm
        self.r1 = r1
        self.x1 = x1
        self.H = H
        self.D = D

        self.tp = (x1 + xm)/(r1*ws)
        self.x0 = (xa + xm)
        self.x_p = xa + (x1*xm)/(x1 + xm)

        state_list = ['e_dqp', 'e_qp', 's', 'i_ds', 'i_qs', 't_m', 'ysh']
        param_list = [
            'ra', 'xa', 'xm', 'r1', 'x1', 'H', 'D', 'tp', 'x0', 'x_p'
        ]

        Motor.__init__(self, id_tag, 7, 5, 2, len(param_list), state_list)

    def initialize_theta(self, theta):

        idx = self.par_ptr

        theta[idx] = self.ra
        theta[idx + 1] = self.xa
        theta[idx + 2] = self.xm
        theta[idx + 3] = self.r1
        theta[idx + 4] = self.x1
        theta[idx + 5] = self.H
        theta[idx + 6] = self.D
        theta[idx + 7] = self.tp
        theta[idx + 8] = self.x0
        theta[idx + 9] = self.x_p

    def init_sens(self, x, v, va, p0, q0, weight):

        e_dp = x[0]
        e_qp = x[1]
        s = x[2]
        t_m = x[3]
        ysh = x[4]
        i_ds = x[5]
        i_qs = x[6]

        tp = self.tp
        x_0 = self.x0
        x_p = self.x_p
        Hm = self.H
        ra = self.ra

        v_ds = -v*np.sin(va)
        v_qs = v*np.cos(va)

        J = np.array(
            [[-1.0/tp, s*ws, e_qp*ws, 0, 0, 0, -1.0*(x_0 - x_p)/tp],
             [-s*ws, -1.0/tp, -e_dp*ws, 0, 0, -1.0*(-x_0 + x_p)/tp, 0], [
                 -0.5*i_ds/Hm, -0.5*i_qs/Hm, 0, 0.5/Hm, 0, -0.5*e_dp/Hm,
                 -0.5*e_qp/Hm
             ], [1, 0, 0, 0, 0, ra, -x_p], [0, 1, 0, 0, 0, x_p, ra],
             [0, 0, 0, 0, 0, -v*np.sin(va),
              v*np.cos(va)], [0, 0, 0, 0, v**2, v*np.cos(va), v*np.sin(va)]])

        JA = np.array([0, 0, 0, 0, 0, -p0, -q0])

        # Hessian
        nstate = 7
        H = nstate*[None]
        H[0] = np.zeros((nstate, nstate))
        H[1] = np.zeros((nstate, nstate))
        H[2] = np.zeros((nstate, nstate))

        H[0][1, 2] = ws
        H[0][2, 1] = ws
        H[1][0, 2] = -ws
        H[1][2, 0] = -ws
        H[2][0, 5] = -0.5/Hm
        H[2][1, 6] = -0.5/Hm
        H[2][5, 0] = -0.5/Hm
        H[2][6, 1] = -0.5/Hm

        return J, JA, H

    def residualFinit(self, x, v, va, p0, q0):

        F = np.zeros(self.initdim)

        e_dp = x[0]
        e_qp = x[1]
        s = x[2]
        t_m = x[3]
        ysh = x[4]
        i_ds = x[5]
        i_qs = x[6]

        tp = self.tp
        x0 = self.x0
        x_p = self.x_p
        Hm = self.H
        ra = self.ra

        v_ds = -v*np.sin(va)
        v_qs = v*np.cos(va)

        F[0] = (-1.0/tp)*(e_dp + (x0 - x_p)*i_qs) + s*ws*e_qp
        F[1] = (-1.0/tp)*(e_qp - (x0 - x_p)*i_ds) - s*ws*e_dp

        F[2] = (1.0/(2.0*Hm))*(t_m - e_dp*i_ds - e_qp*i_qs)

        F[3] = ra*i_ds - x_p*i_qs + e_dp - v_ds
        F[4] = ra*i_qs + x_p*i_ds + e_qp - v_qs

        F[5] = v_ds*i_ds + v_qs*i_qs + p0
        F[6] = (v_qs*i_ds - v_ds*i_qs + ysh*v*v) + q0

        return F

    def initialize(self, vm, va, p, q, x, y, psys):

        x0 = np.ones(self.initdim)
        w = self.weight
        x0[2] = 0.01
        sol = optimize.root(
            self.residualFinit,
            x0,
            args=(vm, va, w*p, w*q),
            method='krylov',
            options={
                'xtol': 1e-10,
                'disp': False
            })
        assert sol.success == True
        self.initialized = True
        x[self.dif_ptr:self.dif_ptr + 5] = sol.x[0:5]
        y[self.alg_ptr:self.alg_ptr + 2] = sol.x[5:7]

        return None

    def initialize_sens(self, vm, va, p, q, z, u, v, psys, diff_size):

        mot_x = np.zeros(self.initdim)
        mot_x[0:5] = z[self.dif_ptr:self.dif_ptr + 5]
        mot_x[5:7] = z[self.alg_ptr + diff_size:self.alg_ptr + diff_size + 2]
        w = self.weight

        # compute initial sensitivity vectors
        J, JA, H = self.init_sens(mot_x, vm, va, p, q, w)
        u_mot = -np.dot(np.linalg.inv(J), JA)

        # set sensitivities
        u[self.dif_ptr:self.dif_ptr + 5] = u_mot[0:5]
        u[self.alg_ptr + diff_size:self.alg_ptr + diff_size + 2] = u_mot[5:7]

        # second order
        b = np.zeros(7)
        for i in range(len(H)):
            if H[i] is not None:
                b[i] += u_mot.dot(H[i].dot(u_mot))

        v_mot = -np.dot(np.linalg.inv(J), b)
        v[self.dif_ptr:self.dif_ptr + 5] = v_mot[0:5]
        v[self.alg_ptr + diff_size:self.alg_ptr + diff_size + 2] = v_mot[5:7]

        return None

    def residual_diff(self, F, z, v, theta, idxs, ctrl_idx, ctrl_var):

        dp = idxs[0]
        ap = idxs[1]

        # paramters
        tp = self.tp
        x_0 = self.x0
        x_p = self.x_p
        Hm = self.H
        ra = self.ra

        # states
        e_dpm = z[dp]
        e_qpm = z[dp + 1]
        s = z[dp + 2]
        t_m = z[dp + 3]
        ysh = z[dp + 4]
        i_dm = z[ap]
        i_qm = z[ap + 1]

        vm = v[2*self.bus]
        va = v[2*self.bus + 1]

        v_dm = -vm*np.sin(va)
        v_qm = vm*np.cos(va)

        F[dp] = (-1.0/tp)*(e_dpm + (x_0 - x_p)*i_qm) + s*ws*e_qpm
        F[dp + 1] = (-1.0/tp)*(e_qpm - (x_0 - x_p)*i_dm) - s*ws*e_dpm
        F[dp + 2] = (1.0/(2.0*Hm))*(t_m - e_dpm*i_dm - e_qpm*i_qm)
        F[dp + 3] = 0.0
        F[dp + 4] = 0.0

        F[ap] = ra*i_dm - x_p*i_qm + e_dpm - v_dm
        F[ap + 1] = ra*i_qm + x_p*i_dm + e_qpm - v_qm

        return None

    def residual_pinj(self, F, z, v, theta, idxs, alpha=0.0):

        dp = idxs[0]
        ap = idxs[1]

        ysh = z[dp + 4]
        i_dm = z[ap]
        i_qm = z[ap + 1]

        vm = v[2*self.bus]
        va = v[2*self.bus + 1]

        v_dm = -vm*np.sin(va)
        v_qm = vm*np.cos(va)

        F[2*self.bus] -= (v_dm*i_dm + v_qm*i_qm)
        F[2*self.bus + 1] -= (v_qm*i_dm - v_dm*i_qm)
        F[2*self.bus + 1] -= ysh*(vm*vm)

        return None

    def preallocate_jacobian(self, idxs, psys):

        coord = []

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]

        # these are INDEXES
        e_dpm = dp
        e_qpm = dp + 1
        s = dp + 2
        t_m = dp + 3
        ysh = dp + 4
        i_dm = ap
        i_qm = ap + 1

        vm = dev + 2*self.bus
        va = dev + 2*self.bus + 1

        # first row
        row = dp
        cols = [e_dpm, e_qpm, s, i_qm]
        coord.append([row, cols])

        # second row
        row = dp + 1
        cols = [e_dpm, e_qpm, s, i_dm]
        coord.append([row, cols])

        # third row
        row = dp + 2
        cols = [e_dpm, e_qpm, t_m, i_dm, i_qm]
        coord.append([row, cols])

        row = ap
        cols = [e_dpm, i_dm, i_qm, vm, va]
        coord.append([row, cols])

        row = ap + 1
        cols = [e_qpm, i_dm, i_qm, vm, va]
        coord.append([row, cols])

        row = dev + 2*self.bus
        cols = [i_dm, i_qm]
        coord.append([row, cols])

        row = dev + 2*self.bus + 1
        cols = [ysh, i_dm, i_qm]
        coord.append([row, cols])

        return coord

    def residual_jac(self, J, z, v, theta, idxs, ctrl_idx, ctrl_var):

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]

        # parameters
        tp = self.tp
        x_0 = self.x0
        x_p = self.x_p
        Hm = self.H
        ra = self.ra

        # states
        e_dpm = z[dp]
        e_qpm = z[dp + 1]
        s = z[dp + 2]
        t_m = z[dp + 3]
        ysh = z[dp + 4]
        i_dm = z[ap]
        i_qm = z[ap + 1]

        vm = v[2*self.bus]
        va = v[2*self.bus + 1]

        # indeces
        e_dpm_idx = dp
        e_qpm_idx = dp + 1
        s_idx = dp + 2
        t_m_idx = dp + 3
        ysh_idx = dp + 4
        i_dm_idx = ap
        i_qm_idx = ap + 1
        vm_idx = dev + 2*self.bus
        va_idx = dev + 2*self.bus + 1

        # column and value vectors
        col = np.zeros(10)
        val = np.zeros(10)

        # first row
        row = dp
        col[0] = e_dpm_idx
        val[0] = -1.0/tp
        col[1] = e_qpm_idx
        val[1] = s*ws
        col[2] = s_idx
        val[2] = e_qpm*ws
        col[3] = i_qm_idx
        val[3] = -1.0*(x_0 - x_p)/tp
        csr_set_row(J.data, J.indptr, J.indices, 4, row, col, val)

        # second row
        row = dp + 1
        col[0] = e_dpm_idx
        val[0] = -s*ws
        col[1] = e_qpm_idx
        val[1] = -1.0/tp
        col[2] = s_idx
        val[2] = -e_dpm*ws
        col[3] = i_dm_idx
        val[3] = -1.0*(-x_0 + x_p)/tp
        csr_set_row(J.data, J.indptr, J.indices, 4, row, col, val)

        # third row
        row = dp + 2
        col[0] = e_dpm_idx
        val[0] = -0.5*i_dm/Hm
        col[1] = e_qpm_idx
        val[1] = -0.5*i_qm/Hm
        col[2] = t_m_idx
        val[2] = 0.5/Hm
        col[3] = i_dm_idx
        val[3] = -0.5*e_dpm/Hm
        col[4] = i_qm_idx
        val[4] = -0.5*e_qpm/Hm
        csr_set_row(J.data, J.indptr, J.indices, 5, row, col, val)

        # algebraic fist  row
        row = ap
        col[0] = e_dpm_idx
        val[0] = 1.0
        col[1] = i_dm_idx
        val[1] = ra
        col[2] = i_qm_idx
        val[2] = -x_p
        col[3] = vm_idx
        val[3] = np.sin(va)
        col[4] = va_idx
        val[4] = vm*np.cos(va)
        csr_set_row(J.data, J.indptr, J.indices, 5, row, col, val)

        # algebraic fist  row
        row = ap + 1
        col[0] = e_qpm_idx
        val[0] = 1.0
        col[1] = i_dm_idx
        val[1] = x_p
        col[2] = i_qm_idx
        val[2] = ra
        col[3] = vm_idx
        val[3] = -np.cos(va)
        col[4] = va_idx
        val[4] = vm*np.sin(va)
        csr_set_row(J.data, J.indptr, J.indices, 5, row, col, val)

        # POWER INJECTION (SET ENTRIES)
        alpha = 1.0

        row = dev + 2*self.bus
        col[0] = i_dm_idx
        col[0] = i_dm_idx
        val[0] = alpha*vm*np.sin(va)
        col[1] = i_qm_idx
        val[1] = -alpha*vm*np.cos(va)
        csr_set_row(J.data, J.indptr, J.indices, 2, row, col, val)

        row = dev + 2*self.bus + 1
        col[0] = ysh_idx
        val[0] = -vm*vm
        col[1] = i_dm_idx
        val[1] = -alpha*vm*np.cos(va)
        col[2] = i_qm_idx
        val[2] = -alpha*vm*np.sin(va)
        csr_set_row(J.data, J.indptr, J.indices, 3, row, col, val)

        # POWER INJECTION (ADD ENTRIES)
        row = dev + 2*self.bus
        col[0] = dev + 2*self.bus
        val[0] = -alpha*(-i_dm*np.sin(va) + i_qm*np.cos(va))
        col[1] = dev + 2*self.bus + 1
        val[1] = -alpha*(-i_dm*vm*np.cos(va) - i_qm*vm*np.sin(va))
        csr_add_row(J.data, J.indptr, J.indices, 2, row, col, val)

        row = dev + 2*self.bus + 1
        col[0] = dev + 2*self.bus
        val[0] = -alpha*(i_dm*np.cos(va) + i_qm*np.sin(va)) - 2*ysh*vm
        col[1] = dev + 2*self.bus + 1
        val[1] = -alpha*(-i_dm*vm*np.sin(va) + i_qm*vm*np.cos(va))
        csr_add_row(J.data, J.indptr, J.indices, 2, row, col, val)

    def preallocate_hessian(self, h_nnz, idxs, psys):

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]

        # these are INDEXES
        e_dpm = dp
        e_qpm = dp + 1
        s = dp + 2
        t_m = dp + 3
        ysh = dp + 4
        i_dm = ap
        i_qm = ap + 1

        vm = dev + 2*self.bus
        va = dev + 2*self.bus + 1

        # F0
        h_nnz[dp]['rows'].append(e_qpm)
        h_nnz[dp]['cols'].append([s])

        h_nnz[dp]['rows'].append(s)
        h_nnz[dp]['cols'].append([e_qpm])

        # F1
        h_nnz[dp + 1]['rows'].append(e_dpm)
        h_nnz[dp + 1]['cols'].append([s])

        h_nnz[dp + 1]['rows'].append(s)
        h_nnz[dp + 1]['cols'].append([e_dpm])

        # F2
        h_nnz[dp + 2]['rows'].append(e_dpm)
        h_nnz[dp + 2]['cols'].append([i_dm])

        h_nnz[dp + 2]['rows'].append(e_qpm)
        h_nnz[dp + 2]['cols'].append([i_qm])

        h_nnz[dp + 2]['rows'].append(i_dm)
        h_nnz[dp + 2]['cols'].append([e_dpm])

        h_nnz[dp + 2]['rows'].append(i_qm)
        h_nnz[dp + 2]['cols'].append([e_qpm])

        # F3
        h_nnz[ap]['rows'].append(vm)
        h_nnz[ap]['cols'].append([va])

        h_nnz[ap]['rows'].append(va)
        h_nnz[ap]['cols'].append([vm, va])

        # F4
        h_nnz[ap + 1]['rows'].append(vm)
        h_nnz[ap + 1]['cols'].append([va])

        h_nnz[ap + 1]['rows'].append(va)
        h_nnz[ap + 1]['cols'].append([vm, va])

        # F5
        h_nnz[dev + 2*self.bus]['rows'].append(i_dm)
        h_nnz[dev + 2*self.bus]['cols'].append([vm, va])

        h_nnz[dev + 2*self.bus]['rows'].append(i_qm)
        h_nnz[dev + 2*self.bus]['cols'].append([vm, va])

        h_nnz[dev + 2*self.bus]['rows'].append(vm)
        h_nnz[dev + 2*self.bus]['cols'].append([i_dm, i_qm, va])

        h_nnz[dev + 2*self.bus]['rows'].append(va)
        h_nnz[dev + 2*self.bus]['cols'].append([i_dm, i_qm, vm, va])

        # F6
        h_nnz[dev + 2*self.bus + 1]['rows'].append(ysh)
        h_nnz[dev + 2*self.bus + 1]['cols'].append([vm])

        h_nnz[dev + 2*self.bus + 1]['rows'].append(i_dm)
        h_nnz[dev + 2*self.bus + 1]['cols'].append([vm, va])

        h_nnz[dev + 2*self.bus + 1]['rows'].append(i_qm)
        h_nnz[dev + 2*self.bus + 1]['cols'].append([vm, va])

        h_nnz[dev + 2*self.bus + 1]['rows'].append(vm)
        h_nnz[dev + 2*self.bus + 1]['cols'].append([ysh, i_dm, i_qm, vm, va])

        h_nnz[dev + 2*self.bus + 1]['rows'].append(va)
        h_nnz[dev + 2*self.bus + 1]['cols'].append([i_dm, i_qm, vm, va])

    def residual_hess(self, HESS, z, v, theta, idxs, ctrl_idx, ctrl_var):

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]
        pp = idxs[3]
        bus = idxs[4]

        H0 = HESS[dp]
        H1 = HESS[dp + 1]
        H2 = HESS[dp + 2]
        H3 = HESS[dp + 3]
        H4 = HESS[dp + 4]
        H5 = HESS[ap]
        H6 = HESS[ap + 1]
        H7 = HESS[dev + 2*bus]
        H8 = HESS[dev + 2*bus + 1]

        # parameters
        tp = self.tp
        x_0 = self.x0
        x_p = self.x_p
        Hm = self.H
        ra = self.ra

        # states
        e_dp = z[dp]
        e_qp = z[dp + 1]
        s = z[dp + 2]
        t_m = z[dp + 3]
        ysh = z[dp + 4]
        i_ds = z[ap]
        i_qs = z[ap + 1]

        vm = v[2*bus]
        va = v[2*bus + 1]

        # indeces
        e_dp_idx = dp
        e_qp_idx = dp + 1
        s_idx = dp + 2
        t_m_idx = dp + 3
        ysh_idx = dp + 4
        i_ds_idx = ap
        i_qs_idx = ap + 1
        vm_idx = dev + 2*bus
        va_idx = dev + 2*bus + 1

        # column and value vectors
        col = np.zeros(10)
        val = np.zeros(10)

        ### HESSIAN OF F0 ###

        row = e_qp_idx
        col[0] = s_idx
        val[0] = ws
        csr_set_row(H0.data, H0.indptr, H0.indices, 1, row, col, val)

        row = s_idx
        col[0] = e_qp_idx
        val[0] = ws
        csr_set_row(H0.data, H0.indptr, H0.indices, 1, row, col, val)

        ### HESSIAN OF F1 ###

        row = e_dp_idx
        col[0] = s_idx
        val[0] = -ws
        csr_set_row(H1.data, H1.indptr, H1.indices, 1, row, col, val)

        row = s_idx
        col[0] = e_dp_idx
        val[0] = -ws
        csr_set_row(H1.data, H1.indptr, H1.indices, 1, row, col, val)

        ### HESSIAN OF F2 ###

        row = e_dp_idx
        col[0] = i_ds_idx
        val[0] = -0.5/Hm
        csr_set_row(H2.data, H2.indptr, H2.indices, 1, row, col, val)

        row = e_qp_idx
        col[0] = i_qs_idx
        val[0] = -0.5/Hm
        csr_set_row(H2.data, H2.indptr, H2.indices, 1, row, col, val)

        row = i_ds_idx
        col[0] = e_dp_idx
        val[0] = -0.5/Hm
        csr_set_row(H2.data, H2.indptr, H2.indices, 1, row, col, val)

        row = i_qs_idx
        col[0] = e_qp_idx
        val[0] = -0.5/Hm
        csr_set_row(H2.data, H2.indptr, H2.indices, 1, row, col, val)

        ### HESSIAN OF F3 ###

        ### HESSIAN OF F4 ###

        ### HESSIAN OF F5 ###

        row = vm_idx
        col[0] = va_idx
        val[0] = np.cos(va)
        csr_set_row(H5.data, H5.indptr, H5.indices, 1, row, col, val)

        row = va_idx
        col[0] = vm_idx
        val[0] = np.cos(va)
        col[1] = va_idx
        val[1] = -vm*np.sin(va)
        csr_set_row(H5.data, H5.indptr, H5.indices, 2, row, col, val)

        ### HESSIAN OF F6 ###

        row = vm_idx
        col[0] = va_idx
        val[0] = np.sin(va)
        csr_set_row(H6.data, H6.indptr, H6.indices, 1, row, col, val)

        row = va_idx
        col[0] = vm_idx
        val[0] = np.sin(va)
        col[1] = va_idx
        val[1] = vm*np.cos(va)
        csr_set_row(H6.data, H6.indptr, H6.indices, 2, row, col, val)

        ### HESSIAN OF F7 ###

        row = i_ds_idx
        col[0] = vm_idx
        val[0] = np.sin(va)
        col[1] = va_idx
        val[1] = vm*np.cos(va)
        csr_set_row(H7.data, H7.indptr, H7.indices, 2, row, col, val)

        row = i_qs_idx
        col[0] = vm_idx
        val[0] = -np.cos(va)
        col[1] = va_idx
        val[1] = vm*np.sin(va)
        csr_set_row(H7.data, H7.indptr, H7.indices, 2, row, col, val)

        row = vm_idx
        col[0] = i_ds_idx
        val[0] = np.sin(va)
        col[1] = i_qs_idx
        val[1] = -np.cos(va)
        csr_set_row(H7.data, H7.indptr, H7.indices, 2, row, col, val)

        row = vm_idx
        col[0] = va_idx
        val[0] = i_ds*np.cos(va) + i_qs*np.sin(va)
        csr_add_row(H7.data, H7.indptr, H7.indices, 1, row, col, val)

        row = va_idx
        col[0] = i_ds_idx
        val[0] = vm*np.cos(va)
        col[1] = i_qs_idx
        val[1] = vm*np.sin(va)
        csr_set_row(H7.data, H7.indptr, H7.indices, 2, row, col, val)

        row = va_idx
        col[0] = vm_idx
        val[0] = i_ds*np.cos(va) + i_qs*np.sin(va)
        col[1] = va_idx
        val[1] = vm*(-i_ds*np.sin(va) + i_qs*np.cos(va))
        csr_add_row(H7.data, H7.indptr, H7.indices, 2, row, col, val)

        ### HESSIAN OF F8 ###

        row = ysh_idx
        col[0] = vm_idx
        val[0] = -2*vm
        csr_set_row(H8.data, H8.indptr, H8.indices, 1, row, col, val)

        row = i_ds_idx
        col[0] = vm_idx
        val[0] = -np.cos(va)
        col[1] = va_idx
        val[1] = vm*np.sin(va)
        csr_set_row(H8.data, H8.indptr, H8.indices, 2, row, col, val)

        row = i_qs_idx
        col[0] = vm_idx
        val[0] = -np.sin(va)
        col[1] = va_idx
        val[1] = -vm*np.cos(va)
        csr_set_row(H8.data, H8.indptr, H8.indices, 2, row, col, val)

        row = vm_idx
        col[0] = ysh_idx
        val[0] = -2*vm
        col[1] = i_ds_idx
        val[1] = -np.cos(va)
        col[2] = i_qs_idx
        val[2] = -np.sin(va)
        csr_set_row(H8.data, H8.indptr, H8.indices, 3, row, col, val)

        row = va_idx
        col[0] = i_ds_idx
        val[0] = vm*np.sin(va)
        col[1] = i_qs_idx
        val[1] = -vm*np.cos(va)
        csr_set_row(H8.data, H8.indptr, H8.indices, 2, row, col, val)

        row = vm_idx
        col[0] = vm_idx
        val[0] = -2*ysh
        col[1] = va_idx
        val[1] = i_ds*np.sin(va) - i_qs*np.cos(va)
        csr_add_row(H8.data, H8.indptr, H8.indices, 2, row, col, val)

        row = va_idx
        col[0] = vm_idx
        val[0] = i_ds*np.sin(va) - i_qs*np.cos(va)
        col[1] = va_idx
        val[1] = vm*(i_ds*np.cos(va) + i_qs*np.sin(va))
        csr_add_row(H8.data, H8.indptr, H8.indices, 2, row, col, val)
