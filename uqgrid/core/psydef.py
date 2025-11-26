import numpy as np
from itertools import count
from uqgrid.utils.tools import csr_add_row, csr_set_row
import networkx as nx
import json

# Import base classes from the new file
from uqgrid.core.base_models import DeviceModel, DynamicGenerator, Exciter, Governor, Motor

# IMPORT DEVICE IMPLEMENTATIONS
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

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
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

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        jac_load(z, v, theta, idxs, J.data, J.indptr, J.indices, power_injection)

    def residual_hess(self, H, z, v, theta, idxs):
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
    def __init__(self, bus, idx_name, psch, qsch, basemva, internal_id, mbase):
        self.bus = bus
        self.idx = idx_name
        self.psch = psch/basemva
        self.qsch = qsch/basemva
        self.has_dynamic_model = False
        self.internal_id = internal_id

        if mbase > 0:
            self.mbase = mbase
        else:
            self.mbase = -1

    def set_dynamic_true(self):
        self.has_dynamic_model = True

    def set_dynamic_false(self):
        self.has_dynamic_model = False

class Shunt(object):
    def __init__(self, bus, gsh, bsh, basemva):
        self.bus = bus
        self.gsh = gsh/basemva
        self.bsh = bsh/basemva

class BusFault(object):
    def __init__(self, bus, rfault):

        self.bus = bus
        self.rfault = rfault
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
    
    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        ap = idxs[1]
        wsum = np.dot(self.H, z[self.w_idx])
        hsum = np.sum(self.H)
        F[ap] = z[ap] - (wsum/hsum)
        return None

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        pass
    
    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        
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

    def add_gen(self, bus, idx_name, psch, qsch, mbase=-1):
        self.gens.append(Generator(bus, idx_name, psch, qsch, self.basemva, self.ngens, mbase=mbase))
        self.ngens += 1

    def add_busfault(self, bus, rfault):
        self.fault_events.append(BusFault(bus, rfault))

    def add_gen_dynamics(self, gen, gendynamics):
        assert isinstance(gen, Generator)
        assert isinstance(gendynamics, DynamicGenerator)
        self.gendyn.append(gendynamics)
        self.add_device(self.gendyn[-1])
        gendynamics.set_bus(gen.bus)
        if gen.mbase > 0:
            ratio = gen.mbase/self.basemva
            gendynamics.set_ratio(ratio)
        # pair gen dynamics with static generator
        gendynamics.set_static_gen_idx(gen.internal_id)

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
        ng = len(self.gendyn)
        dif = self.num_dof_dif

        self.gen_pm_ref_idx = np.zeros(ng, dtype=np.int32)
        self.gen_efd_ref_idx = np.zeros(ng, dtype=np.int32)
        self.gen_pm_out_idx = np.zeros(ng, dtype=np.int32)
        self.gen_efd_out_idx = np.zeros(ng, dtype=np.int32)

        self.gen_pm_ctrl_col = np.full(ng, -1, dtype=np.int32)
        self.gen_efd_ctrl_col = np.full(ng, -1, dtype=np.int32)

        self.gov_devices = []
        self.exc_devices = []

        for gi, gen in enumerate(self.gendyn):
            gen.device_index = gi
            gen.has_governor = False
            gen.has_exciter = False

            self.gen_pm_ref_idx[gi] = gen.dif_ptr + 6
            self.gen_efd_ref_idx[gi] = gen.dif_ptr + 7
            self.gen_pm_out_idx[gi] = dif + gen.alg_ptr + 4
            self.gen_efd_out_idx[gi] = dif + gen.alg_ptr + 5

        for gov in self.gov:
            mapped = False
            for gi, gen in enumerate(self.gendyn):
                if gen.governor is gov:
                    gen.has_governor = True
                    gov.gen_index = gi
                    gov.w_idx = gen.dif_ptr + 4
                    self.gen_pm_ctrl_col[gi] = dif + gov.alg_ptr + 0
                    self.gov_devices.append(gov)
                    mapped = True
                    break
            if not mapped:
                raise AssertionError("Governor is not attached to any generator")

        for exc in self.exc:
            mapped = False
            for gi, gen in enumerate(self.gendyn):
                if gen.exciter is exc:
                    gen.has_exciter = True
                    exc.gen_index = gi
                    self.gen_efd_ctrl_col[gi] = exc.dif_ptr + 2
                    self.exc_devices.append(exc)
                    mapped = True
                    break
            if not mapped:
                raise AssertionError("Exciter is not attached to any generator")

        self.gov_mask = np.array([1.0 if gen.has_governor else 0.0
                                  for gen in self.gendyn], dtype=np.float64)
        self.exc_mask = np.array([1.0 if gen.has_exciter else 0.0
                                  for gen in self.gendyn], dtype=np.float64)

        self.p_m_ctrl_aligned = np.zeros(ng, dtype=np.float64)
        self.e_fd_ctrl_aligned = np.zeros(ng, dtype=np.float64)

        for gen in self.gendyn:
            ap = dif + gen.alg_ptr
            gen.pm_idx = ap + 4
            gen.set_pm_idx(ap + 4)
            gen.efd_idx = ap + 5
            gen.set_efd_idx(ap + 5)

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
            
    def create_bus_to_gen_map(self):
        """
        Creates a mapping from bus numbers to generator indices.
        
        Returns:
            list of lists: bus_to_gen[bus_idx] contains the indices of generators connected to bus_idx.
                           For example, if bus_to_gen[3] = [10, 20], it means generators at 
                           indices 10 and 20 in self.gens are connected to bus 3.
        """
        bus_to_gen = [[] for _ in range(self.nbuses)]
        
        for gen_idx, gen in enumerate(self.gens):
            bus_to_gen[gen.bus].append(gen_idx)
        
        return bus_to_gen

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

    def get_load_pq(self):
        """Returns two numpy arrays containing the real and reactive power load values.
        
        Returns:
            tuple: (p_load, q_load) where each is a numpy array of length nloads
        """
        import numpy as np
        p_load = np.zeros(self.nloads)
        q_load = np.zeros(self.nloads)
        
        for i, load in enumerate(self.loads):
            p_load[i] = load.pload
            q_load[i] = load.qload
            
        return p_load, q_load

    def set_load_pq(self, p_load, q_load):
        """Sets the real and reactive power load values.
        
        Args:
            p_load (numpy.ndarray): Array of real power load values
            q_load (numpy.ndarray): Array of reactive power load values
        """
        import numpy as np
        assert len(p_load) == self.nloads, f"p_load length ({len(p_load)}) must match nloads ({self.nloads})"
        assert len(q_load) == self.nloads, f"q_load length ({len(q_load)}) must match nloads ({self.nloads})"
        
        for i, load in enumerate(self.loads):
            load.pload = p_load[i]
            load.qload = q_load[i]

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

    def export_state_metadata(self, filename='state_metadata.json'):
        import json
        metadata = {}
        dif_size = self.num_dof_dif
        alg_size = self.num_dof_alg
        total_size = dif_size + alg_size + 2 * self.nbuses
        
        # Track current index in state vector
        state_idx = 0
        
        # Process differential states
        for device in self.devices:
            model_name = device.__class__.__name__
            device_id = device.id_tag
            
            # Handle differential states
            for i in range(device.dif_dim):
                # Get state name if available, otherwise use generic name
                if hasattr(device, 'state_list') and i < len(device.state_list):
                    state_name = device.state_list[i]
                else:
                    state_name = f"state_{i}"
                    
                metadata[str(state_idx)] = {
                    'type': 'Differential',
                    'model': model_name,
                    'device_id': device_id,
                    'state_name': state_name,
                    'description': f"{model_name} {device_id} {state_name}"
                }
                state_idx += 1
        
        # Process algebraic states
        for device in self.devices:
            model_name = device.__class__.__name__
            device_id = device.id_tag
            
            # Handle algebraic states
            for i in range(device.alg_dim):
                # Get state name if available, otherwise use generic name
                if hasattr(device, 'state_list') and (i + device.dif_dim) < len(device.state_list):
                    state_name = device.state_list[i + device.dif_dim]
                else:
                    state_name = f"alg_state_{i}"
                    
                metadata[str(state_idx)] = {
                    'type': 'Algebraic',
                    'model': model_name,
                    'device_id': device_id,
                    'state_name': state_name,
                    'description': f"{model_name} {device_id} {state_name}"
                }
                state_idx += 1
        
        # Process network voltages - real and imaginary parts
        for i, bus in enumerate(self.buses):
            bus_num = bus.id  # Use the proper attribute (external ID)
            
            # Real part
            metadata[str(state_idx)] = {
                'type': 'Network Voltage',
                'bus_num': bus_num,
                'component': 'real',
                'state_name': 'vr',
                'description': f"Bus {bus_num} voltage real part"
            }
            state_idx += 1
            
            # Imaginary part
            metadata[str(state_idx)] = {
                'type': 'Network Voltage',
                'bus_num': bus_num,
                'component': 'imaginary',
                'state_name': 'vi',
                'description': f"Bus {bus_num} voltage imaginary part"
            }
            state_idx += 1
        
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=4)

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