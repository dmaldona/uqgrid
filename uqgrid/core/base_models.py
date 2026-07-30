import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from uqgrid.utils.tools import csr_add_row, csr_set_row

# constants
ws = 2*np.pi*60


@dataclass(frozen=True)
class BoundedStateMetadata:
    """Model-relative locations for one bounded differential state."""

    state_name: str
    state_offset: int
    lower_parameter_offset: int
    upper_parameter_offset: int
    enabled_parameter_offset: int
    device_type: str


class DeviceModel(ABC):
    """ Base class for device model object.


        Attibutes:
            dif_dim (int): differential degrees of freedom
            alg_dim (int): algebraic degrees of freedom
            id_tag (int): device element tag (external)
            model_type (string): model type
            bus (int) internal bus pointer

            dif_ptr (int): location pointer within global state vector
            alg_ptr (int): location pointer within global algebraic vector
            ndev (int): device number (local to bus)

    """

    bounded_state_metadata = ()

    def __init__(self, ddim, adim, pdim, id_tag, model_type):
        self.dif_dim = ddim
        self.alg_dim = adim
        self.par_dim = pdim
        self.id_tag = id_tag
        self.model_type = model_type
        self.bus = -1
        self.rhs_funs = []

    def getdim(self):
        return self.dif_dim, self.alg_dim, self.par_dim

    def set_pointers(self, dif_ptr, alg_ptr, par_ptr, ndev):
        self.dif_ptr = dif_ptr
        self.alg_ptr = alg_ptr
        self.par_ptr = par_ptr
        # device number
        self.ndev = ndev

    def set_bus(self, bus_ptr):
        self.bus = bus_ptr

    def __str__(self):
        return ("DEVICE ID: {0}\n".format(self.id_tag) +
                "Type: {0}\n".format(self.model_type) +
                "Algebraic dof: %d\tGlobal Pointer: %d\n" %
                (self.alg_dim, self.alg_ptr) +
                "Differential dof: %d\tGlobal Pointer: %d" %
                (self.dif_dim, self.dif_ptr))

    @abstractmethod
    def initialize(self, vm, va, p, q, x, y, psys):
        pass

    @abstractmethod
    def initialize_theta(self, theta):
        pass

    @abstractmethod
    def preallocate_jacobian(self, idxs, psys, power_injection):
        """ Returns a list of coordinates for the Jacobian matrix in the
            format coord = [[row1, [col1, col2], [row2, [col1, col2], ...]

            If no contribution to Jacobian -> return empty list []
        """
        pass

    @abstractmethod
    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        pass

class DynamicGenerator(DeviceModel):
    """ Generic generator class.

        Refer to DeviceModel for additional parameters/methods.

        Attributes:
            initdim (int): degrees of freedom for initialization.

    """

    def __init__(self, id_tag, initdim, ddim, adim, pdim, state_list):
        self.initdim = initdim
        self.state_list = state_list
        DeviceModel.__init__(self, ddim, adim, pdim, id_tag, 'generator')
        self.initialized = False
        self.static_gen_idx = -1  # index of static generator in the bus
        # attached devices
        self.exciter = False
        self.governor = False
        # indexes for control devices (-1 if not present)
        self.pm_idx = -1
        self.efd_idx = -1
        self.pm_initial = 0.0
        self.efd_initial = 0.0

    def set_static_gen_idx(self, idx):
        assert idx >= 0
        self.static_gen_idx = idx

    def set_pm_idx(self, idx):
        assert idx >= 0
        self.pm_idx = idx

    def set_efd_idx(self, idx):
        assert idx >= 0
        self.efd_idx = idx

    def set_pm_val(self, val):
        self.pm_initial = val

    def set_efd_val(self, val):
        self.efd_initial = val

    def set_initpow(self, p0, q0):
        # set initial power, from power flow solution.
        # this will be used in initialization.
        self.p0 = p0
        self.q0 = q0

    def attach_exciter(self, exciter):
        self.exciter = exciter

    def attach_governor(self, governor):
        self.governor = governor
    
    def __str__(self):
        st = "\nInitialized: {0}".format(self.initialized)
        return super().__str__()+st


class Governor(DeviceModel):
    def __init__(self, id_tag, initdim, ddim, adim, pdim, state_list):
        self.initdim = initdim
        self.state_list = state_list
        DeviceModel.__init__(self, ddim, adim, pdim, id_tag, 'governor')
        self.p_m0 = None  # this will be initialized by the generator
        self.w_idx = -1  # location of generator's frequency
        self.pref = None
        self.initialized = False

class Exciter(DeviceModel):
    def __init__(self, id_tag, initdim, ddim, adim, pdim, state_list):
        self.initdim = initdim
        self.state_list = state_list
        DeviceModel.__init__(self, ddim, adim, pdim, id_tag, 'exciter')
        self.e_fd0 = None  # this will be initialized by the generator
        self.vref = None
        self.initialized = False

class Motor(DeviceModel):
    def __init__(self, id_tag, initdim, ddim, adim, pdim, state_list):
        self.initdim = initdim
        self.state_list = state_list
        DeviceModel.__init__(self, ddim, adim, pdim, id_tag, 'motor')
        self.initialized = False

    def set_weight(self, weight):
        self.weight = weight
