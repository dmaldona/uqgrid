from uqgrid.models.esdc1a_imp import ExcESDC1A


class ExcESDC2A(ExcESDC1A):
    """Direct-current commutator exciter with voltage-scaled regulator bounds."""

    bound_scale = "terminal_voltage"
    device_type = "ESDC2A"
