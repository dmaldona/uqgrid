from __future__ import annotations

from uqgrid.core.psydef import Psystem


def build_two_bus_system() -> Psystem:
    psys = Psystem(basemva=1.0)
    psys.add_bus(1, 3)
    psys.add_bus(2, 1)

    for bus in psys.buses:
        bus.set_vinit(1.0, 0.0)

    psys.add_branch(0, 1, r=0.0, x=0.25)
    psys.add_gen(bus=0, idx_name="G1", psch=0.0, qsch=0.0)
    psys.add_load(bus=1, tag="LD1", pload=0.5, qload=0.3)

    psys.assemble()
    psys.createYbusComplex()
    return psys
