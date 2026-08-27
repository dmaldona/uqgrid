from uqgrid.core.psydef import Psystem, Bus
from uqgrid.models import (
    ExcESDC1A,
    ExcESDC2A,
    ExcESST4B,
    ExcESAC1A,
    ExcEXAC1,
    ExcEXAC2,
    ExcIEEET1,
    ExcSEXS,
    GenGENROU,
    GenGENSAL,
    GovGAST,
    GovHYGOV,
    GovIEEEG1,
    GovIEESGO,
    GovTGOV1,
    PssIEEEST,
    StaticGenerator,
)
from uqgrid.models.cim5_imp import MotCIM5
from uqgrid.io.parse_psse import read_raw
import numpy as np
import warnings
import re
import logging
import math

logger = logging.getLogger(__name__)


_SUPPORTED_DYR_MODELS = frozenset(
    {
        "CIM5BL",
        "ESAC1A",
        "ESAC6A",
        "ESDC1A",
        "ESDC2A",
        "ESST4B",
        "EXAC1",
        "EXAC2",
        "EXPIC1",
        "GAST",
        "GENROU",
        "GENSAL",
        "GGOV1",
        "HYGOV",
        "IEEEG1",
        "IEEEST",
        "IEEET1",
        "IEESGO",
        "SCRX",
        "SEXS",
        "TGOV1",
    }
)


def _generator_power_ratios(psys, bus, gen_id):
    for gen in psys.gens:
        static_id = gen.idx.replace("'", "").strip()
        if gen.bus == bus and static_id == gen_id.strip():
            if gen.mbase > 0:
                return psys.basemva / gen.mbase, gen.mbase / psys.basemva
            break
    return 1.0, 1.0


def _dynamic_generator(psys, bus, gen_id):
    for gen in psys.gendyn:
        if gen.bus == bus and gen.id_tag.strip() == gen_id.strip():
            return gen
    return None

def load_psse(raw_filename):

    case = read_raw(raw_filename)

    nbus = len(case.buses)
    nbranch = len(case.branches)
    nloads = len(case.loads)
    psse_to_int = {}
    
    baseMVA = case.baseMVA
    psys = Psystem(basemva=float(baseMVA))
    
    # Track inactive generators to skip their dynamic models in .dyr files
    psys.inactive_gens = set()

    # add buses
    for i in range(nbus):
        psys.add_bus(i, bus_type=case.buses[i].type)
        psys.buses[i].set_vinit(case.buses[i].vm, (np.pi/180.0)*case.buses[i].va)
        psys.buses[i].baseKV = case.buses[i].baseKV
        psse_to_int[case.buses[i].busn] = i

    # add branches
    for branch in case.branches:
        fr_internal = psse_to_int[int(branch.fbus)]
        to_internal = psse_to_int[int(branch.tbus)]

        psys.add_branch(fr_internal, to_internal, branch.r, branch.x, 
                sh=branch.b, rateA=branch.rateA, rateB=branch.rateB, rateC=branch.rateC)
    # add transformers
    for tran in case.transformers:
        
        fr_internal = psse_to_int[int(tran.fbus)]
        to_internal = psse_to_int[int(tran.tbus)]

        if tran.CW == 2:
            assert False, "Not implemented yet"
            volt2 = tran.WINDV2
            volt1 = tran.WINDV1
        else:
            volt2 = tran.WINDV2
            volt1 = tran.WINDV1

        zbase_ratio = 1.0
        if tran.CW == 1 and tran.NOMV1 > 0.0:
            zbase_ratio = (tran.NOMV1 / psys.buses[fr_internal].baseKV) ** 2.0

        if tran.CZ == 1:
            r12 = tran.r*(volt2)**2.0*zbase_ratio
            x12 = tran.x*(volt2)**2.0*zbase_ratio
        elif tran.CZ == 2:
            r12 = tran.r*(baseMVA/case.sbase12)*(volt2)**2.0
            x12 = tran.x*(baseMVA/case.sbase12)*(volt2)**2.0
        elif tran.CZ == 3:
            assert False, "Not implemented yet"

        tap = (volt1/volt2)
        psys.add_branch(fr_internal, to_internal, r12, x12, 
                tran.MAG2 if abs(tran.MAG2) > 0.0 else 0.0, tap=tap, shift=tran.ANG1)

        if tran.COD1 == 1:
            psys.add_shunt(fr_internal, tran.MAG1*baseMVA, tran.MAG2*baseMVA)
        else:
            # We don't have this implemented. Ensure it it 0
            MAG1 = tran.MAG1
            MAG2 = tran.MAG2
            warnings.warn("Transformer Magnetizing Impedance not Implemented")       
            #assert np.isclose(np.abs(MAG1) + np.abs(MAG2), 0.0), "Not implemented"


    # we will need to create dummy buses if we find three-winding transformers
    MAX_BUSN = max(psse_to_int) + 1
    kdummy = 0

    for tran in case.transthree:

        # first, we need to add a dummy bus
        bus_internal = len(psys.buses)
        psys.add_bus(bus_internal, bus_type=Bus.PQ)
        psys.buses[bus_internal].set_vinit(tran.vmstar, (np.pi/180.0)*tran.anstar)
        psse_to_int[MAX_BUSN + kdummy] = bus_internal
        psys.buses[bus_internal].dummy = True
        kdummy += 1

        ibus = psse_to_int[tran.ibus]
        jbus = psse_to_int[tran.jbus]
        kbus = psse_to_int[tran.kbus]

        if tran.CW == 2:
            baseKV1 = psys.buses[ibus].baseKV
            baseKV2 = psys.buses[jbus].baseKV
            baseKV3 = psys.buses[kbus].baseKV

            volt1 = tran.WINDV1/baseKV1
            volt2 = tran.WINDV2/baseKV2
            volt3 = tran.WINDV3/baseKV3
        else:
            volt1 = tran.WINDV1
            volt2 = tran.WINDV2
            volt3 = tran.WINDV3

        #then the impedances are converted to a proper value
        if tran.CZ == 1:
            r12 = tran.r12
            r23 = tran.r23
            r13 = tran.r13

            x12 = tran.x12
            x23 = tran.x23
            x13 = tran.x13

        elif tran.CZ == 2:
            r12 = tran.r12*(baseMVA/tran.sbase12)
            r23 = tran.r23*(baseMVA/tran.sbase23)
            r13 = tran.r13*(baseMVA/tran.sbase13)

            x12 = tran.x12*(baseMVA/tran.sbase12)
            x23 = tran.x23*(baseMVA/tran.sbase23)
            x13 = tran.x13*(baseMVA/tran.sbase13)

        else:
            r12 = (tran.r12 / pow(10,6)) / tran.sbase12
            r23 = (tran.r23 / pow(10,6)) / tran.sbase23
            r13 = (tran.r13 / pow(10,6)) / tran.sbase13

            x12 = np.sqrt(pow(tran.x12, 2) - pow(r12, 2))
            x23 = np.sqrt(pow(tran.x23, 2) - pow(r23, 2))
            x13 = np.sqrt(pow(tran.x13, 2) - pow(r13, 2))

            r12 = r12*(baseMVA/tran.sbase12)
            r23 = r23*(baseMVA/tran.sbase23)
            r13 = r13*(baseMVA/tran.sbase13)

            x12 = x12*(baseMVA/tran.sbase12)
            x23 = x23*(baseMVA/tran.sbase23)
            x13 = x13*(baseMVA/tran.sbase13)

        r1 = 0.5 * (r12 + r13 - r23)
        r2 = 0.5 * (r12 - r13 + r23)
        r3 = 0.5 * (r13 + r23 - r12)

        x1 = 0.5 * (x12 + x13 - x23)
        x2 = 0.5 * (x12 - x13 + x23)
        x3 = 0.5 * (x13 + x23 - x12)

        tap1 = volt1
        tap2 = volt2
        tap3 = volt3

        #The initial status (service) of the transformer.

        if tran.status == 0: s1, s2, s3 = 0, 0, 0
        elif tran.status: s1, s2, s3 = 1, 1, 1
        elif tran.status == 2: s1, s2, s3 = 1, 0, 1
        elif tran.status == 3: s1, s2, s3 = 1, 1, 0
        elif tran.status == 4: s1, s2, s3 = 0, 1, 1

        psys.add_branch(ibus, bus_internal, r1, x1,
                0.0, tap=tap1, shift=tran.ANG1)
        psys.add_branch(bus_internal, jbus, r2, x2,
                0.0, tap=tap2, shift=tran.ANG2)
        psys.add_branch(bus_internal, kbus, r3, x3,
                0.0, tap=tap3, shift=tran.ANG3)

    # add generators
    # First pass: track inactive generators and buses with active generators
    buses_with_active_gens = set()
    for gen in case.gens:
        if gen.status == 0:
            # Track inactive generator for .dyr parsing
            bus = psse_to_int[int(gen.busn)]
            idx = gen.name.strip().replace("'", "").strip()
            psys.inactive_gens.add((bus, idx))
            continue
        bus = psse_to_int[int(gen.busn)]
        buses_with_active_gens.add(bus)
        if psys.buses[bus].type in (Bus.PV, Bus.SLACK):
            psys.buses[bus].set_vinit(gen.vs, psys.buses[bus].v0a)
        psys.add_gen(bus, gen.name, gen.pg, gen.qg, 
                     pgub=gen.pt, pglb=gen.pb, qgub=gen.qt, qglb=gen.qb,
                     mbase=gen.mbase, vset=gen.vs)

    # Downgrade PV buses to PQ only if they have no active generators
    for (bus, idx) in psys.inactive_gens:
        if bus not in buses_with_active_gens and psys.buses[bus].type == Bus.PV:
            psys.buses[bus].type = Bus.PQ

    # add loads
    for i in range(nloads):
        bus = psse_to_int[case.loads[i].busn]
        if case.loads[i].status == 1:
            # Considering only constant-power loads
            psys.add_load(bus, case.loads[i].name, case.loads[i].pl, -case.loads[i].ql)
    
    # adjust alpha for buses where there are multiple loads.
    # Example, in bus 2 there are two loads:
    # ID "A" 100 MW
    # ID "B" 200 MW
    # We calculate the total power -> 300 MW
    # Then alpha = 100/300 = 100.

    for shunt in case.shunts:
        if shunt.status == 1:
            bus = psse_to_int[shunt.busn]
            psys.add_shunt(bus, shunt.gshunt, shunt.bshunt) 

    for shunt in getattr(case, "switched_shunts", []):
        if shunt.status == 1:
            bus = psse_to_int[int(shunt.busn)]
            psys.add_shunt(bus, 0.0, shunt.binit)

    psys.add_ext2int(psse_to_int)
    psys.assemble()

    return psys

def _strip_matpower_comments(lines):
    cleaned = []
    for line in lines:
        if "%" in line:
            line = line.split("%", 1)[0]
        cleaned.append(line)
    return cleaned


def _parse_matpower_matrix(text, key):
    import re

    match = re.search(rf"mpc\.{key}\s*=\s*\[", text)
    if not match:
        raise ValueError(f"MATPOWER file missing required section: mpc.{key}")
    start = match.end()
    end_match = re.search(r"\];", text[start:])
    if not end_match:
        raise ValueError(f"MATPOWER file has unterminated section: mpc.{key}")
    block = text[start:start + end_match.start()]

    rows = []
    for raw_line in _strip_matpower_comments(block.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        for row_text in line.split(";"):
            row = row_text.strip()
            if not row:
                continue
            values = [float(val) for val in row.split()]
            rows.append(values)

    if not rows:
        raise ValueError(f"MATPOWER file section mpc.{key} is empty")
    return np.array(rows, dtype=float)


def _load_matpower_m(m_file):
    import re

    with open(m_file, "r") as f:
        lines = f.readlines()
    text = "\n".join(_strip_matpower_comments(lines))

    base_match = re.search(r"mpc\.baseMVA\s*=\s*([0-9eE+.-]+)", text)
    if not base_match:
        raise ValueError("MATPOWER file missing required field: mpc.baseMVA")
    basemva = float(base_match.group(1))

    mat_buses = _parse_matpower_matrix(text, "bus")
    mat_gens = _parse_matpower_matrix(text, "gen")
    mat_branches = _parse_matpower_matrix(text, "branch")

    return basemva, mat_buses, mat_gens, mat_branches


def load_matpower_raw(m_file):
    """
        Load raw MATPOWER data from a .m case file.
        Returns (baseMVA, bus, gen, branch) as numpy arrays.
    """
    if not m_file.endswith(".m"):
        raise ValueError("MATPOWER raw loader only supports .m case files")
    return _load_matpower_m(m_file)


def load_matpower(mat_file):

    """
        Load MATPOWER data from a case file (.m preferred, .mat supported).
    """

    if not mat_file.endswith(".m"):
        raise ValueError("MATPOWER parser only supports .m case files")

    basemva, mat_buses, mat_gens, mat_branches = _load_matpower_m(mat_file)
        
    nbus = mat_buses.shape[0]
    nbranch = mat_branches.shape[0]
    ngens = mat_gens.shape[0]
    mat_to_int = {}
 
    psys = Psystem(basemva=float(basemva))

    for i in range(nbus):
        psys.add_bus(i, bus_type=mat_buses[i, 1])
        psys.buses[i].set_vinit(mat_buses[i, 7], (np.pi/180.0)*mat_buses[i, 8])
 
        # add shunt (allow negative susceptance/conductance)
        if mat_buses[i, 4] != 0.0 or mat_buses[i, 5] != 0.0:
                psys.add_shunt(i, mat_buses[i, 4], mat_buses[i, 5])
        # add load
        psys.add_load(i, str(i), mat_buses[i, 2], -mat_buses[i, 3])
        mat_to_int[mat_buses[i, 0]] = i

    buses_with_active_gens = set()
    for i in range(ngens):
        bus = mat_to_int[int(mat_gens[i, 0])]
        # MATPOWER gen columns: bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin
        pg = mat_gens[i, 1]
        qg = mat_gens[i, 2]
        vg = mat_gens[i, 5] if mat_gens.shape[1] > 5 else psys.buses[bus].v0m
        qmax = mat_gens[i, 3] if mat_gens.shape[1] > 3 else 0.0
        qmin = mat_gens[i, 4] if mat_gens.shape[1] > 4 else 0.0
        pmax = mat_gens[i, 8] if mat_gens.shape[1] > 8 else 0.0
        pmin = mat_gens[i, 9] if mat_gens.shape[1] > 9 else 0.0
        status = int(mat_gens[i, 7]) if mat_gens.shape[1] > 7 else 1
        if status == 0:
            continue
        buses_with_active_gens.add(bus)
        if psys.buses[bus].type in (Bus.PV, Bus.SLACK):
            psys.buses[bus].set_vinit(vg, psys.buses[bus].v0a)
        psys.add_gen(bus, "id", pg, qg, pgub=pmax, pglb=pmin, qgub=qmax, qglb=qmin, vset=vg)

    for bus_idx, bus in enumerate(psys.buses):
        if bus.type == Bus.PV and bus_idx not in buses_with_active_gens:
            bus.type = Bus.PQ

    for i in range(nbranch):
        fr_internal = mat_to_int[int(mat_branches[i, 0])]
        to_internal = mat_to_int[int(mat_branches[i, 1])]
        # MATPOWER branch columns: fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle
        rateA = mat_branches[i, 5] if mat_branches.shape[1] > 5 else 0.0
        rateB = mat_branches[i, 6] if mat_branches.shape[1] > 6 else 0.0
        rateC = mat_branches[i, 7] if mat_branches.shape[1] > 7 else 0.0

        psys.add_branch(fr_internal, to_internal, mat_branches[i, 2], mat_branches[i, 3], 
                sh=mat_branches[i, 4], tap=mat_branches[i, 8], shift=mat_branches[i, 9],
                rateA=rateA, rateB=rateB, rateC=rateC)

    psys.add_ext2int(mat_to_int)
    psys.assemble()

    return psys


def return_dyr_device(data, dev, ptr):
    ptr += 1
    while dev[-1] != '/':
        dev.extend(data[ptr].strip('\n').split())
        ptr = ptr + 1
    return ptr, dev

def add_dyr(
    psys,
    dyr_filename,
    verbose=False,
    *,
    limit_initialization_policy="adjust",
):

    assert isinstance(psys, Psystem)
    if limit_initialization_policy not in {"adjust", "strict"}:
        raise ValueError(
            "limit_initialization_policy must be 'adjust' or 'strict'."
        )
    adjust_initial_limits = limit_initialization_policy == "adjust"
    
    devices = []

    with open(dyr_filename) as f:
        data = f.readlines()
        ptr = 0
        data_len = len(data)
    
        while ptr < data_len:
            if ',' in data[ptr]:
                # Comma delimited file
                dev = data[ptr].strip('\n').split(",")
            else:
                dev = data[ptr].strip('\n').split()

            if len(dev) == 0:
                # Empty
                ptr = ptr + 1
            elif dev[0][0:2] == '//':
                # Comment
                ptr = ptr + 1
            else:
                ptr, dev = return_dyr_device(data, dev, ptr)
                devices.append(dev)

    pending_ieeest = []

    def attach_ieeest(device, *, final=False):
        bus_external = int(device[0])
        bus = psys.ext2int[bus_external]
        gen_id = str(device[2]).strip().replace("'", "")
        if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
            return True
        source_parameters = device[3:-1]
        if len(source_parameters) != 19:
            raise ValueError(
                f"IEEEST at bus {bus_external}, generator {gen_id} "
                "requires 19 parameters."
            )
        gen = _dynamic_generator(psys, bus, gen_id)
        if gen is None:
            if final:
                logger.warning(
                    "Cannot pair IEEEST with bus %d and idx %s. Skipping.",
                    bus_external, gen_id,
                )
            return False
        if not gen.exciter:
            if final:
                raise ValueError(
                    f"IEEEST at bus {bus_external}, generator {gen_id} "
                    "requires an attached exciter."
                )
            return False
        if verbose:
            logger.info("Adding IEEEST at bus %d. GENID %s.", bus_external, gen_id)
        values = [float(value) for value in source_parameters]
        psys.add_pss(gen, PssIEEEST(gen_id, *values))
        return True

    def retry_pending_ieeest(*, final=False):
        remaining = []
        for pending in pending_ieeest:
            if not attach_ieeest(pending, final=final):
                remaining.append(pending)
        pending_ieeest[:] = remaining

    for device in devices:
        model_name = str(device[1]).strip().strip("'\"").upper()
        if model_name not in _SUPPORTED_DYR_MODELS:
            raise ValueError(
                f"Unsupported DYR model {model_name!r} at bus {device[0]}."
            )

        if model_name == "GENROU":
            bus = psys.ext2int[int(device[0])]
            idx = str(device[2]).strip().replace("'", "")
            
            # Skip dynamic models for inactive generators
            if hasattr(psys, 'inactive_gens') and (bus, idx) in psys.inactive_gens:
                if verbose:
                    logger.info(
                        "Skipping GENROU for inactive generator at bus %d, idx %s.",
                        int(device[0]),
                        idx,
                    )
                continue
            
            T_d0p = float(device[3])
            T_d0dp = float(device[4])
            T_q0p = float(device[5])
            T_q0dp = float(device[6])
            H = float(device[7])
            D = float(device[8])
            x_d = float(device[9])
            x_q = float(device[10])
            x_dp = float(device[11])
            x_qp = float(device[12])
            x_ddp = float(device[13])
            xl = float(device[14])
            S1 = float(device[15])
            S2 = float(device[16])

            # print data
            found_match = False

            for i in range(len(psys.gens)):
                static_bus = psys.gens[i].bus
                static_idx = (psys.gens[i].idx).strip().replace("'", "")

                if static_bus == bus and static_idx.strip() == idx.strip():
                    gen_dyn = GenGENROU(idx, x_d, x_q, x_dp, x_qp, x_ddp,
                        xl, H, D, T_d0p, T_q0p, T_d0dp, T_q0dp, S1, S2)
                    psys.add_gen_dynamics(psys.gens[i], gen_dyn)
                    found_match = True
                    psys.gens[i].set_dynamic_true()
                    if verbose:
                        logger.info(
                            "Adding GENROU at bus %d. GENID %s.",
                            int(device[0]),
                            idx,
                        )
                    break

            if not found_match:
                logger.warning(
                    "Cannot pair GENROU with bus %d and idx %s. Skipping.",
                    int(device[0]),
                    idx,
                )

        if model_name == "GENSAL":
            bus = psys.ext2int[int(device[0])]
            idx = str(device[2]).strip().replace("'", "")

            if hasattr(psys, 'inactive_gens') and (bus, idx) in psys.inactive_gens:
                if verbose:
                    logger.info(
                        "Skipping GENSAL for inactive generator at bus %d, idx %s.",
                        int(device[0]),
                        idx,
                    )
                continue

            T_d0p = float(device[3])
            T_d0dp = float(device[4])
            T_q0dp = float(device[5])
            H = float(device[6])
            D = float(device[7])
            x_d = float(device[8])
            x_q = float(device[9])
            x_dp = float(device[10])
            x_qp = x_dp
            x_ddp = float(device[11])
            xl = float(device[12])
            S1 = float(device[13])
            S2 = float(device[14])
            T_q0p = T_d0p

            found_match = False

            for i in range(len(psys.gens)):
                static_bus = psys.gens[i].bus
                static_idx = (psys.gens[i].idx).strip().replace("'", "")

                if static_bus == bus and static_idx.strip() == idx.strip():
                    gen_dyn = GenGENSAL(
                        idx, x_d, x_q, x_dp, x_ddp, xl, H, D,
                        T_d0p, T_d0dp, T_q0dp, S1, S2,
                    )
                    psys.add_gen_dynamics(psys.gens[i], gen_dyn)
                    found_match = True
                    psys.gens[i].set_dynamic_true()
                    if verbose:
                        logger.info(
                            "Adding GENSAL at bus %d. GENID %s.",
                            int(device[0]),
                            idx,
                        )
                    break

            if not found_match:
                logger.warning(
                    "Cannot pair GENSAL with bus %d and idx %s. Skipping.",
                    int(device[0]),
                    idx,
                )

        if model_name == "IEESGO":
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            if verbose:
                logger.info(
                    "Adding IEESGO at bus %d. GENID %s.",
                    int(device[0]),
                    gen_id,
                )

            T1 = float(device[3])
            T2 = float(device[4])
            T3 = float(device[5])
            T4 = float(device[6])
            T5 = float(device[7])
            T6 = float(device[8])
            K1 = float(device[9])
            K2 = float(device[10])
            K3 = float(device[11])

            for gen in psys.gendyn:
                if gen.bus == bus and gen.id_tag == gen_id:
                    psys.add_gov(gen, GovIEESGO(gen_id, T1, T2, T3, T4, T5, T6,
                        K1, K2, K3))
                    break

        if model_name == "TGOV1":
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            if verbose:
                logger.info(
                    "Adding TGOV1 at bus %d. GENID %s.",
                    int(device[0]),
                    gen_id,
                )

            R = float(device[3])
            T1 = float(device[4])
            VMAX = float(device[5])
            VMIN = float(device[6])
            T2 = float(device[7])
            T3 = float(device[8])
            DT = float(device[9])

            power_ratio, inverse_power_ratio = _generator_power_ratios(
                psys, bus, gen_id
            )
            R *= power_ratio
            VMAX *= inverse_power_ratio
            VMIN *= inverse_power_ratio
            DT *= inverse_power_ratio

            if not math.isfinite(VMIN) or not math.isfinite(VMAX) or VMIN >= VMAX:
                raise ValueError(
                    f"Invalid TGOV1 limits at bus {int(device[0])}, generator {gen_id}."
                )

            found_match = False
            for gen in psys.gendyn:
                if gen.bus == bus and gen.id_tag.strip() == gen_id.strip():
                    psys.add_gov(
                        gen,
                        GovTGOV1(
                            gen_id, R, T1, VMAX, VMIN, T2, T3, DT,
                            enable_limits=True,
                        ),
                    )
                    found_match = True
                    break

            if not found_match:
                logger.warning(
                    "Cannot pair TGOV1 with bus %d and idx %s. Skipping.",
                    int(device[0]),
                    gen_id,
                )

        if model_name == "GGOV1":
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            rselect = int(float(device[3]))
            fswitch = int(float(device[4]))
            if (rselect, fswitch) != (1, 1):
                raise ValueError(
                    "GGOV1 compatibility redirect requires Rselect=1 and Fswitch=1 "
                    f"at bus {int(device[0])}, generator {gen_id}."
                )
            R = float(device[5])
            power_ratio, inverse_power_ratio = _generator_power_ratios(
                psys, bus, gen_id
            )
            governor = GovTGOV1(
                gen_id,
                R * power_ratio,
                0.1,
                1.2 * inverse_power_ratio,
                0.0,
                0.2,
                10.0,
                0.0,
                enable_limits=True,
            )
            governor.source_model = "GGOV1"
            governor.source_parameters = tuple(device[3:-1])
            gen = _dynamic_generator(psys, bus, gen_id)
            if gen is None:
                logger.warning(
                    "Cannot pair GGOV1 with bus %d and idx %s. Skipping.",
                    int(device[0]), gen_id,
                )
            else:
                psys.add_gov(gen, governor)
                psys.dynamic_model_redirects.append(
                    {
                        "source_model": "GGOV1",
                        "effective_model": "TGOV1",
                        "bus": int(device[0]),
                        "device_id": gen_id,
                        "source_parameters": governor.source_parameters,
                    }
                )

        if model_name == "GAST":
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            source_parameters = device[3:-1]
            if len(source_parameters) != 9:
                raise ValueError(
                    f"GAST at bus {int(device[0])}, generator {gen_id} "
                    f"requires 9 parameters; got {len(source_parameters)}."
                )
            values = [float(value) for value in source_parameters]
            R, T1, T2, T3, AT, KT, VMAX, VMIN, DT = values
            power_ratio, inverse_power_ratio = _generator_power_ratios(
                psys, bus, gen_id
            )
            R *= power_ratio
            AT *= inverse_power_ratio
            VMAX *= inverse_power_ratio
            VMIN *= inverse_power_ratio
            DT *= inverse_power_ratio
            gen = _dynamic_generator(psys, bus, gen_id)
            if gen is None:
                logger.warning(
                    "Cannot pair GAST with bus %d and idx %s. Skipping.",
                    int(device[0]), gen_id,
                )
            else:
                psys.add_gov(
                    gen,
                    GovGAST(
                        gen_id, R, T1, T2, T3, AT, KT, VMAX, VMIN, DT,
                        enable_limits=True,
                    ),
                )

        if model_name == "HYGOV":
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            source_parameters = device[3:-1]
            if len(source_parameters) != 12:
                raise ValueError(
                    f"HYGOV at bus {int(device[0])}, generator {gen_id} "
                    f"requires 12 parameters; got {len(source_parameters)}."
                )
            values = [float(value) for value in source_parameters]
            R, r, Tr, Tf, Tg, VELM, GMAX, GMIN, Tw, At, DT, qNL = values
            power_ratio, inverse_power_ratio = _generator_power_ratios(
                psys, bus, gen_id
            )
            R *= power_ratio
            r *= power_ratio
            VELM *= inverse_power_ratio
            GMAX *= inverse_power_ratio
            GMIN *= inverse_power_ratio
            DT *= inverse_power_ratio
            qNL *= inverse_power_ratio
            gen = _dynamic_generator(psys, bus, gen_id)
            if gen is None:
                logger.warning(
                    "Cannot pair HYGOV with bus %d and idx %s. Skipping.",
                    int(device[0]), gen_id,
                )
            else:
                psys.add_gov(
                    gen,
                    GovHYGOV(
                        gen_id, R, r, Tr, Tf, Tg, VELM, GMAX, GMIN, Tw,
                        At, DT, qNL, g_floor=1e-8, enable_limits=True,
                        adjust_initial_limits=adjust_initial_limits,
                    ),
                )

        if model_name == "IEEEG1":
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            source_parameters = device[3:-1]
            if len(source_parameters) != 22:
                raise ValueError(
                    f"IEEEG1 at bus {int(device[0])}, generator {gen_id} "
                    "requires 22 parameters including secondary bus and id; "
                    f"got {len(source_parameters)}."
                )
            bus2_external = int(float(source_parameters[0]))
            id2 = str(source_parameters[1]).strip().replace("'", "")
            values = [float(value) for value in source_parameters[2:]]
            (
                K, T1, T2, T3, UO, UC, PMAX, PMIN, T4, K1, K2, T5,
                K3, K4, T6, K5, K6, T7, K7, K8,
            ) = values
            _, inverse_power_ratio = _generator_power_ratios(psys, bus, gen_id)
            K *= inverse_power_ratio
            UO *= inverse_power_ratio
            UC *= inverse_power_ratio
            PMAX *= inverse_power_ratio
            PMIN *= inverse_power_ratio
            primary = _dynamic_generator(psys, bus, gen_id)
            if primary is None:
                logger.warning(
                    "Cannot pair IEEEG1 with bus %d and idx %s. Skipping.",
                    int(device[0]), gen_id,
                )
                continue
            secondary = None
            if bus2_external != 0:
                if bus2_external not in psys.ext2int:
                    raise ValueError(f"IEEEG1 secondary bus {bus2_external} not found.")
                secondary = _dynamic_generator(
                    psys, psys.ext2int[bus2_external], id2
                )
                if secondary is None:
                    raise ValueError(
                        f"Cannot pair IEEEG1 secondary generator at bus {bus2_external}, idx {id2}."
                    )
            governor = GovIEEEG1(
                gen_id, bus2_external, id2, K, T1, T2, T3, UO, UC,
                PMAX, PMIN, T4, K1, K2, T5, K3, K4, T6, K5, K6,
                T7, K7, K8, enable_limits=True,
                adjust_initial_limits=adjust_initial_limits,
            )
            psys.add_gov(primary, governor, secondary_gen=secondary)

        if model_name == "EXAC1":
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            source_parameters = device[3:-1]
            if len(source_parameters) != 17:
                raise ValueError(
                    f"EXAC1 at bus {int(device[0])}, generator {gen_id} requires 17 parameters."
                )
            values = [float(value) for value in source_parameters]
            gen = _dynamic_generator(psys, bus, gen_id)
            if gen is None:
                logger.warning(
                    "Cannot pair EXAC1 with bus %d and idx %s. Skipping.",
                    int(device[0]), gen_id,
                )
            else:
                if verbose:
                    logger.info("Adding EXAC1 at bus %d. GENID %s.", int(device[0]), gen_id)
                psys.add_exc(gen, ExcEXAC1(gen_id, gen, *values))

        if model_name == "EXAC2":
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            source_parameters = device[3:-1]
            if len(source_parameters) != 23:
                raise ValueError(
                    f"EXAC2 at bus {int(device[0])}, generator {gen_id} requires 23 parameters."
                )
            values = [float(value) for value in source_parameters]
            gen = _dynamic_generator(psys, bus, gen_id)
            if gen is None:
                logger.warning(
                    "Cannot pair EXAC2 with bus %d and idx %s. Skipping.",
                    int(device[0]), gen_id,
                )
            else:
                if verbose:
                    logger.info("Adding EXAC2 at bus %d. GENID %s.", int(device[0]), gen_id)
                psys.add_exc(gen, ExcEXAC2(gen_id, gen, *values))

        if model_name == "ESAC1A":
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            source_parameters = device[3:-1]
            if len(source_parameters) != 19:
                raise ValueError(
                    f"ESAC1A at bus {int(device[0])}, generator {gen_id} requires 19 parameters."
                )
            values = [float(value) for value in source_parameters]
            gen = _dynamic_generator(psys, bus, gen_id)
            if gen is None:
                logger.warning(
                    "Cannot pair ESAC1A with bus %d and idx %s. Skipping.",
                    int(device[0]), gen_id,
                )
            else:
                if verbose:
                    logger.info("Adding ESAC1A at bus %d. GENID %s.", int(device[0]), gen_id)
                (
                    TR, TB, TC, KA, TA, VAMAX, VAMIN, TE, KF, TF,
                    KC, KD, KE, E1, SE1, E2, SE2, VRMAX, VRMIN,
                ) = values
                psys.add_exc(
                    gen,
                    ExcESAC1A(
                        gen_id, gen, TR, TB, TC, VAMAX, VAMIN, KA, TA,
                        VRMAX, VRMIN, TE, E1, SE1, E2, SE2, KC, KD, KE,
                        KF, TF,
                    ),
                )

        if model_name == "ESST4B":
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            source_parameters = device[3:-1]
            values = [float(value) for value in source_parameters]
            if len(values) != 17:
                raise ValueError(
                    f"ESST4B at bus {int(device[0])}, generator {gen_id} requires 17 parameters."
                )
            found_match = False
            for gen in psys.gendyn:
                if gen.bus == bus and gen.id_tag.strip() == gen_id.strip():
                    psys.add_exc(gen, ExcESST4B(gen_id, gen, *values))
                    found_match = True
                    if verbose:
                        logger.info(
                            "Adding ESST4B at bus %d. GENID %s.", int(device[0]), gen_id
                        )
                    break
            if not found_match:
                logger.warning(
                    "Cannot pair ESST4B with bus %d and idx %s. Skipping.",
                    int(device[0]), gen_id,
                )

        if model_name == "IEEEST":
            if not attach_ieeest(device):
                pending_ieeest.append(device)

        if model_name == "ESDC2A":
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            source_parameters = device[3:-1]
            if len(source_parameters) != 16:
                raise ValueError(
                    f"ESDC2A at bus {int(device[0])}, generator {gen_id} requires 16 parameters."
                )
            values = [float(value) for value in source_parameters]
            gen = _dynamic_generator(psys, bus, gen_id)
            if gen is None:
                logger.warning(
                    "Cannot pair ESDC2A with bus %d and idx %s. Skipping.",
                    int(device[0]), gen_id,
                )
            else:
                if verbose:
                    logger.info("Adding ESDC2A at bus %d. GENID %s.", int(device[0]), gen_id)
                (
                    TR, KA, TA, TB, TC, VRMAX, VRMIN, KE, TE, KF, TF1,
                    SW, E1, SE1, E2, SE2,
                ) = values
                psys.add_exc(
                    gen,
                    ExcESDC2A(
                        gen_id, KA, TA, KF, TF1, KE, TE, TR,
                        E1, SE1, E2, SE2, TB, TC, VRMAX, VRMIN, SW,
                        adjust_initial_limits=True,
                    ),
                )

        if model_name == "ESDC1A":

            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            source_parameters = device[3:-1]
            if len(source_parameters) != 16:
                raise ValueError(
                    f"ESDC1A at bus {int(device[0])}, generator {gen_id} requires 16 parameters."
                )

            if verbose:
                logger.info(
                    "Adding ESDC1A at bus %d. GENID %s.",
                    int(device[0]),
                    gen_id,
                )

            TR = float(device[3])
            KA = float(device[4])
            TA = float(device[5])
            TB = float(device[6])
            TC = float(device[7])
            VRMAX = float(device[8])
            VRMIN = float(device[9])
            KE = float(device[10])
            TE = float(device[11])
            KF = float(device[12])
            TF1 = float(device[13])
            SW = float(device[14])
            E1 = float(device[15])
            SE1 = float(device[16])
            E2 = float(device[17])
            SE2 = float(device[18])

            for gen in psys.gendyn:
                if gen.bus == bus and gen.id_tag == gen_id:
                    psys.add_exc(
                        gen,
                        ExcESDC1A(
                            gen_id, KA, TA, KF, TF1, KE, TE, TR,
                            E1, SE1, E2, SE2, TB, TC, VRMAX, VRMIN, SW,
                            adjust_initial_limits=True,
                        ),
                    )
                    break

        if model_name == "SEXS":
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            if verbose:
                logger.info(
                    "Adding SEXS at bus %d. GENID %s.",
                    int(device[0]),
                    gen_id,
                )

            TA_TB = float(device[3])
            TB = float(device[4])
            K = float(device[5])
            TE = float(device[6])
            EMIN = float(device[7])
            EMAX = float(device[8])

            if not math.isfinite(EMIN) or not math.isfinite(EMAX):
                raise ValueError(
                    "Invalid SEXS limits at bus "
                    f"{int(device[0])}, generator {gen_id}: "
                    "EMIN and EMAX must be finite."
                )
            if EMIN >= EMAX:
                raise ValueError(
                    "Invalid SEXS limits at bus "
                    f"{int(device[0])}, generator {gen_id}: "
                    f"EMIN ({EMIN}) must be less than EMAX ({EMAX})."
                )

            found_match = False
            for gen in psys.gendyn:
                if gen.bus == bus and gen.id_tag.strip() == gen_id.strip():
                    psys.add_exc(
                        gen,
                        ExcSEXS(
                            gen_id,
                            TA_TB,
                            TB,
                            K,
                            TE,
                            EMIN,
                            EMAX,
                            enable_limits=True,
                        ),
                    )
                    found_match = True
                    break

            if not found_match:
                logger.warning(
                    "Cannot pair SEXS with bus %d and idx %s. Skipping.",
                    int(device[0]),
                    gen_id,
                )

        if model_name == "EXPIC1":
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            source_parameters = tuple(device[3:-1])
            if len(source_parameters) != 24:
                raise ValueError(
                    f"EXPIC1 at bus {int(device[0])}, generator {gen_id} requires 24 parameters."
                )
            gen = _dynamic_generator(psys, bus, gen_id)
            if gen is None:
                logger.warning(
                    "Cannot pair EXPIC1 with bus %d and idx %s. Skipping.",
                    int(device[0]), gen_id,
                )
            else:
                exciter = ExcSEXS(
                    gen_id, 0.4, 5.0, 20.0, 1.0, -99.0, 99.0,
                    enable_limits=True,
                )
                exciter.source_model = "EXPIC1"
                exciter.source_parameters = source_parameters
                psys.add_exc(gen, exciter)
                psys.dynamic_model_redirects.append(
                    {
                        "source_model": "EXPIC1",
                        "effective_model": "SEXS",
                        "bus": int(device[0]),
                        "device_id": gen_id,
                        "source_parameters": source_parameters,
                    }
                )

        source_model = model_name
        if source_model in {"SCRX", "ESAC6A"}:
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            source_parameters = tuple(device[3:-1])
            expected_parameters = {"SCRX": 8, "ESAC6A": 23}[source_model]
            if len(source_parameters) != expected_parameters:
                raise ValueError(
                    f"{source_model} at bus {int(device[0])}, generator {gen_id} "
                    f"requires {expected_parameters} parameters."
                )
            gen = _dynamic_generator(psys, bus, gen_id)
            if gen is None:
                logger.warning(
                    "Cannot pair %s with bus %d and idx %s. Skipping.",
                    source_model, int(device[0]), gen_id,
                )
            else:
                exciter = ExcSEXS(
                    gen_id, 0.4, 5.0, 20.0, 1.0, -99.0, 99.0,
                    enable_limits=True,
                )
                exciter.source_model = source_model
                exciter.source_parameters = source_parameters
                psys.add_exc(gen, exciter)
                psys.dynamic_model_redirects.append(
                    {
                        "source_model": source_model,
                        "effective_model": "SEXS",
                        "bus": int(device[0]),
                        "device_id": gen_id,
                        "source_parameters": source_parameters,
                    }
                )

        if model_name == "IEEET1":
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2]).strip().replace("'", "")
            if hasattr(psys, 'inactive_gens') and (bus, gen_id) in psys.inactive_gens:
                continue
            if verbose:
                logger.info(
                    "Adding IEEET1 at bus %d. GENID %s.",
                    int(device[0]), gen_id,
                )

            source_parameters = device[3:-1]
            if len(source_parameters) != 14:
                raise ValueError(
                    f"IEEET1 at bus {int(device[0])}, generator {gen_id} requires 14 parameters."
                )
            parameters = [float(value) for value in source_parameters]
            found_match = False
            for gen in psys.gendyn:
                if gen.bus == bus and gen.id_tag.strip() == gen_id.strip():
                    psys.add_exc(gen, ExcIEEET1(gen_id, *parameters))
                    found_match = True
                    break
            if not found_match:
                logger.warning(
                    "Cannot pair IEEET1 with bus %d and idx %s. Skipping.",
                    int(device[0]), gen_id,
                )


        if model_name == "CIM5BL":
            bus = psys.ext2int[int(device[0])]
            load_id = str(device[2])
            if verbose:
                logger.info(
                    "Adding CIM5BL at bus %d. LOADID %s.",
                    int(device[0]),
                    load_id,
                )
            ra = float(device[4])
            xa = float(device[5])
            xm = float(device[6])
            r1 = float(device[7])
            x1 = float(device[8])
            r2 = float(device[9])
            x2 = float(device[10])
            E1 = float(device[11])
            SE1 = float(device[12])
            E2 = float(device[13])
            SE2 = float(device[14])
            MBASE = float(device[15])
            PMULT = float(device[16])
            Hin = float(device[17])
            V1 = float(device[18])
            T1 = float(device[19])
            TB = float(device[20])
            Damp = float(device[21])
            TNOM = float(device[22])

            for load in psys.loads:
                if load.bus == bus and load.id_tag == load_id:
                    psys.add_load_dynamics(load, MotCIM5(load_id, ra, xa, xm, r1,
                        x1, Hin, Damp))
                    break

        if pending_ieeest and model_name != "IEEEST":
            retry_pending_ieeest()

    retry_pending_ieeest(final=True)

    static_gens_by_bus = {}
    for gen in psys.gens:
        if not gen.has_dynamic_model:
            static_gens_by_bus.setdefault(gen.bus, []).append(gen)

    for bus, gens in static_gens_by_bus.items():
        vsets = [gen.vset for gen in gens if gen.vset is not None]
        if vsets and not np.allclose(vsets, vsets[0]):
            raise ValueError(f"Static generators at bus {psys.buses[bus].id} have inconsistent voltage setpoints")
        limits = (
            sum(gen.pglb for gen in gens),
            sum(gen.pgub for gen in gens),
            sum(gen.qglb for gen in gens),
            sum(gen.qgub for gen in gens),
        )
        psys.add_static_gen(StaticGenerator(
            bus,
            [gen.internal_id for gen in gens],
            psys.buses[bus].type,
            vsets[0] if vsets else psys.buses[bus].v0m,
            psys.buses[bus].v0a,
            limits,
        ))

    if static_gens_by_bus:
        logger.info(
            "Retained %d static generators at %d buses.",
            sum(len(gens) for gens in static_gens_by_bus.values()),
            len(static_gens_by_bus),
        )

    if psys.dynamic_model_redirects:
        counts = {}
        for redirect in psys.dynamic_model_redirects:
            key = (redirect["source_model"], redirect["effective_model"])
            counts[key] = counts.get(key, 0) + 1
        logger.warning(
            "Applied dynamic-model compatibility redirects: %s.",
            ", ".join(
                f"{source}->{effective}: {count}"
                for (source, effective), count in sorted(counts.items())
            ),
        )

def load_gic(psys, gis_filename):

    substations = {}
    bus2subs = np.zeros(psys.nbuses, dtype='int64')

    with open(gis_filename, 'r') as f:
        line = f.readline()
        
        sub_re = re.compile(r'(\d+)\s+[\'](.*)[\']\s+(\d+)\s+([\+\-]?\d+[.]\d+)\s+([\+\-]?\d+[.]\d+)')
        while(True):
            line = f.readline()
            if "0 / End of Substation data, Begin Bus Substation Data" in line: break
            if not line: break
            else:
                sub = sub_re.search(line)
                if sub is not None:
                    substations[int(sub.group(1))] = (float(sub.group(5)), float(sub.group(4)))
        
        bus_re = re.compile(r'(\d+)\s+(\d+)')
        while(True):
            line = f.readline()
            if "0 / End of Bus Substation Data, Begin Transformer Data" in line: break
            if not line: break
            else:
                bbs = bus_re.search(line)
                if bbs is not None:
                    extbus = int(bbs.group(1))
                    if extbus in psys.ext2int:
                        bus = psys.ext2int[extbus]
                        subs = int(bbs.group(2))
                        bus2subs[bus] = subs

    psys.add_geo(substations, bus2subs)
