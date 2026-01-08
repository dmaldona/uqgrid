from uqgrid.core.psydef import Psystem, Bus
from uqgrid.models import GenGENROU, ExcESDC1A, GovIEESGO
from uqgrid.models.cim5_imp import MotCIM5
from uqgrid.io.parse_psse import read_raw
import numpy as np
import warnings
import re

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

        if tran.CZ == 1:
            r12 = tran.r*(volt2)**2.0
            x12 = tran.x*(volt2)**2.0
        elif tran.CZ == 2:
            r12 = tran.r*(baseMVA/case.sbase12)*(volt2)**2.0
            x12 = tran.x*(baseMVA/case.sbase12)*(volt2)**2.0
        elif tran.CZ == 3:
            assert False, "Not implemented yet"

        tap = (volt1/volt2)
        psys.add_branch(fr_internal, to_internal, r12, x12, 
                0.0, tap=tap, shift=tran.ANG1)

        if tran.COD1 == 1:
            psys.add_shunt(fr_internal, tran.MAG1*baseMVA, tran.MAG2*baseMVA)
        else:
            # We don't have this implemented. Ensure it it 0
            MAG1 = tran.MAG1
            MAG2 = tran.MAG2
            warnings.warn("Transformer Magnetizing Impedance not Implemented")       
            #assert np.isclose(np.abs(MAG1) + np.abs(MAG2), 0.0), "Not implemented"


    # we will need to create dummy buses if we find three-winding transformers
    MAX_BUSN = max(psse_to_int, key=psse_to_int.get) + 1
    kdummy = 0

    for tran in case.transthree:

        # first, we need to add a dummy bus
        bus_internal = len(psys.buses)
        psys.add_bus(bus_internal, bus_type=2)
        psys.buses[bus_internal].set_vinit(tran.vmstar, tran.anstar)
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
        psys.add_gen(bus, gen.name, gen.pg, gen.qg, 
                     pgub=gen.pt, pglb=gen.pb, qgub=gen.qt, qglb=gen.qb,
                     mbase=gen.mbase)

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
 
        # add shunt
        if mat_buses[i, 4] > 0.0 or mat_buses[i, 5] > 0.0:
                psys.add_shunt(i, mat_buses[i, 4], mat_buses[i, 5])
        # add load
        psys.add_load(i, str(i), mat_buses[i, 2], -mat_buses[i, 3])
        mat_to_int[mat_buses[i, 0]] = i

    for i in range(ngens):
        bus = mat_to_int[int(mat_gens[i, 0])]
        # MATPOWER gen columns: bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin
        pg = mat_gens[i, 1]
        qg = mat_gens[i, 2]
        qmax = mat_gens[i, 3] if mat_gens.shape[1] > 3 else 0.0
        qmin = mat_gens[i, 4] if mat_gens.shape[1] > 4 else 0.0
        pmax = mat_gens[i, 8] if mat_gens.shape[1] > 8 else 0.0
        pmin = mat_gens[i, 9] if mat_gens.shape[1] > 9 else 0.0
        psys.add_gen(bus, "id", pg, qg, pgub=pmax, pglb=pmin, qgub=qmax, qglb=qmin)

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

def add_dyr(psys, dyr_filename, verbose=False):

    assert isinstance(psys, Psystem)
    
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
    
    for device in devices:

        if 'GENROU' in device[1]:
            bus = psys.ext2int[int(device[0])]
            idx = str(device[2]).strip().replace("'", "")
            
            # Skip dynamic models for inactive generators
            if hasattr(psys, 'inactive_gens') and (bus, idx) in psys.inactive_gens:
                if verbose:
                    print("Skipping GENROU for inactive generator at bus %d, idx %s." % (int(device[0]), idx))
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
                    psys.add_gen_dynamics(psys.gens[i],
                        GenGENROU(idx, x_d, x_q, x_dp, x_qp, x_ddp,
                        xl, H, D, T_d0p, T_q0p, T_d0dp, T_q0dp))
                    found_match = True
                    psys.gens[i].set_dynamic_true()
                    if verbose:
                        print("Adding GENROU at bus %d. GENID %s." % (int(device[0]), idx))
                    break

            if not found_match:
                print("Cannot pair GENROU with bus %d and idx %s. Skipping." % (bus, idx))

        if 'IEESGO' in device[1]:
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2])
            if verbose:
                print("Adding IEESGO at bus %d. GENID %s." % (int(device[0]), gen_id))

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

        if 'ESDC1A' in device[1]:

            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2])

            if verbose:
                print("Adding ESDC1A at bus %d. GENID %s." % (int(device[0]), gen_id))

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
                    #psys.add_exc(gen, ExcESDC1A(gen_id, KA, TA, KF, TF1, KE, TE, TR, E1, E2))
                    psys.add_exc(gen, ExcESDC1A(gen_id, 20.0, 1.0, 0.7, 0.7, 7.0, 0.5, 20.4, 0.006, 0.9))
                    break


        if 'CIM5BL' in device[1]:
            bus = psys.ext2int[int(device[0])]
            load_id = str(device[2])
            if verbose:
                print("Adding CIM5BL at bus %d. LOADID %s." % (int(device[0]), load_id))
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

    # check if at the end of loading the dyr any static generator does not have a dynamic model
    # for now we will create dummy GENROUs with large inertias.

    k = 0
    TAG_DUMMY = "DUMMY"
    for i, gen in enumerate(psys.gens):
        if gen.has_dynamic_model is False:
            id_tag = TAG_DUMMY + str(k)
            # we use the parameters of the last generator except for the inertia and damping
            # VERY HACKY
            H = 100.0
            D = 1.0
            # add dummy generator
            psys.add_gen_dynamics(psys.gens[i],
                    GenGENROU(id_tag, x_d, x_q, x_dp, x_qp, x_ddp,
                    xl, H, D, T_d0p, T_q0p, T_d0dp, T_q0dp))
            psys.gens[i].set_dynamic_true()
            k += 1

    if k > 0:
        print("We added %d dummy GENROU models to the system." % k)

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
