import sys
import numpy as np
import scipy.io as sio
import networkx as nx
from scipy.sparse import csr_matrix

def createYbusComplex(psys):
    """ Create Ybus matrix from bus and branch data """

    dim  = len(psys.buses)
    # ybus = np.zeros((dim, dim), dtype=complex)

    # this is basically a linked list. Better ways exist.
    ybus_dict = {}

    for branch in psys.branches:

        tap = branch.tap
        shift = branch.shift

        if tap > 0.0:
            tpsh = tap*np.exp(1j*np.pi/180.0*shift)
        else:
            tap = 1.0
            tpsh = 1.0

        fr = branch.fr
        to = branch.to
        y  = (1.0/(branch.r + 1j*branch.x))
        if fr not in ybus_dict:
            ybus_dict[fr] = {}
            ybus_dict[fr][fr] = 0.0
        if to not in ybus_dict:
            ybus_dict[to] = {}
            ybus_dict[to][to] = 0.0
        if fr not in ybus_dict[to]:
            ybus_dict[to][fr] = 0.0
        if to not in ybus_dict[fr]:
            ybus_dict[fr][to] = 0.0

        ybus_dict[fr][fr] += y/(tap*tap)
        ybus_dict[to][to] += y
        ybus_dict[fr][to] -= y/(np.conj(tpsh))
        ybus_dict[to][fr] -= y/(tpsh)

        # charging susceptance

        ybus_dict[fr][fr] += ((1j*0.5*branch.sh)/(tap*tap))
        ybus_dict[to][to] += 1j*0.5*branch.sh

    for shunt in psys.shunts:

        if shunt.bus not in ybus_dict:
            ybus_dict[shunt.bus] = {}
            ybus_dict[shunt.bus][shunt.bus] = 0.0

        ybus_dict[shunt.bus][shunt.bus] += shunt.gsh + 1j*shunt.bsh


    # find number of entries in dictionary
    nnz = 0

    for frbus in ybus_dict:
        for tobus in ybus_dict[frbus]:
            nnz += 1

    data = np.zeros(nnz, dtype=complex)
    row = np.zeros(nnz, dtype=int)
    col = np.zeros(nnz, dtype=int)

    k = 0
    # iterate again to fill the arrays
    for frbus in ybus_dict:
        for tobus in ybus_dict[frbus]:
            row[k] = frbus
            col[k] = tobus
            data[k] = ybus_dict[frbus][tobus]
            k += 1

    ybus_spa = csr_matrix((data, (row, col)), shape=(dim, dim))

    return ybus_spa

def distance_graph(graph, fr, to):
    return nx.shortest_path_length(graph, source=fr, target=to)

def distance_resistance(graph, fr, to):
    return nx.resistance_distance(graph, nodeA=fr, nodeB=to)

def realify_ybus(psys):
    """"
        Given:
        (A + iB)(x + iy) = b + ic

        We will have
        [ B -A] ( x) - (c)
        [ A  B] (-y) - (b)
    """

    ybus = psys.ybus_spa
    nbuses = psys.nbuses
    rybus = np.zeros((2*nbuses, 2*nbuses))
    nnz = ybus.nnz

    # the realified ybus has 4 times the complex nnz
    new_val = np.zeros(4*nnz)
    new_row = np.zeros(4*nnz)
    new_col = np.zeros(4*nnz)

    # iterate sparse complex ybus, find out column and row
    # note, we assume CSR matrix
    c_idx = 0
    for row in range(ybus.shape[0]):
        row_s = ybus.indptr[row]
        row_e = ybus.indptr[row + 1]

        ncols = row_e - row_s
        for i in range(ncols):
            col = ybus.indices[row_s + i]
            entry = ybus.data[row_s + i]

            zr = np.real(entry)
            zi = np.imag(entry)

            new_row[c_idx] = 2*row
            new_col[c_idx] = 2*col
            new_val[c_idx] = zr
            
            new_row[c_idx + 1] = 2*row + 1
            new_col[c_idx + 1] = 2*col + 1
            new_val[c_idx + 1] = zr
            
            new_row[c_idx + 2] = 2*row
            new_col[c_idx + 2] = 2*col + 1
            new_val[c_idx + 2] = -zi
            
            new_row[c_idx + 3] = 2*row + 1
            new_col[c_idx + 3] = 2*col
            new_val[c_idx + 3] = zi

            c_idx += 4

    rybus = csr_matrix((new_val, (new_row, new_col)),
            shape=(2*ybus.shape[0], 2*ybus.shape[1]))
    return rybus
