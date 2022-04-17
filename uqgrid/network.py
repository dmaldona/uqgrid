import sys
import numpy as np
import cmath
import scipy.io as sio
import networkx as nx
from scipy.sparse import csr_matrix

from .psysdef import Psystem
from .parse import load_matpower

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

        # ybus[fr, fr] += y/(tap*tap)
        # ybus[to, to] += y
        # ybus[fr, to] -= y/(np.conj(tpsh))
        # ybus[to, fr] -= y/(tpsh)

        ybus_dict[fr][fr] += y/(tap*tap)
        ybus_dict[to][to] += y
        ybus_dict[fr][to] -= y/(np.conj(tpsh))
        ybus_dict[to][fr] -= y/(tpsh)

        # charging susceptance
        # ybus[to, to] += ((1j*0.5*branch.sh)/(tap*tap))
        # ybus[fr, fr] += 1j*0.5*branch.sh

        ybus_dict[to][to] += ((1j*0.5*branch.sh)/(tap*tap))
        ybus_dict[fr][fr] += 1j*0.5*branch.sh

    for shunt in psys.shunts:

        if fr not in ybus_dict:
            ybus_dict[fr] = {}
            ybus_dict[fr][fr] = 0.0
        if to not in ybus_dict:
            ybus_dict[to] = {}
            ybus_dict[to][to] = 0.0

        # ybus[shunt.bus, shunt.bus] += shunt.gsh + 1j*shunt.bsh
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

    #ybus_sp = csr_matrix(ybus)

    return ybus_spa

def distance_graph(graph, fr, to):
    return nx.shortest_path_length(graph, source=fr, target=to)

def distance_resistance(graph, fr, to):
    return nx.resistance_distance(graph, nodeA=fr, nodeB=to)