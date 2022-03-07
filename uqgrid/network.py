import sys
import numpy as np
import cmath
import scipy.io as sio
import networkx as nx
from scipy.sparse import csr_matrix

from .psysdef import Psystem
from .parse import load_matpower

def matprint(mat, fmt="g"):
# code from:
# https://gist.github.com/lbn/836313e283f5d47d2e4e
    if (sys.version_info > (3, 0)):
        col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
        for x in mat:
            for i, y in enumerate(x):
                print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
            print("")
    else:
        print(mat)

def expandComplex(i, j, a, b, mat):
    mat[2*i    , 2*j    ]     += a
    mat[2*i + 1, 2*j + 1]     += a
    mat[2*i    , 2*j + 1]     += -b
    mat[2*i + 1, 2*j    ]     += b


def createYbusReal(psys):
    """ Create Ybus matrix from bus and branch data
        using an homomorphism in R^2 to avoid complex
        algebra """

    dim  = len(psys.buses)
    ybus = np.zeros((2*dim, 2*dim))

    for branch in psys.branches:
        fr = branch.fr
        to = branch.to
        y  = (1.0/(branch.r + 1j*branch.x))
        a  = np.real(y)
        b  = np.imag(y)

        expandComplex(fr, fr, a, b, ybus)
        expandComplex(to, to, a, b, ybus)
        expandComplex(fr, to, -a, -b, ybus)
        expandComplex(to, fr, -a, -b, ybus)

        expandComplex(fr, fr, 0.0, 0.5*branch.sh, ybus)
        expandComplex(to, to, 0.0, 0.5*branch.sh, ybus)

    return ybus

def createYbusComplex(psys):
    """ Create Ybus matrix from bus and branch data """

    dim  = len(psys.buses)
    ybus = np.zeros((dim, dim), dtype=complex)

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
        ybus[fr, fr] += y/(tap*tap)
        ybus[to, to] += y
        ybus[fr, to] -= y/(np.conj(tpsh))
        ybus[to, fr] -= y/(tpsh)

        # charging susceptance
        ybus[to, to] += ((1j*0.5*branch.sh)/(tap*tap))
        ybus[fr, fr] += 1j*0.5*branch.sh

    for shunt in psys.shunts:
        ybus[shunt.bus, shunt.bus] += shunt.gsh + 1j*shunt.bsh

    ybus_sp = csr_matrix(ybus)

    return ybus, ybus_sp

def distance_graph(graph, fr, to):
    return nx.shortest_path_length(graph, source=fr, target=to)

def distance_resistance(graph, fr, to):
    return nx.resistance_distance(graph, nodeA=fr, nodeB=to)
