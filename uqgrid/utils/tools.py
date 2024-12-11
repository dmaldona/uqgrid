import numpy as np
import numba
from numba import jit
import sys

###############
#  ALGEBRA   ##
###############

@jit(nopython=True, cache=True)
def csr_to_zeros(data, indices, indptr):
    " Sets all values of csr matrix to zero "
    for i in range(len(data)):
        data[i] = 0.0

@jit(nopython=True, cache=True)
def csr_add_row(data, indptr, indices, nvalues, row, columns, values):

    # check row is correct
    nrows = indptr.size
    assert row < nrows

    # obtain start and end pointers
    start = indptr[row]
    end = indptr[row + 1]

    # verify number of values 
    assert nvalues <= (end - start)

    for i in range(nvalues):
        nelem = end - start
        for j in range(nelem):
            if columns[i] == indices[start + j]:
                data[start + j] += values[i]
                start += 1
                break
            # This should be false unless entry hasnt been prealocated
            assert j < (nelem - 1)

@jit(nopython=True, cache=True)
def csr_set_row(data, indptr, indices, nvalues, row, columns, values):

    # check row is correct
    nrows = indptr.size
    assert row < nrows

    # obtain start and end pointers
    start = indptr[row]
    end = indptr[row + 1]

    # verify number of values 
    assert nvalues <= (end - start)

    for i in range(nvalues):
        nelem = end - start
        for j in range(nelem):
            if columns[i] == indices[start + j]:
                data[start + j] = values[i]
                start += 1
                break
            # This should be false unless entry hasnt been prealocated
            assert j < (nelem - 1)

@jit(nopython=True, cache=True)
def csr_mult_row(data, indptr, indices, row, scalar):
    # Multiplies all the elements of a row by a scalar

    start = indptr[row]
    end = indptr[row + 1]
    nnz = end - start

    for i in range(nnz):
        data[start + i] = data[start + i]*scalar


def csr_add_row_python(A, row, columns, values):

    for i in range(len(columns)):
        A[row, columns[i]] += values[i]

############
# I/O TOOLS#
############

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

if __name__ == "__main__":

   from scipy.sparse import *

   row = np.array([0, 0, 1, 2, 2, 2])
   col = np.array([0, 2, 2, 0, 1, 2])
   data = np.array([1, 2, 3, 4, 5, 6])

   A = csr_matrix((data, (row, col)), shape=(3, 3), dtype=np.float64)
   csr_to_zeros(A.data, A.indptr, A.indices)
   csr_add_row(A.data, A.indptr, A.indices, 3, 2, np.array([0, 1, 2]), 
           np.array([1, 2, 3]))
   # larger test
   size = 1000
   col = np.arange(size)
   data = np.arange(size)
   row = (size - 1)*np.ones(size)
   
   B = csr_matrix((data, (row, col)), shape=(size, size), dtype=np.float64)

   import time
   start = time.time()
   csr_add_row(B.data, B.indptr, B.indices, size, size - 1, col, data)
   end = time.time()
   print(end - start)
   
   csr_add_row_python(B, size - 1, col, data)
   end2 = time.time()
   print(end2 - end)

   print(A)
   csr_mult_row(A.data, A.indptr, A.indices, 2, 10.0)
   print(A)
