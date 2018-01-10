""" Module storing the kernel of the cythonized calculation of pairwise summation.
"""
import numpy as np
cimport cython # only necessary for the performance-enhancing decorators

@cython.boundscheck(False)  # Assume indexing operations will not cause any IndexErrors to be raised
@cython.wraparound(False)  #  Accessing array elements with negative numbers is not permissible
@cython.nonecheck(False)  #  Never waste time checking whether a variable has been set to None

def simple_cython_engine(double[:] arr, int k):
    
    cdef int i
    for i in range(10):
        arr[i] *= k
    pass


def simple_cython_engine_int(int[:] arr, int k):
    
    cdef int i
    for i in range(10):
        arr[i] *= k
    pass

def simple_cython_engine_truncate(int[:] arr, int k):
    
    return np.sum(arr[:k])


# # The following function does not compile because the 
# # slicing arr[indices] does not work:
# def simple_cython_engine_select(int[:] arr, int[:] indices):
#     return np.sum(arr[indices])