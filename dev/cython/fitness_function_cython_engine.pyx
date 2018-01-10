import numpy as np
cimport cython # only necessary for the performance-enhancing decorators

__all__ = ('fitness_function_cython_engine', )

@cython.boundscheck(False)  # Assume indexing operations will not cause any IndexErrors to be raised
@cython.wraparound(False)  #  Accessing array elements with negative numbers is not permissible
@cython.nonecheck(False)  #  Never waste time checking whether a variable has been set to None

def fitness_function_cython_engine(double[:] x, double[:] y, int[:, :] labels_all, int[:, :] neighbors, 
    double[:] weights, int spherical, double[:] compactness, double[:] equality, double[:, :] counts, 
    double[:] scores):

    def int pop_size = len(labels_all)
    def int ngroup = len(counts[0])

    def int idx_pop, idx_grp

    for idx_pop in range(pop_size):
                
    
    pass