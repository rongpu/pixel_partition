from __future__ import division, print_function
import numpy as np
from simple_cython_engine import simple_cython_engine, simple_cython_engine_int

k = 2
a = np.arange(10, dtype=np.float64)
print(a)
simple_cython_engine(a, k)
print(a)

print('')

# int or long doesn not matter for k
k = long(2)
# the default np.int64 does not work!!
b = np.arange(10, dtype=np.int32)
print(b)
simple_cython_engine_int(b, k)
print(b)


# int or long doesn not matter for k
k = True
# the default np.int64 does not work!!
b = np.arange(10, dtype=np.int32)
print(b)
simple_cython_engine_int(b, k)
print(b)
