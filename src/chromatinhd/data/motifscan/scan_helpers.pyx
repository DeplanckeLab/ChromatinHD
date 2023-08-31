#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#cython: language_level=3
cimport cython
import numpy as np
cimport numpy as np
np.import_array()

INT8 = np.int8
ctypedef np.int8_t INT8_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision
def seq_to_onehot(bytes s, INT8_t [:,::1] out_onehot):
    cdef char* cstr = s
    cdef int i, length = len(s)

    for i in range(length):
        if cstr[i] == b'A':
            out_onehot[i, 0] = 1
        elif cstr[i] == b'C':
            out_onehot[i, 1] = 1
        elif cstr[i] == b'G':
            out_onehot[i, 2] = 1
        elif cstr[i] == b'T':
            out_onehot[i, 3] = 1

    return out_onehot