#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#cython: language_level=3
cimport cython
import numpy as np
cimport numpy as np
np.import_array()

INT64 = np.int64
ctypedef np.int64_t INT64_t
FLOAT64 = np.float64
ctypedef np.float64_t FLOAT64_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision
def extract_fragments(
    INT64_t [::1] cellxgene_oi,
    INT64_t [::1] cellxgene_indptr,
    INT64_t [:,::1] coordinates,
    INT64_t [::1] genemapping,
    INT64_t [:,::1] out_coordinates,
    INT64_t [::1] out_genemapping,
    INT64_t [::1] out_local_cellxgene_ix,
):
    cdef INT64_t out_ix, local_cellxgene_ix, cellxgene_ix, position
    out_ix = 0 # will store where in the output array we are currently
    local_cellxgene_ix = 0 # will store the current fragment counting from 0

    with nogil:
        for local_cellxgene_ix in range(cellxgene_oi.shape[0]):    
            cellxgene_ix = cellxgene_oi[local_cellxgene_ix]
            for position in range(cellxgene_indptr[cellxgene_ix], cellxgene_indptr[cellxgene_ix+1]):
                out_coordinates[out_ix] = coordinates[position]
                out_genemapping[out_ix] = genemapping[position]
                out_local_cellxgene_ix[out_ix] = local_cellxgene_ix

                out_ix += 1

    return out_ix


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision
def multiple_arange(
    INT64_t [::1] a,
    INT64_t [::1] b,
    INT64_t [::1] ixs,
    INT64_t [::1] local_cellxregion_ix,
):
    cdef INT64_t out_ix, pair_ix, position
    out_ix = 0 # will store where in the output array we are currently
    pair_ix = 0 # will store the current a, b pair index

    with nogil:
        for pair_ix in range(a.shape[0]):
            for position in range(a[pair_ix], b[pair_ix]):
                ixs[out_ix] = position
                local_cellxregion_ix[out_ix] = pair_ix
                out_ix += 1
            pair_ix += 1

    return out_ix



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision
def multiple_arange(
    INT64_t [::1] a,
    INT64_t [::1] b,
    INT64_t [::1] ixs,
    INT64_t [::1] local_cellxregion_ix,
):
    cdef INT64_t out_ix, pair_ix, position
    out_ix = 0 # will store where in the output array we are currently
    pair_ix = 0 # will store the current a, b pair index

    with nogil:
        for pair_ix in range(a.shape[0]):
            for position in range(a[pair_ix], b[pair_ix]):
                ixs[out_ix] = position
                local_cellxregion_ix[out_ix] = pair_ix
                out_ix += 1
            pair_ix += 1

    return out_ix

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision
def count_A_cython(bytes s, INT64_t [:,::1] out_onehot):
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
        else:
            out_onehot[i, 4] = 1

    return out_onehot