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