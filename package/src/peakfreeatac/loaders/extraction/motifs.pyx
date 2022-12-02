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
def extract_all(
    INT64_t [:,::1] coordinates,
    INT64_t [::1] genemapping,
    INT64_t [::1] motifscores_indptr,
    INT64_t [::1] motifscores_indices,
    FLOAT64_t [::1] motifscores_data,
    INT64_t window_left,
    INT64_t window_right,
    INT64_t window_width,
    INT64_t cutwindow_left,
    INT64_t cutwindow_right,
    INT64_t [::1] out_fragment_indptr,
    INT64_t [::1] out_motif_ix,
    FLOAT64_t [::1] out_score,
    INT64_t [::1] out_distance,
    INT64_t [:,::1] out_motifcounts,
):
    cdef INT64_t fragment_ix, coord_first, coord_second, gene_ix, position, motifscore_ix, slice_left, slice_right, out_ix, local_fragment_ix, local_motif_n
    
    out_ix = 0 # will store where in the output array we are currently
    local_fragment_ix = 0 # will store the current fragment counting from 0
    local_motif_n = 0 # will store how many motifs have been processed
    
    for fragment_ix in range(coordinates.shape[0]):
        coord_first = coordinates[fragment_ix, 0]
        coord_second = coordinates[fragment_ix, 1]
        gene_ix = genemapping[fragment_ix]
        
        # determine the bounds of the genome slice
        slice_left = coord_first + cutwindow_left - window_left
        if slice_left < 0:
            slice_left = 0
        slice_left = slice_left + gene_ix * window_width
        slice_right = coord_first + cutwindow_right - window_left
        if slice_right >= window_width:
            slice_right = window_width - 1
        slice_right = slice_right + gene_ix * window_width
        
        # loop over each position
        for position in range(slice_left, slice_right):

            # extract for each position its motifs
            for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):
                out_motif_ix[out_ix] = motifscores_indices[motifscore_ix]
                out_score[out_ix] = motifscores_data[motifscore_ix]
                out_distance[out_ix] = position

                # print(motifscores_indices[motifscore_ix])
                
                out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix]] += 1
                
                out_ix += 1
                local_motif_n += 1
                
        out_fragment_indptr[local_fragment_ix + 1] = local_motif_n
                
        local_fragment_ix += 1

    return out_ix



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def extract_motifcounts(
    INT64_t [:,::1] coordinates,
    INT64_t [::1] genemapping,
    INT64_t [::1] motifscores_indptr,
    INT64_t [::1] motifscores_indices,
    FLOAT64_t [::1] motifscores_data,
    INT64_t window_left,
    INT64_t window_right,
    INT64_t window_width,
    INT64_t cutwindow_left,
    INT64_t cutwindow_right,
    INT64_t [:,::1] out_motifcounts,
):
    cdef INT64_t fragment_ix, coord_first, coord_second, gene_ix, position, motifscore_ix, slice_left, slice_right, out_ix, local_fragment_ix, local_motif_n
    
    local_fragment_ix = 0 # will store the current fragment counting from 0
    local_motif_n = 0 # will store how many motifs have been processed
    
    for fragment_ix in range(coordinates.shape[0]):
        coord_first = coordinates[fragment_ix, 0]
        coord_second = coordinates[fragment_ix, 1]
        gene_ix = genemapping[fragment_ix]
        
        # determine the bounds of the genome slice
        slice_left = coord_first + cutwindow_left - window_left
        if slice_left < 0:
            slice_left = 0
        slice_left = slice_left + gene_ix * window_width
        slice_right = coord_first + cutwindow_right - window_left
        if slice_right >= window_width:
            slice_right = window_width - 1
        slice_right = slice_right + gene_ix * window_width
        
        # loop over each position
        for position in range(slice_left, slice_right):

            # extract for each position its motifs
            for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):               
                out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix]] += 1
                
                local_motif_n += 1

        # determine the bounds of the genome slice
        slice_left = coord_second + cutwindow_left - window_left
        if slice_left < 0:
            slice_left = 0
        slice_left = slice_left + gene_ix * window_width
        slice_right = coord_second + cutwindow_right - window_left
        if slice_right >= window_width:
            slice_right = window_width - 1
        slice_right = slice_right + gene_ix * window_width
        
        # loop over each position
        for position in range(slice_left, slice_right):

            # extract for each position its motifs
            for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):               
                out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix]] += 1
                
                local_motif_n += 1
                
        local_fragment_ix += 1

    return None