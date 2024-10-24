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

from libc.stdlib cimport abs as c_abs

# cdef means here that this function is a plain C function (so faster).
# To get all the benefits, we type the arguments and the return value.
cdef INT64_t clip(INT64_t a, INT64_t min_value, INT64_t max_value) nogil:
    return min(max(a, min_value), max_value)

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
    cdef INT64_t fragment_ix, coord_first, coord_second, gene_ix, position, motifscore_ix, slice_first_left, slice_first_right, slice_first_mid, slice_second_left, slice_second_mid, slice_second_right, local_fragment_ix
    
    local_fragment_ix = 0 # will store the current fragment counting from 0

    with nogil:
        for fragment_ix in range(coordinates.shape[0]):
            coord_first = coordinates[fragment_ix, 0]
            coord_second = coordinates[fragment_ix, 1]
            gene_ix = genemapping[fragment_ix]
            
            # determine the bounds of the genome slice
            slice_first_left = clip(coord_first + cutwindow_left - window_left, 0, window_width - 1) + gene_ix * window_width
            slice_first_mid = clip(coord_first - window_left, 0, window_width - 1) + gene_ix * window_width
            slice_first_right = clip(coord_first + cutwindow_right - window_left, 0, window_width - 1) + gene_ix * window_width

            slice_second_left = clip(coord_second - cutwindow_right - window_left, 0, window_width - 1) + gene_ix * window_width
            slice_second_mid = clip(coord_second - window_left, 0, window_width - 1) + gene_ix * window_width
            slice_second_right = clip(coord_second - cutwindow_left - window_left, 0, window_width - 1) + gene_ix * window_width

            # avoid "within fragment" slices to overlap with "outside fragment" slices
            if slice_first_right > slice_second_mid:
                slice_first_right = slice_second_mid
            if slice_second_left < slice_first_mid:
                slice_second_left = slice_first_mid

            # avoid the first "within fragment" slice to overlap with the second "within fragment" slice
            # in that case slice_first_right takes priority
            if slice_second_left < slice_first_right:
                slice_second_left = slice_first_right
                
            # outside fragments
            for position in range(slice_first_left, slice_first_mid):
                for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):               
                    out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix]] += 1
            for position in range(slice_second_mid, slice_second_right):
                for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):               
                    out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix]] += 1
            
            # within fragments
            for position in range(slice_first_mid, slice_first_right):
                for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):               
                    out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix]] += 1
            for position in range(slice_second_left, slice_second_mid):
                for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):               
                    out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix]] += 1

                    
            local_fragment_ix += 1
    return


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def extract_dummy(
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

    return None



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def extract_motifcounts_split(
    INT64_t [:,::1] coordinates,
    INT64_t [::1] genemapping,
    INT64_t [::1] motifscores_indptr,
    INT64_t [::1] motifscores_indices,
    FLOAT64_t [::1] motifscores_data,
    INT64_t n_motifs,
    INT64_t window_left,
    INT64_t window_right,
    INT64_t window_width,
    INT64_t cutwindow_left,
    INT64_t cutwindow_right,
    INT64_t [:,::1] out_motifcounts,
):
    cdef INT64_t coord_first, coord_second, gene_ix, slice_first_left, slice_first_right, slice_first_mid, slice_second_left, slice_second_mid, slice_second_right
    cdef int fragment_ix, position, local_fragment_ix, motifscore_ix
    
    local_fragment_ix = 0 # will store the current fragment counting from 0

    with nogil:
        for fragment_ix in range(coordinates.shape[0]):
            coord_first = coordinates[fragment_ix, 0]
            coord_second = coordinates[fragment_ix, 1]
            gene_ix = genemapping[fragment_ix]
            
            # determine the bounds of the genome slice, 0, window_width - 1) + gene_ix * window_width
            slice_first_left = clip(coord_first + cutwindow_left - window_left, 0, window_width - 1) + gene_ix * window_width
            slice_first_mid = clip(coord_first - window_left, 0, window_width - 1) + gene_ix * window_width
            slice_first_right = clip(coord_first + cutwindow_right - window_left, 0, window_width - 1) + gene_ix * window_width
            slice_second_left = clip(coord_second - cutwindow_right - window_left, 0, window_width - 1) + gene_ix * window_width
            slice_second_mid = clip(coord_second - window_left, 0, window_width - 1) + gene_ix * window_width
            slice_second_right = clip(coord_second - cutwindow_left - window_left, 0, window_width - 1) + gene_ix * window_width

            # avoid "within fragment" slices to overlap with "outside fragment" slices
            if slice_first_right > slice_second_mid:
                slice_first_right = slice_second_mid
            if slice_second_left < slice_first_mid:
                slice_second_left = slice_first_mid

            # avoid the first "within fragment" slice to overlap with the second "within fragment" slice
            # in that case slice_first_right takes priority
            if slice_second_left < slice_first_right:
                slice_second_left = slice_first_right
                
            # outside fragments
            for position in range(slice_first_left, slice_first_mid):
                for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):               
                    out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix]] += 1
            for position in range(slice_second_mid, slice_second_right):
                for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):               
                    out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix]] += 1
            
            # within fragments
            for position in range(slice_first_mid, slice_first_right):
                for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):               
                    out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix] + n_motifs] += 1
            for position in range(slice_second_left, slice_second_mid):
                for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):               
                    out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix] + n_motifs] += 1

                    
            local_fragment_ix += 1
    return





@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def extract_motifcounts_multiple(
    INT64_t [:,::1] coordinates,
    INT64_t [::1] genemapping,
    INT64_t [::1] motifscores_indptr,
    INT64_t [::1] motifscores_indices,
    FLOAT64_t [::1] motifscores_data,
    INT64_t n_motifs,
    INT64_t window_left,
    INT64_t window_right,
    INT64_t window_width,
    INT64_t [::1] cutwindows,
    INT64_t [:,::1] out_motifcounts,
):
    cdef INT64_t coord_first, coord_second, gene_ix, slice_left, slice_right, n_cutwindows
    cdef int local_fragment_ix, fragment_ix, cutwindow_ix, motifscore_ix, position
    
    local_fragment_ix = 0 # will store the current fragment counting from 0
    n_cutwindows = cutwindows.shape[0]

    with nogil:
    # if True:
        for fragment_ix in range(coordinates.shape[0]):
            coord_first = coordinates[fragment_ix, 0]
            coord_second = coordinates[fragment_ix, 1]
            gene_ix = genemapping[fragment_ix]

            # left of first cut
            slice_left = coord_first + cutwindows[0] - window_left
            if slice_left < 0:
                slice_left = 0
            if slice_left > window_width:
                slice_left = window_width -1
            slice_left = slice_left + gene_ix * window_width
            for cutwindow_ix in range(n_cutwindows - 1):
                slice_right = coord_first + cutwindows[cutwindow_ix + 1] - window_left
                if slice_right < 0:
                    slice_right = 0
                if slice_right > window_width:
                    slice_right = window_width -1
                slice_right = slice_right + gene_ix * window_width

                for position in range(slice_left, slice_right):
                    for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):               
                        out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix] + n_motifs * cutwindow_ix] += 1

                slice_left = slice_right

            # right of second cut
            slice_right = coord_second - cutwindows[0] - window_left
            if slice_right < 0:
                slice_right = 0
            if slice_right > window_width:
                slice_right = window_width -1
            slice_right = slice_right + gene_ix * window_width
            for cutwindow_ix in range(n_cutwindows - 1):
                slice_left = coord_second - cutwindows[cutwindow_ix + 1] - window_left
                if slice_left < 0:
                    slice_left = 0
                if slice_left > window_width:
                    slice_left = window_width -1
                slice_left = slice_left + gene_ix * window_width

                for position in range(slice_left, slice_right):
                    for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):               
                        out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix] + n_motifs * cutwindow_ix] += 1

                slice_right = slice_left
                    
            local_fragment_ix += 1
    return




@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def extract_motifcounts_relative(
    INT64_t [:,::1] coordinates,
    INT64_t [::1] genemapping,
    INT64_t [::1] motifscores_indptr,
    INT64_t [::1] motifscores_indices,
    FLOAT64_t [::1] motifscores_data,
    INT64_t n_motifs,
    INT64_t window_left,
    INT64_t window_right,
    INT64_t window_width,
    INT64_t cutwindow_left,
    INT64_t cutwindow_right,
    INT64_t promoter_width,
    INT64_t [:,::1] out_motifcounts,
):
    cdef INT64_t position, coord_first, coord_second, gene_ix, slice_first_left, slice_first_right, slice_first_mid, slice_second_left, slice_second_mid, slice_second_right
    cdef int motifscore_ix, fragment_ix, local_fragment_ix, in_promoter
    
    local_fragment_ix = 0 # will store the current fragment counting from 0

    with nogil:
        for fragment_ix in range(coordinates.shape[0]):
            coord_first = coordinates[fragment_ix, 0]
            coord_second = coordinates[fragment_ix, 1]
            gene_ix = genemapping[fragment_ix]

            in_promoter = 0
            if (c_abs(coord_first) < promoter_width) or (c_abs(coord_second) < promoter_width):
                in_promoter = 1
            
            # determine the bounds of the genome slice
            slice_first_left = clip(coord_first + cutwindow_left - window_left, 0, window_width - 1) + gene_ix * window_width
            slice_first_mid = clip(coord_first - window_left, 0, window_width - 1) + gene_ix * window_width
            slice_first_right = clip(coord_first + cutwindow_right - window_left, 0, window_width - 1) + gene_ix * window_width

            slice_second_left = clip(coord_second - cutwindow_right - window_left, 0, window_width - 1) + gene_ix * window_width
            slice_second_mid = clip(coord_second - window_left, 0, window_width - 1) + gene_ix * window_width
            slice_second_right = clip(coord_second - cutwindow_left - window_left, 0, window_width - 1) + gene_ix * window_width

            # avoid "within fragment" slices to overlap with "outside fragment" slices
            if slice_first_right > slice_second_mid:
                slice_first_right = slice_second_mid
            if slice_second_left < slice_first_mid:
                slice_second_left = slice_first_mid

            # avoid the first "within fragment" slice to overlap with the second "within fragment" slice
            # in that case slice_first_right takes priority
            if slice_second_left < slice_first_right:
                slice_second_left = slice_first_right
                
            # outside fragments
            for position in range(slice_first_left, slice_first_mid):
                for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):         
                    out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix] + n_motifs * in_promoter] += 1
            for position in range(slice_second_mid, slice_second_right):
                for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):          
                    out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix] + n_motifs * in_promoter] += 1
            
            # within fragments
            for position in range(slice_first_mid, slice_first_right):
                for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):
                    out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix] + n_motifs * in_promoter] += 1
            for position in range(slice_second_left, slice_second_mid):
                for motifscore_ix in range(motifscores_indptr[position], motifscores_indptr[position+1]):
                    out_motifcounts[local_fragment_ix, motifscores_indices[motifscore_ix] + n_motifs * in_promoter] += 1

                    
            local_fragment_ix += 1
    return

