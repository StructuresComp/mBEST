#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp
cnp.import_array()


"""
    Taken from scikit-image's implementation:
    https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/morphology/_skeletonize_cy.pyx
    
    Recompiling from source and removal of "nogil" offers considerable speed boost.
"""


def skeletonize(image):
    """Optimized parts of the Zhang-Suen [1]_ skeletonization.
    Iteratively, pixels meeting removal criteria are removed,
    till only the skeleton remains (that is, no further removable pixel
    was found).

    Performs a hard-coded correlation to assign every neighborhood of 8 a
    unique number, which in turn is used in conjunction with a look up
    table to select the appropriate thinning criteria.

    Parameters
    ----------
    image : numpy.ndarray
        A binary image containing the objects to be skeletonized. '1'
        represents foreground, and '0' represents background.

    Returns
    -------
    skeleton : ndarray
        A matrix containing the thinned image.

    References
    ----------
    .. [1] A fast parallel algorithm for thinning digital patterns,
           T. Y. Zhang and C. Y. Suen, Communications of the ACM,
           March 1984, Volume 27, Number 3.

    """

    # look up table - there is one entry for each of the 2^8=256 possible
    # combinations of 8 binary neighbours. 1's, 2's and 3's are candidates
    # for removal at each iteration of the algorithm.
    cdef int *lut = \
      [0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0,
       0, 0, 2, 0, 2, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0,
       0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1,
       0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 0, 0,
       1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3,
       0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0]

    cdef int pixel_removed, first_pass, neighbors

    # indices for fast iteration
    cdef Py_ssize_t row, col, nrows = image.shape[0]+2, ncols = image.shape[1]+2

    # we copy over the image into a larger version with a single pixel border
    # this removes the need to handle border cases below
    _skeleton = np.zeros((nrows, ncols), dtype=np.uint8)
    pixel_counts = np.zeros_like(_skeleton)
    _skeleton[1:nrows-1, 1:ncols-1] = image > 0

    _cleaned_skeleton = _skeleton.copy()

    # cdef'd numpy-arrays for fast, typed access
    cdef cnp.uint8_t [:, ::1] skeleton, cleaned_skeleton

    skeleton = _skeleton
    cleaned_skeleton = _cleaned_skeleton

    pixel_removed = True

    # the algorithm reiterates the thinning till
    # no further thinning occurred (variable pixel_removed set)

    # with nogil:
    while pixel_removed:
        pixel_removed = False

        # there are two phases, in the first phase, pixels labeled (see below)
        # 1 and 3 are removed, in the second 2 and 3

        # nogil can't iterate through `(True, False)` because it is a Python
        # tuple. Use the fact that 0 is Falsy, and 1 is truthy in C
        # for the iteration instead.
        # for first_pass in (True, False):
        for pass_num in range(2):
            first_pass = (pass_num == 0)
            for row in range(1, nrows-1):
                for col in range(1, ncols-1):
                    # all set pixels ...
                    if skeleton[row, col]:
                        # are correlated with a kernel (coefficients spread around here ...)
                        # to apply a unique number to every possible neighborhood ...

                        # which is used with the lut to find the "connectivity type"

                        neighbors = lut[  1*skeleton[row - 1, col - 1] +   2*skeleton[row - 1, col] +\
                                          4*skeleton[row - 1, col + 1] +   8*skeleton[row, col + 1] +\
                                         16*skeleton[row + 1, col + 1] +  32*skeleton[row + 1, col] +\
                                         64*skeleton[row + 1, col - 1] + 128*skeleton[row, col - 1]]

                        # if the condition is met, the pixel is removed (unset)
                        if ((neighbors == 1 and first_pass) or
                                (neighbors == 2 and not first_pass) or
                                (neighbors == 3)):
                            cleaned_skeleton[row, col] = 0
                            pixel_removed = True

            # once a step has been processed, the original skeleton
            # is overwritten with the cleaned version
            skeleton[:, :] = cleaned_skeleton[:, :]

    return _skeleton[1:nrows - 1, 1:ncols - 1].astype(bool)
