import numpy as np
from numba import njit
from numba.pycc import CC
from numba.np.extensions import cross2d


cc = CC("numba_utils")


@njit
@cc.export('get_unit_vector', 'float32[:](int16[:], int16[:])')
def get_unit_vector(vec1, vec2):
    u_vec = (vec2 - vec1).astype(np.float32)
    u_vec /= np.linalg.norm(u_vec)
    return u_vec


@cc.export('compute_curvature', 'float32(int16[:], int16[:], int16[:])')
def compute_curvature(v1, v2, inter):
    vec1 = get_unit_vector(v1, inter)
    vec2 = get_unit_vector(inter, v2)

    kappa = 2 * cross2d(vec1, vec2) / (1 + np.dot(vec1, vec2))

    curvature = np.abs(kappa)

    return curvature


@cc.export('compute_cumulative_curvature', 'float32(int16[:], int16[:], int16[:], int16[:], int16[:])')
def compute_cumulative_curvature(a1, a2, b1, b2, inter):
    vec_a1 = get_unit_vector(a1, inter)
    vec_a2 = get_unit_vector(inter, a2)
    vec_b1 = get_unit_vector(b1, inter)
    vec_b2 = get_unit_vector(inter, b2)

    kappa1 = 2 * cross2d(vec_a1, vec_a2) / (1 + np.dot(vec_a1, vec_a2))
    kappa2 = 2 * cross2d(vec_b1, vec_b2) / (1 + np.dot(vec_b1, vec_b2))

    total_curvature = np.abs(kappa1) + np.abs(kappa2)

    return total_curvature


@cc.export('traverse_skeleton', 'int16[:, :](uint8[:, :], int16[:])')
def traverse_skeleton(sk, curr_pixel):
    path = [curr_pixel]
    while True:
        x, y = curr_pixel
        sk[x, y] = 0

        view = sk[x-1:x+2, y-1:y+2]
        if view[0, 0]: curr_pixel =   np.array([0, 0], dtype=np.int16)
        elif view[0, 1]: curr_pixel = np.array([0, 1], dtype=np.int16)
        elif view[0, 2]: curr_pixel = np.array([0, 2], dtype=np.int16)
        elif view[1, 0]: curr_pixel = np.array([1, 0], dtype=np.int16)
        elif view[1, 2]: curr_pixel = np.array([1, 2], dtype=np.int16)
        elif view[2, 0]: curr_pixel = np.array([2, 0], dtype=np.int16)
        elif view[2, 1]: curr_pixel = np.array([2, 1], dtype=np.int16)
        elif view[2, 2]: curr_pixel = np.array([2, 2], dtype=np.int16)
        else: curr_pixel = np.array([-1, -1], dtype=np.int16)

        if curr_pixel[0] != -1:
            curr_pixel[0] += x-1
            curr_pixel[1] += y-1
            path.append(curr_pixel)
        else:
            break

    path_len = len(path)
    np_path = np.zeros((path_len, 2), dtype=np.int16)
    for i in range(path_len):
        np_path[i] = path[i]
    return np_path


@cc.export('traverse_skeleton_n_pixels', 'int16[:](uint8[:, :], int16[:], int32)')
def traverse_skeleton_n_pixels(sk, curr_pixel, n):
    for i in range(n):
        x, y = curr_pixel
        sk[x, y] = 0

        view = sk[x-1:x+2, y-1:y+2]
        if view[0, 0]: curr_pixel =   np.array([0, 0], dtype=np.int16)
        elif view[0, 1]: curr_pixel = np.array([0, 1], dtype=np.int16)
        elif view[0, 2]: curr_pixel = np.array([0, 2], dtype=np.int16)
        elif view[1, 0]: curr_pixel = np.array([1, 0], dtype=np.int16)
        elif view[1, 2]: curr_pixel = np.array([1, 2], dtype=np.int16)
        elif view[2, 0]: curr_pixel = np.array([2, 0], dtype=np.int16)
        elif view[2, 1]: curr_pixel = np.array([2, 1], dtype=np.int16)
        elif view[2, 2]: curr_pixel = np.array([2, 2], dtype=np.int16)
        else: return curr_pixel

        curr_pixel[0] += x-1
        curr_pixel[1] += y-1

    return curr_pixel


if __name__ == '__main__':
    print("Compiling utility functions using Numba ahead-of-time (AOT)...")
    cc.compile()
    print("Completed Numba compilation.")
