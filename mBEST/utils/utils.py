import numpy as np


def remove_from_array(base_array, array_to_remove):
    for index in range(len(base_array)):
        if np.array_equal(base_array[index], array_to_remove):
            base_array.pop(index)
            break


def get_boundary_pixels(mask):
    inner_mask = mask[1:-1, 1:-1]
    inner_mask[1:-1, 1:-1] = 0
    vecs = list(np.asarray(np.where(inner_mask)).T)  # oddly faster than np.argwhere
    num_elements = len(vecs)
    curr_ele = 0
    for _ in range(num_elements):
        x, y = vecs[curr_ele]
        x_i = 0 if x-1 < 0 else x-1
        y_i = 0 if y-1 < 0 else y-1
        if np.sum(inner_mask[x_i:x_i+3, y_i:y_i+3]) > 1 and np.sum(mask[x:x + 3, y:y + 3]) == 3:
            vecs.pop(curr_ele)
        else:
            curr_ele += 1
    return np.asarray(vecs, dtype=np.int16)
