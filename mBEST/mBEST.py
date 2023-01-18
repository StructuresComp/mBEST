import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time
from itertools import combinations
from skimage.draw import line
from sklearn.cluster import DBSCAN
import mBEST.utils as ut
import mBEST.skeletonize as sk


class mBEST:
    def __init__(self, epsilon=40, delta=25, colors=None):
        self.epsilon = epsilon
        self.delta = delta

        self.end_point_kernel = np.array(([1, 1, 1], [1, 10, 1], [1, 1, 1]), dtype=np.uint8)

        self.intersection_clusterer = DBSCAN(eps=self.epsilon, min_samples=1)
        self.adjacent_pixel_clusterer = DBSCAN(eps=3, min_samples=1)

        self.image = None
        self.blurred_image = None

        self.colors = colors
        if colors is None:
            self.colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0],
                           [0, 255, 255], [255, 255, 0], [255, 0, 255]]

    def set_image(self, image, blur_size=5):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.blurred_image = cv2.blur(self.image, (blur_size, blur_size))

    def _detect_keypoints(self, skeleton):
        padded_img = np.zeros((skeleton.shape[0]+2, skeleton.shape[1]+2), dtype=np.uint8)
        padded_img[1:-1, 1:-1] = skeleton
        res = cv2.filter2D(src=padded_img, ddepth=-1, kernel=self.end_point_kernel)
        ends = np.argwhere(res == 11) - 1
        intersections = np.argwhere(res > 12) - 1
        return ends, intersections

    def _prune_split_ends(self, skeleton, ends, intersections):
        my_skeleton = skeleton.copy()
        inter_indices = [True for _ in intersections]
        inter_index_dict = {"{},{}".format(x, y): i for i, (x, y) in enumerate(intersections)}
        valid_ends = []
        ends = list(ends)  # so that we can append new ends

        for i, e in enumerate(ends):
            curr_pixel = e
            path = [curr_pixel]
            prune = False
            found_nothing = False

            while True:
                x, y = curr_pixel
                my_skeleton[curr_pixel[0], curr_pixel[1]] = 0
                c_x, c_y = np.where(my_skeleton[x-1:x+2, y-1:y+2])
                c_x += x-1
                c_y += y-1

                num_neighbors = len(c_x)
                # Keep following segment
                if num_neighbors == 1:
                    curr_pixel = [c_x[0], c_y[0]]
                    path.append(curr_pixel)
                # We've reached an end
                elif num_neighbors == 0:
                    found_nothing = True
                    break
                # Found an intersection
                else:
                    # Remove intersection pixels from list
                    ind = "{},{}".format(curr_pixel[0], curr_pixel[1])
                    inter_indices[inter_index_dict[ind]] = False

                    for j in range(-2, 3):
                        for k in range(-2, 3):
                            ind = "{},{}".format(j+curr_pixel[0], k+curr_pixel[1])
                            if ind in inter_index_dict:
                                inter_indices[inter_index_dict[ind]] = False

                    prune = True
                    break

                # This is most likely a valid segment
                if len(path) > self.delta:
                    break

            if found_nothing: continue
            path = np.asarray(path)

            # Prune noisy segment from skeleton.
            if prune:
                skeleton[path[:, 0], path[:, 1]] = 0
                x, y = path[-1]
                skeleton[x-2:x+3, y-2:y+3] = 0
                vecs = ut.get_boundary_pixels(skeleton[x-4:x+5, y-4:y+5])

                # Reconnect the segments together after pruning a branch
                if len(vecs) == 2:
                    vecs[:, 0] += x - 3
                    vecs[:, 1] += y - 3
                    reconnected_line = line(vecs[0][0], vecs[0][1], vecs[1][0], vecs[1][1])
                    skeleton[reconnected_line[0], reconnected_line[1]] = 1

                # We created a new end so add it.
                elif len(vecs) == 1:
                    vecs = vecs.squeeze()
                    vecs[0] += x - 3
                    vecs[1] += y - 3
                    ends.append(vecs)

                # Reflect the changes on our skeleton copy.
                my_skeleton[x-2:x+3, y-2:y+3] = skeleton[x-2:x+3, y-2:y+3]

            else:
                valid_ends.append(i)

        ends = np.asarray(ends)
        return ends[valid_ends], intersections[inter_indices]

    def _cluster_intersections(self, intersections):
        # Clustering intersections consists of two phases
        # 1st phase: cluster adjacent intersection pixels into clusters
        # 2nd phase: cluster intersection clusters together
        temp_intersections = []
        new_intersections = []

        # 1st phase
        self.adjacent_pixel_clusterer.fit(intersections)
        for i in np.unique(self.adjacent_pixel_clusterer.labels_):
            temp_intersections.append(
                np.round(np.mean(intersections[self.adjacent_pixel_clusterer.labels_ == i], axis=0)).astype(np.uint16))

        temp_intersections = np.asarray(temp_intersections)

        # 2nd phase
        self.intersection_clusterer.fit(temp_intersections)
        for i in np.unique(self.intersection_clusterer.labels_):
            new_intersections.append(
                np.round(np.mean(temp_intersections[self.intersection_clusterer.labels_ == i], axis=0)).astype(np.uint16))

        return np.asarray(new_intersections, dtype=np.int16)

    @staticmethod
    def _compute_minimal_bending_energy_paths(ends, inter):
        indices = [i for i in range(len(ends))]
        all_pairs = list(combinations(indices, 2))
        best_paths = {}

        # This should preferably be the case for every intersection, but could not be because of noise.
        if len(ends) == 4:
            possible_path_pairs = set()
            already_added = set()
            n = sum(indices)
            for c1 in all_pairs:
                if c1 in already_added: continue
                for c2 in all_pairs:
                    if c1 == c2 or sum(c1) + sum(c2) != n: continue
                    already_added.add(c1)
                    already_added.add(c2)
                    possible_path_pairs.add((c1, c2))
                    continue

            minimum_total_curvature = np.inf
            for (a1, a2), (b1, b2) in possible_path_pairs:
                total_curvature = ut.compute_cumulative_curvature(ends[a1], ends[a2],
                                                                  ends[b1], ends[b2], inter)

                if total_curvature < minimum_total_curvature:
                    minimum_total_curvature = total_curvature
                    best_paths[a1] = a2
                    best_paths[a2] = a1
                    best_paths[b1] = b2
                    best_paths[b2] = b1

        elif len(ends) == 3:
            minimum_curvature = np.inf
            total = 3  # 0 + 1 + 2
            for v1, v2 in all_pairs:
                curvature = ut.compute_curvature(ends[v1], ends[v2], inter)

                if curvature < minimum_curvature:
                    minimum_curvature = curvature
                    best_paths[v1] = v2
                    best_paths[v2] = v1

                    # Make the last path end here
                    third_path = total - (v1 + v2)
                    best_paths[third_path] = None

        else:
            return False

        return best_paths

    def _generate_intersection_paths(self, skeleton, intersections):
        paths_to_ends = {}
        crossing_orders = {}

        for inter in intersections:
            x, y = inter
            best_paths = False
            k_size = int(self.epsilon * 0.4)
            three_way = False

            # Compute the best paths through the intersection, i.e. the one that minimizes total bending energy.
            while best_paths is False:
                skeleton[x-k_size:x+k_size+1, y-k_size:y+k_size+1] = 0
                ends = ut.get_boundary_pixels(skeleton[x-k_size-2:x+k_size+3, y-k_size-2:y+k_size+3])
                ends = ends.reshape((-1, 2))

                ends[:, 0] += x-k_size-1
                ends[:, 1] += y-k_size-1

                best_paths = self._compute_minimal_bending_energy_paths(ends, inter)
                k_size += 5

            generated_paths = [list(np.asarray(line(e[0], e[1], inter[0], inter[1])).T[:-1]) for e in ends]

            for i, (x1, y1) in enumerate(ends):
                if best_paths[i] is None:
                    three_way = True
                    continue
                x2, y2 = ends[best_paths[i]]
                # Construct a path that minimizes the total bending energy of the intersection.
                if i < best_paths[i]:
                    constructed_path = generated_paths[i] + [inter] + generated_paths[best_paths[i]][::-1] + [[x2, y2]]
                    constructed_path = np.asarray(constructed_path, dtype=np.int16)
                # If we already constructed the reverse path, just flip and reuse.
                else:
                    constructed_path = np.flip(paths_to_ends["{},{}".format(x2, y2)], axis=0)
                    constructed_path[:-1] = constructed_path[1:]
                    constructed_path[-1] = [x2, y2]
                paths_to_ends["{},{}".format(x1, y1)] = constructed_path

            if three_way: continue

            # Determine crossing order
            possible_paths = [1, 2, 3]
            possible_paths.remove(best_paths[0])
            x11, y11 = ends[0]
            x12, y12 = ends[best_paths[0]]
            x21, y21 = ends[possible_paths[0]]
            x22, y22 = ends[possible_paths[1]]
            id11 = "{},{}".format(x11, y11)
            id12 = "{},{}".format(x12, y12)
            id21 = "{},{}".format(x21, y21)
            id22 = "{},{}".format(x22, y22)
            p1 = paths_to_ends[id11]
            p2 = paths_to_ends[id21]

            # Using blurred image is key to getting rid of influence from glare
            std1 = self.blurred_image[p1[:, 0], p1[:, 1]].std(axis=0).sum()
            std2 = self.blurred_image[p2[:, 0], p2[:, 1]].std(axis=0).sum()

            if std1 > std2:
                crossing_orders[id11] = 0
                crossing_orders[id12] = 0
                crossing_orders[id21] = 1
                crossing_orders[id22] = 1
            else:
                crossing_orders[id11] = 1
                crossing_orders[id12] = 1
                crossing_orders[id21] = 0
                crossing_orders[id22] = 0

        return paths_to_ends, crossing_orders

    @staticmethod
    def _generate_paths(skeleton, ends, intersection_paths):
        ends = list(ends.astype(np.int16))
        paths = []
        path_id = 0
        intersection_path_id = {}

        while len(ends) != 0:
            curr_pixel = ends.pop()
            done = False
            path = []

            visited = set()

            while not done:
                path += list(ut.traverse_skeleton(skeleton, curr_pixel))
                p_x, p_y = path[-1]
                id = "{},{}".format(p_x, p_y)

                # We found an intersection, let's add our precomputed path to it.
                if id in intersection_paths:
                    if id in visited:  # found a cycle
                        paths.append(np.asarray(path) - 1)  # -1 for offset
                        break
                    visited.add(id)
                    path += list(intersection_paths[id])
                    curr_pixel = np.array([path[-1][0], path[-1][1]], dtype=np.int16)
                    intersection_path_id[id] = path_id
                    continue
                # We've finished this path.
                else:
                    paths.append(np.asarray(path)-1)  # -1 for offset
                    # Remove the end so that we don't traverse again in opposite direction.
                    ut.remove_from_array(ends, path[-1])
                    break

            path_id += 1

        return paths, intersection_path_id

    @staticmethod
    def _compute_radii(mask, paths):
        dist_img = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        path_radii = np.round(dist_img).astype(np.int)
        path_radii_avgs = [int(np.round(dist_img[path[:, 0], path[:, 1]].mean())) for path in paths]
        return path_radii, path_radii_avgs

    def _plot_paths(self, paths, intersection_paths, intersection_path_id,
                    crossing_orders, path_radii_data, intersection_color=None):
        path_img = np.zeros_like(self.image)

        path_radii, path_radii_avgs = path_radii_data
        
        end_lengths = self.epsilon
        end_buffer = 10 if end_lengths > 10 else int(end_lengths * 0.5)
        img_height, img_width = self.image.shape[1], self.image.shape[0]
        left_limit = end_lengths
        right_limit = img_width - int(end_lengths * 0.5)
        bottom_limit = end_lengths
        top_limit = img_height - int(end_lengths * 0.5)

        # Generate segmentation along the DLO path(s)
        for i, path in enumerate(paths):
            for x, y in path[:end_buffer]:
                cv2.circle(path_img, (y, x), path_radii[x, y], self.colors[i], -1)
            for x, y in path[end_buffer:end_lengths]:
                if x < left_limit or x > right_limit or y < bottom_limit or y > top_limit:
                    cv2.circle(path_img, (y, x), path_radii[x, y], self.colors[i], -1)
                else:
                    cv2.circle(path_img, (y, x), path_radii_avgs[i], self.colors[i], -1)
            for x, y in path[end_lengths:-end_lengths]:
                cv2.circle(path_img, (y, x), path_radii_avgs[i], self.colors[i], -1)
            for x, y in path[-end_lengths:-end_buffer]:
                if x < left_limit or x > right_limit or y < bottom_limit or y > top_limit:
                    cv2.circle(path_img, (y, x), path_radii[x, y], self.colors[i], -1)
                else:
                    cv2.circle(path_img, (y, x), path_radii_avgs[i], self.colors[i], -1)
            for x, y in path[-end_buffer:]:
                cv2.circle(path_img, (y, x), path_radii[x, y], self.colors[i], -1)

        # Handle intersections with appropriate crossing order
        for id, path_id in intersection_path_id.items():
            if id not in crossing_orders or crossing_orders[id] == 1: continue
            color = self.colors[path_id]
            for x, y in intersection_paths[id]:
                cv2.circle(path_img, (y-1, x-1), path_radii_avgs[path_id], color, -1)
        for id, path_id in intersection_path_id.items():
            if id not in crossing_orders or crossing_orders[id] == 0: continue
            color = self.colors[path_id] if intersection_color is None else intersection_color
            for x, y in intersection_paths[id]:
                cv2.circle(path_img, (y-1, x-1), path_radii_avgs[path_id], color, -1)

        return path_img

    def run(self, orig_mask, intersection_color=None, plot=False, save_fig=False, save_id=0):
        if self.image is None:
            raise RuntimeError("Add image to mBEST using set_image function.")
        times = []

        # Create the skeleton pixels.
        s = time()
        img = np.zeros((orig_mask.shape[0]+2, orig_mask.shape[1]+2), dtype=np.uint8)
        img[1:-1, 1:-1] = orig_mask
        mask = img == 0
        img[mask] = 0
        img[~mask] = 1
        skeleton = sk.skeletonize(img)
        times.append(time() - s)
        print("Skeletonizing time: {:.5f}".format(times[-1]))

        # Keypoint Detection
        s = time()
        ends, intersections = self._detect_keypoints(skeleton)
        times.append(time() - s)
        print("Keypoint detection time: {:.5f}".format(times[-1]))

        # Prune noisy split ends.
        s = time()
        ends, intersections = self._prune_split_ends(skeleton, ends, intersections)
        times.append(time() - s)
        print("Prune time: {:.5f}".format(times[-1]))

        intersection_paths = {}
        crossing_orders = {}
        if len(intersections > 0):
            s = time()
            intersections = self._cluster_intersections(intersections)
            times.append(time() - s)
            print("Intersection cluster time: {:.5f}".format(times[-1]))

            s = time()
            intersection_paths, crossing_orders = self._generate_intersection_paths(skeleton, intersections)
            times.append(time() - s)
            print("Replace intersections time: {:.5f}".format(times[-1]))

        s = time()
        paths, intersection_path_id = self._generate_paths(skeleton, ends, intersection_paths)
        times.append(time() - s)
        print("Path generation time: {:.5f}".format(times[-1]))

        if plot:
            s = time()
            path_radii = self._compute_radii(orig_mask, paths)
            times.append(time() - s)
            print("Computing radii time: {:.5f}".format(times[-1]))

            s = time()
            path_img = self._plot_paths(paths, intersection_paths, intersection_path_id,
                                        crossing_orders, path_radii, intersection_color)
            times.append(time() - s)
            print("Plotting time: {:.5f}".format(times[-1]))

        print("Total time: {:.5f}".format(sum(times)))

        if plot:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(self.image)
            ax[1].imshow(path_img)
            plt.tight_layout()
            if save_fig:
                plt.savefig("img{}.png".format(save_id), dpi=300)
            else:
                plt.show()

        return paths
