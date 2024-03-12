import numpy as np
from python_tsp.distances import euclidean_distance_matrix
from scipy.interpolate import splev, splprep
from skimage.morphology import skeletonize_3d

from plot_results import plot_seg_skeleton
from read_dicoms import import_segmentation


# goal is to essentially create a GREEDY traveling salesman...
def create_distance_matrix(points):

    return -1


# NOTE: this code has a few weird redundant steps that I can clean up later...
def greedy_tsp(cost_matrix: np.ndarray, start_idx: int = 0) -> list:
    """Function that takes inspiration from the classic traveling salesman problem, but implemented with an extremely greedy method.
    An exact solution is far from guaranteed; the outcome is heavily dependent on the starting index. Essentially, each step in the path
    is determined by finding whichever adjacent point is the closest.

    Args:
        cost_matrix (np.ndarray): 2D array/matrix that stores the cost (generally some measure of distance) of moving from one point to another.
        start_idx (int, optional): Index of the point to start with. Defaults to 0.

    Returns:
        list: Returns the path with the lowest cost.
    """

    path = []
    path.append(start_idx)
    N = cost_matrix.shape[0]

    for i in range(N):
        cost_matrix[i, i] = np.nan

    valid_idx = set(np.arange(N))
    valid_idx.remove(start_idx)
    cost_matrix[:, start_idx] = np.nan

    position = start_idx
    while valid_idx:
        step = np.nanargmin(cost_matrix[position, :])
        path.append(step)
        valid_idx.remove(step)
        cost_matrix[:, step] = np.nan
        position = step

    return path


def smooth_skeletonize(segmentation):

    skel = skeletonize_3d(
        segmentation.astype(np.uint8)
    )  # scikit-image will automatically downcast; doing it explicitly will save computation time
    points = np.array(np.nonzero(skel)).T

    distance_matrix = euclidean_distance_matrix(points)
    start = np.argmax(points[:, 0])
    best_path = greedy_tsp(distance_matrix, start)
    points = points[best_path]

    tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]])
    new_points = splev(u, tck)

    return skel, points, new_points


def main():
    print("This isn't a script, but feel free to debug/run tests here!")


if __name__ == "__main__":
    main()
