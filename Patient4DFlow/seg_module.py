import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import splev, splprep
from skimage.morphology import skeletonize


# goal is to essentially create a GREEDY traveling salesman...
# absolutely no idea what this nonsense is.
# might consider just fitting a line directly to the data... locally-weighted regression? Not sure...
def create_distance_matrix(points):
    return np.sqrt(((points[:, :, None] - points[:, :, None].T) ** 2).sum(axis=1))


def gen_orthogonal_vectors(v: np.ndarray) -> tuple:
    """Calculate a vector orthogonal to the given vector v.

    Args:
        v (np.ndarray): Input vector

    Returns:
        tuple: Orthogonal vectors to form local basis

    """
    # Choose a second vector that is not parallel to v
    second_vector = np.array([0, 1, 0]) if np.all(v == [1, 0, 0]) else np.array([1, 0, 0])

    # Calculate the cross product
    orthogonal_vec = np.cross(v, second_vector)

    # Normalize the orthogonal vector
    orthogonal_vec = orthogonal_vec / np.linalg.norm(orthogonal_vec)

    orthogonal_vec2 = np.cross(orthogonal_vec, v)

    # Normalize the orthogonal vector
    orthogonal_vec2 = orthogonal_vec2 / np.linalg.norm(orthogonal_vec2)

    return orthogonal_vec, orthogonal_vec2


# VECTORIZE AND FIND ALL??
# Could also attempt to defind some kind of polar coordinate system along centerline???
# trouble is that I'll have to then transform to cartesian coordinates...
def find_planes(point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Generate a plane based on a point and its normal vector.

    Args:
        points (np.ndarray): position vector of the plane center
        normals (np.ndarray): normal vector of the plane

    Returns:
        np.ndarray: _description_

    """
    # ensure that normal vector is actually normalized
    normal = normal / np.linalg.norm(normal)
    u, v = gen_orthogonal_vectors(normal)

    u_full = np.outer(np.linspace(-15, 15, 50), u)
    v_full = np.outer(np.linspace(-15, 15, 50), v)
    # uv, vv = np.meshgrid(u, v, indexing="ij")

    u_grid = np.zeros((50 * 50, 3))
    for i in range(50):
        u_grid[i * 50 : (i + 1) * 50, :] = u_full + v * (i - 25) / 2

    return u_grid + point


# NOTE: this code has a few weird redundant steps that I can clean up later...
def greedy_tsp(cost_matrix: np.ndarray, start_idx: int = 0) -> list:
    """Return the path with the lowest cost using a greedy algorithm.

    Function that takes inspiration from the classic traveling salesman problem, but implemented
    with an extremely greedy method. An exact solution is far from guaranteed; the outcome is heavily
    dependent on the starting index. Essentially, each step in the path is determined by finding
    whichever adjacent point is the closest.

    Args:
        cost_matrix (np.ndarray): 2D array/matrix that stores the cost (generally some measure of distance)
        of moving from one point to another.
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


def smooth_skeletonize(segmentation: np.ndarray) -> tuple:
    # scikit-image will automatically downcast, but doing it explicitly will save computation time
    skel = skeletonize(segmentation.astype(np.uint8))
    points = np.array(np.nonzero(skel)).T

    distance_matrix = create_distance_matrix(points)
    start = np.argmax(points[:, 0])
    best_path = greedy_tsp(distance_matrix, start)
    points = points[best_path]

    tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]])
    new_points = splev(u, tck)
    first_deriv = splev(u, tck, der=1)

    return skel, points, new_points, first_deriv


def plane_drawer(segmentation, spline_points, spline_deriv):
    xv = np.arange(0, segmentation.shape[0], 1)
    yv = np.arange(0, segmentation.shape[1], 1)
    zv = np.arange(0, segmentation.shape[2], 1)

    X, Y, Z = np.meshgrid(xv, yv, zv, indexing="ij")

    fig = go.Figure()

    fig.add_trace(
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=segmentation.flatten(),
            isomin=0.9,
            isomax=1.0,
            opacity=0.1,  # needs to be small to see through all surfaces
            surface_count=1,  # needs to be a large number for good volume rendering
            showscale=False,
        ),
    )

    fig.add_trace(
        go.Scatter3d(
            x=spline_points[0],
            y=spline_points[1],
            z=spline_points[2],
            marker={
                "size": 4,
                "colorscale": "Viridis",
            },
            line={"color": "red", "width": 2},
        ),
    )

    for i in range(len(spline_points[0])):
        center_point = np.array(
            [spline_points[0][i], spline_points[1][i], spline_points[2][i]],
        )

        plane = find_planes(
            np.array([spline_points[0][i], spline_points[1][i], spline_points[2][i]]),
            np.array([spline_deriv[0][i], spline_deriv[1][i], spline_deriv[2][i]]),
        )

        plane_vol_idx = np.round(plane).astype(int)
        plane_vol = np.zeros(segmentation.shape)

        plane_vol_idx = plane_vol_idx[
            np.linalg.norm(plane_vol_idx - center_point, axis=1) < 10,
            :,
        ]

        for j in range(len(plane_vol_idx)):
            plane_vol[plane_vol_idx[j, 0], plane_vol_idx[j, 1], plane_vol_idx[j, 2]] = 1

        plane_vol = plane_vol * segmentation

        # inverse to scatter for better plotting
        plane_vol_idx = np.nonzero(plane_vol)
        plane = np.array(plane_vol_idx).T

        fig.add_trace(
            go.Scatter3d(
                visible=False,
                x=plane[:, 0],
                y=plane[:, 1],
                z=plane[:, 2],
                marker={"size": 4, "color": "blue"},
            ),
        )

    # Make 12th trace visible
    fig.data[12].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},  # NOTE: THIS ISDEFINITELY WRONG
                {"title": "Slider switched to plane: " + str(i)},
            ],  # layout attribute
        )

        # step["args"][0]["visible"][0] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][1] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [
        {
            "active": 10,
            "currentvalue": {"prefix": "Plane: "},
            "pad": {"t": 50},
            "steps": steps,
        },
    ]
    fig.update_layout(
        scene={
            "aspectmode": "data",
        },
        sliders=sliders,
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,
    )

    # fig.layout.yaxis.scaleanchor = "x"
    # fig.layout.zaxis.scaleanchor = "x"

    fig.show()
