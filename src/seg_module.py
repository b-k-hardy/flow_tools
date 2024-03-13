import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import splev, splprep
from skimage.morphology import skeletonize_3d


# goal is to essentially create a GREEDY traveling salesman...
# absolutely no idea what this nonsense is.
# might consider just fitting a line directly to the data... locally-weighted regression? Not sure...
def create_distance_matrix(points):
    return np.sqrt(((points[:, :, None] - points[:, :, None].T) ** 2).sum(axis=1))


# VECTORIZE AND FIND ALL??
# Could also attempt to defind some kind of polar coordinate system along centerline???
# trouble is that I'll have to then transform to cartesian coordinates...
def find_planes(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        points (np.ndarray): _description_
        normals (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    normals = normals / np.linalg.norm(normals)
    d = np.dot(normals, points)

    const = d / normals[0]
    param_1 = -normals[1] / normals[0]
    param_2 = -normals[2] / normals[0]

    u = np.linspace(50, 100, 50)  # FUCK IT ACTUALLY HAS TO BE A MESH GRID SITUATION
    v = np.linspace(0, 50, 50)

    uv, vv = np.meshgrid(u, v)

    plane = (
        np.array([const, 0.0, 0.0])
        + np.outer(uv, np.array([param_1, 1.0, 0.0]))
        + np.outer(vv, np.array([param_2, 0.0, 1.0]))
    )

    return plane


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
            surface_count=2,  # needs to be a large number for good volume rendering
            showscale=False,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=spline_points[0],
            y=spline_points[1],
            z=spline_points[2],
            marker=dict(
                size=4,
                colorscale="Viridis",
            ),
            line=dict(color="red", width=2),
        )
    )

    for i in range(len(spline_points[0])):
        plane = find_planes(
            np.array([spline_points[0][i], spline_points[1][i], spline_points[2][i]]),
            np.array([spline_deriv[0][i], spline_deriv[1][i], spline_deriv[2][i]]),
        )

        fig.add_trace(
            go.Scatter3d(
                visible=False,
                x=plane[:, 0],
                y=plane[:, 1],
                z=plane[:, 2],
                marker=dict(
                    size=4,
                    colorscale="Viridis",
                ),
                line=dict(color="blue", width=2),
            )
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
                {"title": "Slider switched to step: " + str(i)},
            ],  # layout attribute
        )

        step["args"][0]["visible"][0] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][1] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [
        dict(
            active=10,
            currentvalue={"prefix": "Frequency: "},
            pad={"t": 50},
            steps=steps,
        )
    ]

    fig.update_layout(sliders=sliders, xaxis_fixedrange=True, yaxis_fixedrange=True)

    fig.show()


def main():
    print("This isn't a script, but feel free to debug/run tests here!")


if __name__ == "__main__":
    main()
