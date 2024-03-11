import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def plot_dp(time: np.ndarray, dp: np.ndarray, id: str) -> plt.Figure:
    """_summary_

    Args:
        time (np.ndarray): Pressure estimation timepoints
        dp (np.ndarray): Estimated relative pressure curve
        id (str): Name of input dataset

    Returns:
        plt.Figure: Pressure trace between two planes over cardiac cycle
    """

    fig, ax = plt.subplots()

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$\Delta$p [mmHg]")
    ax.set_title(f"{id} Pressure Drop")

    ax.grid()
    fig.tight_layout()
    ax.plot(time, dp)

    return fig


def plot_seg(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    segmentation,
    skeleton,
    skeleton_points,
    spline,
):

    X, Y, Z = grid

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
        )
    )

    fig.add_trace(
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=skeleton.flatten(),
            isomin=0.9,
            isomax=1.0,
            opacity=0.1,  # needs to be small to see through all surfaces
            surface_count=2,  # needs to be a large number for good volume rendering
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=skeleton_points[:, 0],
            y=skeleton_points[:, 1],
            z=skeleton_points[:, 2],
            marker=dict(
                size=4,
                colorscale="Viridis",
            ),
            line=dict(color="darkblue", width=2),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=spline[0],
            y=spline[1],
            z=spline[2],
            marker=dict(
                size=4,
                colorscale="Viridis",
            ),
            line=dict(color="red", width=2),
        )
    )

    fig.show()

    return None


def main():
    print("This isn't a script, but feel free to debug/run tests here!")


if __name__ == "__main__":
    main()
