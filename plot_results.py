import matplotlib.pyplot as plt
import numpy as np


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


def main():
    print("This isn't a script, but feel free to debug/run tests here!")


if __name__ == "__main__":
    main()
