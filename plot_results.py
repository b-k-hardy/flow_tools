import matplotlib.pyplot as plt
import numpy as np


# FIXME: add various additional arguments for title, labels, etc.
def plot_dp(time: np.array, dp: np.array, id: str) -> plt.figure:
    fig, ax = plt.subplots()

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$Delta p$ [mmHg]")
    ax.set_title(f"{id} Pressure Drop")

    ax.grid()
    fig.tight_layout()
    ax.plot(time, dp)

    plt.show()

    return fig


def main():
    print("hello")


if __name__ == "__main__":
    main()
