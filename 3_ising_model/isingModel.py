"""
From Schroeder section 8.2
Using Metropolis algorithm

temperature in units of e/k (energy per boltzman constant)
unitless because the boltzman factors are e^(energy/temp) and there is no boltzman constant in there
"""

import numpy as np
import imageio.v3 as iio
import os
import matplotlib.pyplot as plt
import time


def deltaU(s, i, j, size):
    """
    Find the difference in energy if the site (i,j) was flipped

    Difference in energy is twice the energy of the site times the energy of its neighbor,
    and sum over all 4 neighbors (not including diagonal neighbors)

    Enforce periodic boundary conditions. Left side goes to right side, down side goes to up side
    Really means the shape of spins are on the surface of a torus

    Args:
    `s`: the spin 2d matrix
    `i`: x index
    `j`: y index
    """
    if i == 0:
        top = s[size - 1, j]
    else:
        top = s[i - 1, j]
    if i == size - 1:
        bottom = s[0, j]
    else:
        bottom = s[i + 1, j]
    if j == 0:
        left = s[i, size - 1]
    else:
        left = s[i, j - 1]
    if j == size - 1:
        right = s[i, 0]
    else:
        right = s[i, j + 1]

    return 2 * s[i, j] * (top + bottom + left + right)


def initRandom(s, size):
    """
    Initialize the grid to a random collection of spin up and spin down
    """
    for i in range(size):
        for j in range(size):
            if np.random.random() < 0.5:
                s[i, j] = 1
            else:
                s[i, j] = -1
    return s


def initMagnetized(s, state: int, size):
    """
    Initialize the grid to a magnetized state, works better for lower temperature simulations

    Args:
    `state`: 1 or -1
    """
    for i in range(size):
        for j in range(size):
            s[i, j] = state
    return s


def calcMagnetization(s: np.ndarray, size):
    sum = 0
    for i in range(size):
        for j in range(size):
            sum += s[i, j]
    return sum


def calcCorrelation(s: np.ndarray):
    """ """
    dim_x = s.shape[0]
    dim_y = s.shape[1]

    correlation_x_per_r = np.zeros((dim_x // 2, dim_x * dim_y))
    correlation_y_per_r = np.zeros((dim_y // 2, dim_x * dim_y))

    # calculate x correlation
    for r in range(dim_x // 2):
        for j in range(dim_y):
            for i in range(dim_x):
                this_site = i
                other_site = (i + r) % dim_x
                product = s[this_site, j] * s[other_site, j]
                offset = s[this_site, j] ** 2

                correlation_x_per_r[r, dim_y * j + i] = product - offset

    avg_corr_x = correlation_x_per_r.mean(axis=1)

    # calculate y correlation
    for r in range(dim_y // 2):
        for i in range(dim_x):
            for j in range(dim_y):
                this_site = j
                other_site = (j + r) % dim_y
                product = s[i, this_site] * s[i, other_site]
                offset = s[i, this_site] ** 2

                correlation_y_per_r[r, dim_x * i + j] = product - offset

    avg_corr_y = correlation_y_per_r.mean(axis=1)

    return avg_corr_x, avg_corr_y


def normal_sim(
    size,
    temp,
    iterations=400,
    create_gif=False,
    show_correlation=False,
    correlation_step=50):
    
    s = np.empty((size, size))
    # s = initializeMagnetized(s, state=1)
    s = initRandom(s, size)
    cwd = os.getcwd()
    folder_name = f"{cwd}/ising_model_states"
    os.makedirs(folder_name, exist_ok=True)

    fig, ax = plt.subplots()

    startTime = time.time()

    magnetization_over_time = []
    magnetization_bins = []

    print("Starting...")
    for state_iter in range(iterations):
        print(f"Calculating state {state_iter}")
        for site_iter in range(size**2):
            # choose a random site
            i = int(np.random.random() * size)
            j = int(np.random.random() * size)
            # calculate energy difference
            Ediff = deltaU(s, i, j, size)
            # if flipping reduces the energy, flip it
            if Ediff <= 0:
                s[i, j] = -s[i, j]
            # if not, then flip it randomly, where the cutoff is that the number needs to be less than the Boltzman factor
            elif np.random.random() < np.exp(-Ediff / temp):
                s[i, j] = -s[i, j]

        if show_correlation:
            if state_iter % correlation_step == 0:
                corr_x, corr_y = calcCorrelation(s)
                r = [i + 1 for i in range(size // 2)]
                plt.plot(r, corr_x, marker=".", color="orange", label="X")
                plt.xlabel("r")
                plt.ylabel("Correlation")
                plt.plot(r, corr_y, marker=".", color="royalblue", label="Y")
                plt.legend()
                plt.show()

        if create_gif:
            plt.imsave(f"{folder_name}/state_{state_iter}.png", s, cmap="Blues")

        m = calcMagnetization(s, size)
        magnetization_over_time.append(m)

        magnetization_bins.append(m)

    print(f"Total time: {time.time() - startTime:.2f} s")

    if create_gif:
        images = []
        for state_iter in range(iterations):
            images.append(iio.imread(f"{folder_name}/state_{state_iter}.png"))

        iio.imwrite(f"{cwd}/ising_model_T_{temp}.gif", images, format="gif", fps=30)

    plt.plot(magnetization_over_time)
    plt.xlabel("Iteration")
    plt.ylabel("Magnetization")
    plt.show()

    plt.hist(magnetization_bins, bins=int(iterations / 10))
    plt.show()


def correlation_vs_temp_sim(size, temp_list):
    """
    Plot the correlation function at several temperatures
    Waits for the system to equilibriate before calculating the correlation function

    """
    s = np.empty((size, size))
    # s = initializeMagnetized(s, state=1)
    s = initRandom(s, size)
    cwd = os.getcwd()
    folder_name = f"{cwd}\\ising_model_states"
    os.makedirs(folder_name, exist_ok=True)

    fig, ax = plt.subplots()

    startTime = time.time()

    corr_x_vs_T = []
    corr_y_vs_T = []

    print("Starting...")
    for T in temp_list:
        print(f"Calculating T={T}K")
        for state_iter in range(500):
            for site_iter in range(size**2):
                # choose a random site
                i = int(np.random.random() * size)
                j = int(np.random.random() * size)
                # calculate energy difference
                Ediff = deltaU(s, i, j, size=size)
                # if flipping reduces the energy, flip it
                if Ediff <= 0:
                    s[i, j] = -s[i, j]
                # if not, then flip it randomly, where the cutoff is that the number needs to be less than the Boltzman factor
                elif np.random.random() < np.exp(-Ediff / T):
                    s[i, j] = -s[i, j]

        corr_x, corr_y = calcCorrelation(s)
        avg_corr_x = corr_x.mean()
        avg_corr_y = corr_y.mean()

        corr_x_vs_T.append(avg_corr_x)
        corr_y_vs_T.append(avg_corr_y)

    print(f"Total time: {time.time() - startTime:.2f} s")

    plt.plot(temp_list, corr_x_vs_T, marker=".", color="magenta", label="X")
    plt.plot(temp_list, corr_y_vs_T, marker=".", color="royalblue", label="Y")
    plt.xlabel("Temperature [K]")
    plt.ylabel("Correlation")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    normal_sim(size=400, temp=10, iterations=500, create_gif=True)

    temps = np.logspace(0, 1, 100)
    correlation_vs_temp_sim(size=50, temp_list=temps)
