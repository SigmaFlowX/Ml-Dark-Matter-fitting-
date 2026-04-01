from create_synthetic_data import create_data
import numpy as np
from tqdm import tqdm
from itertools import product

def generate_parameter_grid():
    grid = {
        "rho_s": np.logspace(6, 8, 6),
        "r_s": np.linspace(0.5, 5, 6),
        "a": np.linspace(0.3, 1.5, 5),
        "beta_inf": np.linspace(-0.2, 0.5, 6),
        "r_beta": np.linspace(0.5, 5, 6),
        "m0": [5e6]
    }

    keys = list(grid.keys())
    combinations = list(product(*grid.values()))

    param_list = [
        dict(zip(keys, values))
        for values in combinations
    ]

    return param_list

def generate_dataset(n_stars = 5000, r_grid = np.logspace(-2, 1, 120), save_path = "train.npz"):

    param_list = generate_parameter_grid()
    dataset = []

    for params in tqdm(param_list):

        R, v_los = create_data(r_grid = r_grid, n_stars = n_stars, **params)
        galaxy = {
            "R": R,
            "vlos": v_los,
            **params
        }
        dataset.append(galaxy)

    np.savez_compressed(save_path, dataset = dataset)
    print("N galaxies =", len(dataset))

if __name__ == "__main__":
    generate_dataset()