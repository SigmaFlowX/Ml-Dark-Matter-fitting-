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

def generate_dataset_deepsets(n_stars = 800, r_grid = np.logspace(-2, 1, 120), save_path = "train.npz"):

    param_list = generate_parameter_grid()

    R_all = []
    vlos_all = []
    params_all = []

    for params in tqdm(param_list):

        R, v_los = create_data(r_grid = r_grid, n_stars = n_stars, **params)
        params_vector = np.array([

            params["rho_s"],
            params["r_s"],
            params["a"],
            params["beta_inf"],
            params["r_beta"]

        ])

        R_all.append(R)
        vlos_all.append(v_los)
        params_all.append(params_vector)

    np.savez_compressed(
        save_path,
        R = np.array(R_all),
        vlos = np.array(vlos_all),
        params = np.array(params_all)
    )

    print("N galaxies =", len(R_all))

if __name__ == "__main__":
    generate_dataset()