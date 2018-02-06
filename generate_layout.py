import json
import numpy as np

# These methods need to be copied from thimble_brain.py
# Due to the restricted execution environment of the brain file
# where imports don't work from local packages, and
# building the circuit can not be in a if __name__ == "__main__" block.
# This causes a crash. since the simulation package is not available
# in the default python install.


def generate_coordinates(rows, cols):
    x = np.arange(cols)
    x_only = np.tile(x, (rows, 1))

    y = np.arange(rows)
    y_only = np.tile(y, (cols, 1))
    grid = np.stack([x_only, np.transpose(y_only), np.zeros((rows, cols))], axis=-1)
    grid = grid.reshape((-1, 3))
    return grid[::-1, :]


def add_additions(grid, additions):

    return np.append(grid, np.transpose(additions),
                     axis=0)

def write_layout(filename, rows, cols):
    with open(filename, "w") as f:
        f.write(json.dumps({"positions":
                            add_additions(generate_coordinates(rows,
                                                           cols),
                                          additions=[[-1, -2, -3, -1, -2, -3, -1.5, -2.5, 10, 20, 30],
                                                     [0, 0, 0, 1, 1, 1, 2, 2, 15, 15, 15],
                                                     [0, 0, 0, 0, 0, 0, 0, 0, -5, -5, -5]]).tolist()}))

if __name__ == "__main__":
    write_layout("layout.json", 30, 40)
