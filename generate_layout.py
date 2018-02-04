import numpy as np
import json

def generate_coordinates(rows, cols):
    x = np.arange(cols)
    x_only = np.tile(x, (rows, 1))

    y = np.arange(rows)
    y_only = np.tile(y, (cols, 1))
    grid = np.stack([x_only, np.transpose(y_only), np.zeros((rows, cols))], axis=-1)
    grid = grid.reshape((-1, 3))
    return grid

def additions(grid):
    additions = [[-1, -2, -3, -1, -2, -3, -1.5, -2.5, -1, -2, -3],
                 [0,0,0,1,1,1, 2,2, 3,3,3],
                 [0]*11]

    return np.append(grid, np.transpose(additions),
                     axis=0)

def generate_layout(filename, rows, cols):
    with open(filename, "w") as f:
        f.write(json.dumps({"positions":
                            additions(generate_coordinates(rows,
                                                           cols)).tolist()}))

if __name__ == "__main__":
    generate_layout("layout.json", 30, 40)
