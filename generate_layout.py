import json
from generate_layout import add_additions, generate_coordinates


def write_layout(filename, rows, cols):
    with open(filename, "w") as f:
        f.write(json.dumps({"positions":
                            add_additions(generate_coordinates(rows,
                                                           cols)).tolist()}))

if __name__ == "__main__":
    write_layout("layout.json", 30, 40)
