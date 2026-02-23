import json

NB_PATH = "notebooks/exp01_chunking_optimization.ipynb"

def list_headers():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print("Current Notebook Structure:")
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            lines = source.split("\n")
            for line in lines:
                if line.strip().startswith("#"):
                    print(f"Cell {i}: {line.strip()}")

if __name__ == "__main__":
    list_headers()
