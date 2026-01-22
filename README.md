# PhD Project

Jupyter notebook setup for data analysis. Outputs go to `outputs/runs/<notebook_name>/`.

## Project Structure

```
msc/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── runs/
│       └── <notebook_name>/
│           ├── data/
│           ├── figures/
│           └── tables/
└── src/
    ├── project_setup.py
    └── plotting.py
```

## Setup (copy this into each new notebook)

```python
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path.cwd().resolve().parent / "src"))

from project_setup import get_project_root, make_paths, new_run_dir, save_data
from plotting import savefig

root = get_project_root()
paths = make_paths(root)
run_dir = new_run_dir(paths, label="week_2")  # change to your notebook name
```

## Saving Figures

```python
fig, ax = plt.subplots()
ax.scatter([1, 2, 3], [2, 1, 4])
savefig(fig, run_dir, "scatter_plot")
# Saves to: runs/<notebook_name>/figures/scatter_plot.png and .pdf
```

## Saving Data

```python
import pandas as pd
import numpy as np

# CSV
save_data(df, run_dir, "results.csv")

# Parquet
save_data(df, run_dir, "results.parquet")

# NumPy array
save_data(arr, run_dir, "array.npy")
```

## Functions
**`save_data(data, run_dir, filename, **kwargs)`**  
Saves to `runs/<script_name>/data/<filename>`. Format determined by extension.

**`savefig(fig, run_dir, name, dpi=200)`**  
Saves to `runs/<script_name>/figures/<name>.png` and `.pdf`.
