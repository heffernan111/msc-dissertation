# MSc Project

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
run_dir = new_run_dir(paths, label="week_2")
```
