from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    data_external: Path
    outputs: Path
    figures: Path
    tables: Path
    runs: Path


def get_project_root(marker_file: str = "README.md") -> Path:
    """Find the project root directory by looking for a marker file."""
    here = Path.cwd().resolve()
    for parent in [here, *here.parents]:
        if (parent / marker_file).exists():
            return parent
    raise FileNotFoundError(
        f"Could not find project root (looked for '{marker_file}' in parents of {here})"
    )


def get_notebook_name() -> str | None:
    """Try to get the current notebook name."""
    try:
        ipython = get_ipython()  # type: ignore
        if ipython and hasattr(ipython, "kernel"):
            # Get the notebook name from IPython
            if hasattr(ipython, "user_ns") and "__file__" in ipython.user_ns:
                notebook_path = Path(ipython.user_ns["__file__"])
                return notebook_path.stem
            # Alternative: try to get from kernel info
            if hasattr(ipython, "kernel") and hasattr(ipython.kernel, "session"):
                # This might work in some Jupyter setups
                pass
    except NameError:
        pass
    return None


def make_paths(root: Path) -> ProjectPaths:
    """Create and return all project paths."""
    data_raw = root / "data" / "raw"
    data_interim = root / "data" / "interim"
    data_processed = root / "data" / "processed"
    data_external = root / "data" / "external"
    
    outputs = root / "outputs"
    figures = outputs / "figures"
    tables = outputs / "tables"
    runs = outputs / "runs"

    for p in [data_raw, data_interim, data_processed, data_external, figures, tables, runs]:
        p.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        root=root,
        data_raw=data_raw,
        data_interim=data_interim,
        data_processed=data_processed,
        data_external=data_external,
        outputs=outputs,
        figures=figures,
        tables=tables,
        runs=runs,
    )


def new_run_dir(paths: ProjectPaths, label: str | None = None) -> Path:
    """
    Create a new run directory based on notebook name.
    
    The directory structure will be: runs/<notebook_name>/data/
    
    Args:
        paths: ProjectPaths object
        label: If None, tries to auto-detect notebook name.
               If auto-detection fails, uses 'run' as default.
    Returns:
        Path to the run directory (runs/<notebook_name>/)
    """
    if label is None:
        notebook_name = get_notebook_name()
        if notebook_name:
            label = notebook_name
        else:
            try:
                import __main__
                if hasattr(__main__, "__file__"):
                    label = Path(__main__.__file__).stem
                else:
                    label = "run"
            except:
                label = "run"
    
    safe_label = label.strip().replace(" ", "_").replace(".ipynb", "")

    # runs/<notebook_name>/
    run_dir = paths.runs / safe_label

    # Standard subfolders for each run
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    return run_dir


def save_data(data, run_dir: Path, filename: str, add_date: bool = True, **kwargs) -> Path:
    """
    Save data to runs/<notebook_name>/data/<filename>.
    
    Args:
        data: Data to save
        run_dir: Run directory from new_run_dir()
        filename: Filename
        add_date: Whether to prepend the current date (YYYY-MM-DD) to the filename

    Returns:
        Path to saved file
    """
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if add_date:
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"{today}_{filename}"
    
    filepath = data_dir / filename
    
    # Handle different data types based on extension
    ext = filepath.suffix.lower()
    
    if ext == ".csv":
        import pandas as pd
        if not isinstance(data, pd.DataFrame):
            raise TypeError("CSV format requires pandas DataFrame")
        data.to_csv(filepath, **kwargs)
    elif ext in [".parquet", ".pq"]:
        import pandas as pd
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Parquet format requires pandas DataFrame")
        data.to_parquet(filepath, **kwargs)
    elif ext == ".npy":
        import numpy as np
        np.save(filepath, data, **kwargs)
    else:
        # Default to csv
        import pandas as pd
        if not isinstance(data, pd.DataFrame):
            raise TypeError("CSV format requires pandas DataFrame")
        data.to_csv(filepath, **kwargs)
    
    return filepath

