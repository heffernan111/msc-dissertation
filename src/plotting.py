from __future__ import annotations

from pathlib import Path


from datetime import datetime


def savefig(fig, run_dir: Path, name: str, add_date: bool = True, dpi: int = 600) -> Path:
    """
    Save a matplotlib figure to runs/<script_name>/figures/<name>.png and .pdf.
    
    Args:
        fig: Matplotlib figure object
        run_dir: Run directory from new_run_dir() (figures will be saved to run_dir/figures/)
        name: Base name for the figure (spaces will be replaced with underscores)
        add_date: Whether to prepend the current date (YYYY-MM-DD) to the filename
        dpi: Resolution for PNG output (default: 600)
    
    Returns:
        Path to the saved PNG file
    """
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    stem = name.replace(" ", "_")
    if add_date:
        today = datetime.now().strftime("%Y-%m-%d")
        stem = f"{today}_{stem}"
        
    png_path = figures_dir / f"{stem}.png"
    pdf_path = figures_dir / f"{stem}.pdf"

    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path

# To be used for many light curves. TODO: write a function to send all light curves to this function
def plot_light_curve(df, run_dir: Path, title: str = "Light Curve", filename: str | None = None) -> Path | None:
    """
    Plot a light curve (MJD vs flux) with error bars, grouped by filter.
    
    Args:
        df: DataFrame containing 'MJD', 'forced_ujy', 'forced_ujy_error', and 'filter' columns
        run_dir: Run directory to save the figure
        title: Title of the plot
        filename: Filename to save (if None, derived from title)
        
    Returns:
        Path to saved figure or None if plotting failed
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Ensure numeric types
    df = df.copy()
    for col in ['MJD', 'forced_ujy', 'forced_ujy_error']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop invalid rows for plotting
    plot_data = df.dropna(subset=['MJD', 'forced_ujy', 'filter'])
    
    if plot_data.empty:
        print("No valid data to plot.")
        return None
        
    plot_data = plot_data.sort_values(by='MJD')
    
    # Standard ZTF filter colors
    filter_colors = {'g': 'green', 'r': 'red', 'i': 'orange'}
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for filt, group in plot_data.groupby('filter'):
        # Handle cases where error might be missing or all NaN
        yerr = group['forced_ujy_error'] if 'forced_ujy_error' in group.columns else None
        
        ax.errorbar(
            group['MJD'], 
            group['forced_ujy'], 
            yerr=yerr, 
            fmt='o', 
            label=f'Filter {filt}', 
            color=filter_colors.get(filt, 'blue'), 
            alpha=0.7,
            markersize=4,
            capsize=2
        )

    ax.set_xlabel('MJD')
    ax.set_ylabel('Forced Flux (uJy)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if filename is None:
        filename = title.lower()
        
    return savefig(fig, run_dir, filename)
