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
