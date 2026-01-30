from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os
import requests
import pandas as pd
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import lasair


def ra_to_degrees(ra):
    arr = ra.split(':')
    hours = float(arr[0])
    mins = float(arr[1])
    secs = float(arr[2])
    
    return hours * 15 + mins / 4 + secs / 240

def dec_to_degrees(dec):
    sign = 1 if dec[0] == '+' else -1
    arr = dec[1:].split(':')
    degrees = float(arr[0])
    mins = float(arr[1])
    secs = float(arr[2])
    
    return sign * (degrees + mins / 60 + secs / 3600)

def mag_to_flux(mag, magerr, zp=25.0):
    flux = 10**(-0.4*(mag - zp))
    fluxerr = (np.log(10)/2.5) * flux * magerr
    return flux, fluxerr


def save_data(data, run_dir: Path, filename: str, add_date: bool = True, **kwargs) -> Path:
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if add_date:
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"{today}_{filename}"
    
    filepath = data_dir / filename

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

def _resolve_run_name(run_name: Optional[str]) -> str:
    return (run_name or '').strip() or datetime.now().strftime('%Y-%m-%d')


def update_tracker(ztf_id: str, lasair_status: str | None = None, lasair_path: Path | None = None,
                   tns_status: str | None = None, tns_path: Path | None = None,
                   sncosmo_status: str | None = None, sncosmo_path: Path | None = None,
                   project_root: Optional[Path] = None, run_name: Optional[str] = None) -> None:

    if project_root is None:
        project_root = Path(__file__).parent.parent
    run_name = _resolve_run_name(run_name)
    tracker_path = project_root / 'data' / run_name / 'tracker.csv'
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    
    lasair_path_str = str(lasair_path.relative_to(project_root)) if lasair_path else ''
    tns_path_str = str(tns_path.relative_to(project_root)) if tns_path else ''
    sncosmo_path_str = str(sncosmo_path.relative_to(project_root)) if sncosmo_path else ''
    
    # Read existing tracker or create new
    required_columns = ['ztf_id', 'lasair_status', 'lasair_path', 'tns_status', 'tns_path', 'sncosmo_status', 'sncosmo_path', 'download_date']
    
    if tracker_path.exists() and tracker_path.stat().st_size > 0:
        try:
            tracker_df = pd.read_csv(tracker_path)
            for col in required_columns:
                if col not in tracker_df.columns:
                    tracker_df[col] = ''
                # Convert to object type to allow string values
                if tracker_df[col].dtype != 'object':
                    tracker_df[col] = tracker_df[col].astype('object')
            
            # Check if this ZTF ID already exists
            if not tracker_df.empty and ztf_id in tracker_df['ztf_id'].values:
                # Update existing row
                idx = tracker_df.index[tracker_df['ztf_id'] == ztf_id][0]
                if lasair_status is not None:
                    tracker_df.at[idx, 'lasair_status'] = lasair_status
                if lasair_path_str:
                    tracker_df.at[idx, 'lasair_path'] = lasair_path_str
                if tns_status is not None:
                    tracker_df.at[idx, 'tns_status'] = tns_status
                if tns_path_str:
                    tracker_df.at[idx, 'tns_path'] = tns_path_str
                if sncosmo_status is not None:
                    tracker_df.at[idx, 'sncosmo_status'] = sncosmo_status
                if sncosmo_path_str:
                    tracker_df.at[idx, 'sncosmo_path'] = sncosmo_path_str
                tracker_df.at[idx, 'download_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            else:
                # Append new row
                tracker_entry = {
                    'ztf_id': ztf_id,
                    'lasair_status': lasair_status or '',
                    'lasair_path': lasair_path_str,
                    'tns_status': tns_status or '',
                    'tns_path': tns_path_str,
                    'sncosmo_status': sncosmo_status or '',
                    'sncosmo_path': sncosmo_path_str,
                    'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                tracker_df = pd.concat([tracker_df, pd.DataFrame([tracker_entry])], ignore_index=True)
        except (pd.errors.EmptyDataError, ValueError, KeyError):
            # If tracker file is corrupted or empty, start fresh
            tracker_entry = {
                'ztf_id': ztf_id,
                'lasair_status': lasair_status or '',
                'lasair_path': lasair_path_str,
                'tns_status': tns_status or '',
                'tns_path': tns_path_str,
                'sncosmo_status': sncosmo_status or '',
                'sncosmo_path': sncosmo_path_str,
                'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            tracker_df = pd.DataFrame([tracker_entry])
    else:
        # Create new tracker
        tracker_entry = {
            'ztf_id': ztf_id,
            'lasair_status': lasair_status or '',
            'lasair_path': lasair_path_str,
            'tns_status': tns_status or '',
            'tns_path': tns_path_str,
            'sncosmo_status': sncosmo_status or '',
            'sncosmo_path': sncosmo_path_str,
            'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        tracker_df = pd.DataFrame([tracker_entry])
    
    # Ensure all string columns are object type before saving
    for col in ['ztf_id', 'lasair_status', 'lasair_path', 'tns_status', 'tns_path', 'sncosmo_status', 'sncosmo_path', 'download_date']:
        if col in tracker_df.columns:
            tracker_df[col] = tracker_df[col].astype('object')
    
    tracker_df.to_csv(tracker_path, index=False)


def get_tracker_row(ztf_id: str, project_root: Optional[Path] = None, run_name: Optional[str] = None) -> Optional[dict]:
    if project_root is None:
        project_root = Path(__file__).parent.parent
    run_name = _resolve_run_name(run_name)
    tracker_path = project_root / 'data' / run_name / 'tracker.csv'
    if not tracker_path.exists() or tracker_path.stat().st_size == 0:
        return None
    try:
        tracker_df = pd.read_csv(tracker_path)
        row = tracker_df[tracker_df['ztf_id'] == ztf_id]
        if row.empty:
            return None
        r = row.iloc[0]
        lasair_path = None
        if pd.notna(r.get('lasair_path')) and str(r['lasair_path']).strip():
            p = project_root / str(r['lasair_path']).strip()
            lasair_path = p if p.exists() else None
        tns_path = None
        if pd.notna(r.get('tns_path')) and str(r['tns_path']).strip():
            p = project_root / str(r['tns_path']).strip()
            tns_path = p if p.exists() else None
        return {
            'lasair_status': r.get('lasair_status', ''),
            'tns_status': r.get('tns_status', ''),
            'lasair_path': lasair_path,
            'tns_path': tns_path,
        }
    except Exception:
        return None


def filter_objects_by_lasair_mag_error(
    objects_to_process: pd.DataFrame,
    project_root: Path,
    run_name: str,
    id_column: str = 'ZTFID',
) -> pd.DataFrame:
    run_name = _resolve_run_name(run_name)
    tracker_path = project_root / 'data' / run_name / 'tracker.csv'
    if not tracker_path.exists() or objects_to_process.empty:
        return objects_to_process
    tracker_df = pd.read_csv(tracker_path)
    keep_ids = []
    for _, row in objects_to_process.iterrows():
        ztf_id = row[id_column]
        tr = tracker_df[tracker_df['ztf_id'] == ztf_id]
        if tr.empty:
            continue
        lasair_path_str = tr.iloc[0].get('lasair_path')
        if not (pd.notna(lasair_path_str) and str(lasair_path_str).strip()):
            continue
        lasair_path = project_root / str(lasair_path_str).strip()
        if not lasair_path.exists():
            continue
        try:
            lc_df = load_lasair_lightcurve(lasair_path)
        except Exception:
            continue
        if 'unforced_mag_error' not in lc_df.columns:
            continue
        # Keep only if at least one non-null unforced_mag_error
        if lc_df['unforced_mag_error'].notna().any():
            keep_ids.append(ztf_id)
    return objects_to_process[objects_to_process[id_column].isin(keep_ids)].copy()


def savefig(fig, run_dir: Path, name: str, add_date: bool = True, dpi: int = 600) -> Path:
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    stem = name.replace(" ", "_")
    if add_date:
        today = datetime.now().strftime("%Y-%m-%d")
        stem = f"{today}_{stem}"
        
    png_path = figures_dir / f"{stem}.png"

    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    return png_path

def plot_light_curve(df, run_dir: Path, title: str = "Light Curve", filename: str | None = None) -> Path | None:
    # Validate required columns exist
    required_cols = ['MJD', 'unforced_mag', 'filter']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Ensure numeric types
    df = df.copy()
    for col in ['MJD', 'unforced_mag', 'unforced_mag_error']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop empty data rows
    plot_data = df.dropna(subset=['MJD', 'unforced_mag', 'filter']).copy()
    
    if plot_data.empty:
        print("No valid data to plot.")
        return None
    
    plot_data['filter'] = plot_data['filter'].astype(str).str.strip().str.lower()
    plot_data = plot_data.sort_values(by='MJD')
    
    # ZTF filter colors: g=green, r=red
    filter_colors = {'g': 'green', 'r': 'red'}
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for filt, group in plot_data.groupby('filter'):
        if group.empty:
            continue
        
        # Handle error bars
        yerr = group['unforced_mag_error'].replace([np.inf, -np.inf], np.nan)
        if yerr.isna().all():
            yerr = None
        
        color = filter_colors.get(filt, 'blue')
        ax.errorbar(
            group['MJD'], 
            group['unforced_mag'], 
            yerr=yerr, 
            fmt='o', 
            label=f'Filter {filt}', 
            color=color, 
            alpha=0.7,
            markersize=4,
            capsize=2
        )

    ax.set_xlabel('MJD')
    ax.set_ylabel('Unforced Magnitude')
    ax.set_title(title)
    ax.invert_yaxis()  # Magnitude scale: brighter = lower values
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if filename is None:
        filename = title.lower()
    
    primary_path = savefig(fig, run_dir, filename)

    # forced_ujy (flux in uJy)
    if 'forced_ujy' in plot_data.columns and plot_data['forced_ujy'].notna().any():
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        for filt, group in plot_data.groupby('filter'):
            if group.empty:
                continue
            x = group['MJD']
            y = pd.to_numeric(group.get('forced_ujy'), errors='coerce')
            yerr = pd.to_numeric(group.get('forced_ujy_error'), errors='coerce')
            if y.dropna().empty:
                continue
            color = filter_colors.get(filt, 'blue')
            # If no valid yerr, set to None
            yerr_plot = yerr.replace([np.inf, -np.inf], np.nan)
            if yerr_plot.isna().all():
                yerr_plot = None

            ax2.errorbar(
                x,
                y,
                yerr=yerr_plot,
                fmt='o',
                label=f'Filter {filt}',
                color=color,
                alpha=0.7,
                markersize=4,
                capsize=2
            )

        ax2.set_xlabel('MJD')
        ax2.set_ylabel('Forced flux (uJy)')
        ax2.set_title(f"Forced flux: {title}")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        forced_filename = (filename + "_forced_ujy") if filename else None
        savefig(fig2, run_dir, forced_filename)

    return primary_path


def get_lasair_token() -> Optional[str]:
    token = os.getenv('LASAIR_TOKEN')
    if token:
        return token.strip() or None

    project_root = Path(__file__).parent.parent
    token_file = project_root / '.lasair_token'
    if token_file.exists():
        try:
            return token_file.read_text().strip() or None
        except OSError as e:
            raise ValueError(
                f"Could not read Lasair token from {token_file}. "
                f"Check that the file is readable: {e}"
            ) from e

    return None

def download_lasair_csv(ztf_id: str, save_path: Optional[Path] = None, token: Optional[str] = None, project_root: Optional[Path] = None, run_name: Optional[str] = None) -> Path:
    try:
        if token is None:
            token = get_lasair_token()
        
        # Create Lasair client
        if token:
            client = lasair.lasair_client(token)
        else:
            raise ValueError(f"No token provided for Lasair API access. Set LASAIR_TOKEN.")
        
        # Get object data
        result = client.object(ztf_id)
        
        if not result or 'candidates' not in result:
            raise ValueError(f"No candidates found for {ztf_id}. The object may not exist.")
        
        candidates = result['candidates']
        
        if not candidates:
            raise ValueError(f"No light curve data found for {ztf_id}.")
        
        rows = []
        for cand in candidates:
            # fid: The filter ID for the detection (1 = g and 2 = r)
            # https://lasair.readthedocs.io/en/develop/core_functions/rest-api.html#--api-lightcurves-
            fid = cand.get("fid")
            fid_to_letter = {1: "g", 2: "r"}
            filter_letter = fid_to_letter.get(fid)
            
            row = {
                'MJD': cand.get('mjd', None),
                'filter': filter_letter,
                'unforced_mag': cand.get('magpsf', None),
                'unforced_mag_error': cand.get('sigmapsf', None),
                'unforced_mag_status': 'positive' if cand.get('isdiffpos', 't') == 't' else 'negative',
                'forced_ujy': cand.get('forcediffimflux', None),
                'forced_ujy_error': cand.get('forcediffimfluxunc', None),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Convert MJD to numeric
        if 'MJD' in df.columns:
            df['MJD'] = pd.to_numeric(df['MJD'], errors='coerce')
        
        # Sort by MJD
        if 'MJD' in df.columns:
            df = df.sort_values('MJD').reset_index(drop=True)
        
        # Default save path to data/<run_name>/lasair/
        if save_path is None:
            if project_root is None:
                project_root = Path(__file__).parent.parent
            run_name = _resolve_run_name(run_name)
            lasair_dir = project_root / 'data' / run_name / 'lasair'
            lasair_dir.mkdir(parents=True, exist_ok=True)
            save_path = lasair_dir / f"{ztf_id}_lightcurve.csv"
        
        # Save CSV
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        
        return save_path
        
    except Exception as e:
        if isinstance(e, (ImportError, ValueError)):
            raise
        raise ValueError(f"Failed to download data for {ztf_id}: {str(e)}")

def clean_lasair_lightcurve(lasair_csv_path: Path) -> bool:

    df = pd.read_csv(lasair_csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Basic required columns
    required = ["MJD", "filter", "forced_ujy", "forced_ujy_error", "unforced_mag", "unforced_mag_error"]
    missing = [c for c in ["MJD", "filter"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing} in {lasair_csv_path}")

    # Normalize time + filter
    df["MJD"] = pd.to_numeric(df["MJD"], errors="coerce")
    df["filter"] = df["filter"].astype(str).str.strip().str.lower()

    # Keep g/r only and valid times
    df = df[df["filter"].isin(["g", "r"])].dropna(subset=["MJD"]).copy()

    # Ensure numeric columns exist + are numeric
    for col in ["forced_ujy", "forced_ujy_error", "unforced_mag", "unforced_mag_error"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing forced flux/error from mag if possible
    forced_missing = df["forced_ujy"].isna() | df["forced_ujy_error"].isna()
    compute = forced_missing & df["unforced_mag"].notna() & df["unforced_mag_error"].notna()

    if compute.any():
        mag = df.loc[compute, "unforced_mag"].to_numpy()
        magerr = df.loc[compute, "unforced_mag_error"].to_numpy()

        # AB mag to uJy: m = 23.9 - 2.5 log10(F_uJy)
        f = 10.0 ** (-0.4 * (mag - 23.9))  # uJy
        ferr = f * (np.log(10.0) * 0.4) * magerr

        df.loc[compute, "forced_ujy"] = f
        df.loc[compute, "forced_ujy_error"] = ferr

    # Drop anything still missing forced values or with bad errors
    df = df.dropna(subset=["forced_ujy", "forced_ujy_error"])
    df = df[df["forced_ujy_error"] > 0]

    # Sort by time and write back in-place
    df = df.sort_values("MJD").reset_index(drop=True)
    df.to_csv(lasair_csv_path, index=False)

    return True

def load_lasair_lightcurve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Map raw API column names to normalized names
    col_map = {
        'mjd': 'MJD',
        'fid': 'filter',
        'magpsf': 'unforced_mag',
        'sigmapsf': 'unforced_mag_error',
        'isdiffpos': 'unforced_mag_status',
        'forcediffimflux': 'forced_ujy',
        'forcediffimfluxunc': 'forced_ujy_error',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if 'MJD' in df.columns:
        df['MJD'] = pd.to_numeric(df['MJD'], errors='coerce')
    if 'MJD' in df.columns:
        df = df.sort_values('MJD').reset_index(drop=True)
    return df


def download_tns_ascii(tns_id: str, save_path: Optional[Path] = None,
                      project_root: Optional[Path] = None, run_name: Optional[str] = None) -> Path:

    base_url = "https://www.wis-tns.org"
    object_url = f"{base_url}/object/{tns_id}"
    
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # Get the HTML page
        response = requests.get(object_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse HTML to find spectrum table
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the spectrum table
        spectrum_table = soup.find('table', class_='class-results-table')
        if not spectrum_table:
            raise ValueError(f"No spectrum table found for TNS ID {tns_id}")
        
        # Find all spectrum rows
        spectrum_rows = spectrum_table.find_all('tr', class_='spectrum-row')
        
        if not spectrum_rows:
            raise ValueError(f"No spectrum rows found for TNS ID {tns_id}")
        
        # Extract spectrum info: date and ascii file link
        spectra_info = []
        for row in spectrum_rows:
            # Get observation date
            obs_date_cell = row.find('td', class_='cell-obsdate')
            if not obs_date_cell:
                continue
            
            obs_date_str = obs_date_cell.get_text(strip=True)
            
            # Get ascii file link
            ascii_cell = row.find('td', class_='cell-asciifile')
            if not ascii_cell:
                continue
            
            ascii_link = ascii_cell.find('a')
            if not ascii_link or not ascii_link.get('href'):
                continue
            
            ascii_url = ascii_link['href']
            # Make absolute URL if relative
            if ascii_url.startswith('/'):
                ascii_url = base_url + ascii_url
            elif not ascii_url.startswith('http'):
                ascii_url = base_url + '/' + ascii_url
            
            # Parse date for sorting (format: YYYY-MM-DD HH:MM:SS)
            try:
                from datetime import datetime
                obs_date = datetime.strptime(obs_date_str, '%Y-%m-%d %H:%M:%S')
                spectra_info.append({
                    'date': obs_date,
                    'date_str': obs_date_str,
                    'url': ascii_url,
                    'filename': ascii_link.get_text(strip=True)
                })
            except ValueError:
                # If date parsing fails, still include it but with a default date
                spectra_info.append({
                    'date': datetime.min,
                    'date_str': obs_date_str,
                    'url': ascii_url,
                    'filename': ascii_link.get_text(strip=True)
                })
        
        if not spectra_info:
            raise ValueError(f"No valid spectrum ASCII files found for TNS ID {tns_id}")
        
        # Sort by date (newest first)
        spectra_info.sort(key=lambda x: x['date'], reverse=True)
        
        # Get the newest spectrum
        newest_spectrum = spectra_info[0]
        ascii_url = newest_spectrum['url']
        
        # Download the ASCII file
        ascii_response = requests.get(ascii_url, headers=headers, timeout=30)
        ascii_response.raise_for_status()
        
        # Default save path to data/<run_name>/tns/
        if save_path is None:
            if project_root is None:
                project_root = Path(__file__).parent.parent
            run_name = _resolve_run_name(run_name)
            tns_dir = project_root / 'data' / run_name / 'tns'
            tns_dir.mkdir(parents=True, exist_ok=True)
            save_path = tns_dir / f"{tns_id}_spectrum.ascii"
        
        # Save the file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(ascii_response.text)
            
        return save_path
        
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to download TNS spectrum for {tns_id}: {str(e)}")
    except Exception as e:
        if isinstance(e, (ImportError, ValueError)):
            raise
        raise ValueError(f"Failed to process TNS page for {tns_id}: {str(e)}")


def read_tns_ascii(ascii_path: Path) -> pd.DataFrame:
    # Skip header lines (starting with #) and read data with 3 columns (wavelength, flux, error)
    with open(ascii_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    # Parse the data lines
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:  # At least wavelength and flux
            try:
                wavelength = float(parts[0])
                flux = float(parts[1])
                data.append({'wavelength': wavelength, 'flux': flux})
            except ValueError:
                continue  # Skip lines that can't be parsed
    
    return pd.DataFrame(data)


def plot_spectrum(df, run_dir: Path, metadata: dict | None = None, title: str | None = None, filename: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df["wavelength"], df["flux"], color="black", linewidth=1)

    # ±1sigma uncertainty band
    if "fluxerr" in df.columns:
        ax.fill_between(
            df["wavelength"],
            df["flux"] - df["fluxerr"],
            df["flux"] + df["fluxerr"],
            alpha=0.25
        )

    ax.set_title(title)
    ax.set_xlabel(r"Wavelength ($\AA$)")
    ax.set_ylabel("Flux")
    ax.grid(True, alpha=0.3)

    if filename is None:
        base = title or "spectrum"
        filename = base.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace(".", "")

    return savefig(fig, run_dir, filename)

def plot_light_curve_from_lasair(ztf_id: str, lasair_csv_path: Path, run_dir: Path) -> Path | None:
    """Load Lasair light curve CSV and plot. Returns path to saved figure or None."""
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        light_curve_df = load_lasair_lightcurve(lasair_csv_path)
        df = light_curve_df.copy()
        df.head()

        plot_light_curve(
            df,
            run_dir,
            title=f"Light Curve: {ztf_id}",
            filename=f"{ztf_id}_light_curve"
        )
        return True
    except Exception as e:
        print(f"  Light curve plotting failed: {str(e)}")
        return None


def plot_spectrum_from_tns(ztf_id: str, tns_ascii_path: Path, run_dir: Path) -> Path | None:
    """Load TNS ASCII spectrum and plot. Returns path to saved figure or None."""
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        spectrum_df = read_tns_ascii(tns_ascii_path)
        return plot_spectrum(
            spectrum_df,
            run_dir,
            title=f"Spectrum: {ztf_id}",
            filename=f"{ztf_id}_spectrum"
        )
    except Exception as e:
        print(f"  Spectrum plotting failed: {str(e)}")
        return None