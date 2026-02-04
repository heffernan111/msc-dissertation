"""
Light-curve utilities for the ZTF SN Ia pipeline.

Cleans Lasair light-curve CSVs and fills forced flux (forced_ujy, forced_ujy_error)
from unforced magnitude when missing. Flux is in microjanskys (μJy) with AB zeropoint 23.9.
"""
import pandas as pd
import numpy as np


def processLasairData(df_obj, out_path):
    """Read light curve CSV, fill forced_ujy/forced_ujy_error from unforced_mag where missing, clean and overwrite."""
    df = pd.read_csv(out_path)
    df.columns = [c.strip() for c in df.columns]

    if "MJD" not in df.columns or "filter" not in df.columns:
        raise KeyError(f"Missing required columns MJD/filter in {out_path}")

    df["MJD"] = pd.to_numeric(df["MJD"], errors="coerce")
    df["filter"] = df["filter"].astype(str).str.strip().str.lower()
    df = df[df["filter"].isin(["g", "r"])].dropna(subset=["MJD"]).copy()

    for col in ["forced_ujy", "forced_ujy_error", "unforced_mag", "unforced_mag_error"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # AB mag to μJy: m = 23.9 - 2.5*log10(F_μJy) => F_μJy = 10^(-0.4*(m - 23.9))
    # Error propagation: first-order in mag (dF/dm = F * ln(10) * 0.4)
    forced_missing = df["forced_ujy"].isna() | df["forced_ujy_error"].isna()
    compute = forced_missing & df["unforced_mag"].notna() & df["unforced_mag_error"].notna()
    if compute.any():
        mag = df.loc[compute, "unforced_mag"].to_numpy()
        magerr = df.loc[compute, "unforced_mag_error"].to_numpy()
        f = 10.0 ** (-0.4 * (mag - 23.9))  # μJy
        ferr = f * (np.log(10.0) * 0.4) * magerr
        df.loc[compute, "forced_ujy"] = f
        df.loc[compute, "forced_ujy_error"] = ferr

    df = df.dropna(subset=["forced_ujy", "forced_ujy_error"])
    df = df[df["forced_ujy_error"] > 0]
    df = df.sort_values("MJD").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"Successfully cleaned and wrote light curve data to {out_path} ({len(df)} rows)")