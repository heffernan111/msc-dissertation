# MSc Project

## Goal and scientific context

This pipeline builds a **low-redshift Hubble diagram** from ZTF Type Ia supernovae: it infers distance moduli from light-curve fits and compares them to a fiducial Flat ΛCDM cosmology.

- **Why Type Ia:** Type Ia SNe are standardizable candles. The SALT2 model yields parameters (t0, x0, x1, c); the Tripp relation then gives the distance modulus: μ = m_B − M + α x1 − β c.
- **Data flow:** ZTF catalogue -> light curves (Lasair) -> host redshifts (TNS) -> SALT2 fits (sncosmo) -> distance moduli and Hubble residuals -> Hubble diagram and H0 summary.


**Main quantities:**  
- **SALT2:** t0 (time of B-band maximum), x0 (flux scale), x1 (stretch), c (colour).  
- **Distance:** mB (peak B mag from fit), μ_obs = mB − M + α x1 − β c (observed distance modulus), μ_th from FlatLambdaCDM(H0=70, Om0=0.3), residual = μ_obs − μ_th.  
Step 5 uses **host redshift** from TNS when available (stored in `ztf_cleansed.csv` as `host_redshift`).  
References: Tripp (1998) for standardization; SALT2 (Guy et al.); [sncosmo](https://sncosmo.readthedocs.io/).

---

## ZTF CSV workflow

1. Export/download your ZTF data as a CSV (e.g. from the [ZTF BTS explorer](https://sites.astro.caltech.edu/ztf/bts/explorer.php)).
2. Put the file in the project root folder and name it **exactly** `ztf.csv`.

---

## Pipeline and run order

Run the steps in order: **1 -> 2 -> 3 -> 4 -> 5 -> 6**. Step 3 (TNS) can be run after step 2 and before step 4.

### 1. Build catalogue

- **a. Download ztf.csv**  
  Place `ztf.csv` in the project root.

- **b. Run `1_cleanseZTF`**
  - **Used:** Column **type** - filter to `type == 'SN Ia'`.
  - **Used:** Column **peakt** - convert to MJD: `peakt + 57999.5` (peakt is JD-2458000).
  - **Used:** Column **IAUID** - keep only rows with `IAUID.startswith('SN')` for Lasair compatibility.

- **c. Produces** `ztf_cleansed.csv` (project root).

### 2. Lasair light curves

- **a. Run `2_downloadLasair`**
  - Lasair API: `client.object(ztf_id)` per object (from `ztf_cleansed.csv`).

- **b. Enter folder name for current run**  
  Creates `runs/<run_name>/` and writes one light-curve CSV per object.

- **c. Download light curve data**  
  Per object: MJD, filter (g/r), unforced_mag, unforced_mag_error, forced_ujy, forced_ujy_error (from Lasair).

- **d. Produce forced_ujy if not in data**  
  `processLasairData()` (in `src/utils.py`) fills **forced_ujy** and **forced_ujy_error** from unforced mag when empty, using AB mag -> μJy: `F_μJy = 10^(-0.4*(mag - 23.9))`.

**Output:** `runs/<run>/<ZTFID>_lightcurve.csv` (MJD, filter, forced_ujy, forced_ujy_error, etc.).

### 3. TNS host redshifts

- **a. Run `3_downloadTNS`**
  - Reads `ztf_cleansed.csv` and run folder name (to know which objects are in the run).
  - For each object, fetches the object page from [TNS](https://www.wis-tns.org/) and parses **host redshift** (or redshift).
  - Caches results in `src/tns_data.csv` (ZTFID, IAUID, host_redshift).
  - Merges host redshifts into `ztf_cleansed.csv` as column **host_redshift**.

- **b. Note:** TNS rate-limits requests; the notebook uses delays between calls. This step is optional if you already have redshifts in ZTF; step 5 falls back to the redshift column from the SALT2 run when `host_redshift` is missing.

### 4. SALT2 fits (sncosmo)

- **a. Run `4_generateSNCosmo`**  
  Enter the same run folder name.

- **b. For each light curve, fit a SALT2 model**
  - **From light-curve CSV:** MJD -> `time`, filter (g/r) -> bands (ztfg/ztfr), forced_ujy -> `flux`, forced_ujy_error -> `fluxerr`, zp=23.9, zpsys=ab.
  - **From `ztf_cleansed.csv`:** **redshift** (and **host_redshift** when available), **A_V** -> mwebv as A_V/3.1 (MW dust).
  - **Fitted parameters:** t0, x0, x1, c (mwebv and z fixed per object).

- **Produces:** `runs/<run>/sncosmo_fits.csv` - columns include ztf_id, redshift, ncall, ndof, chisq, t0, x0, x1, c.

### 5. Distance moduli and Hubble residuals

- **a. Run `5_processDistance`**
  - **Input:** Run folder name; file `runs/<run>/sncosmo_fits.csv`.
  - **Redshift:** Uses `host_redshift` from `ztf_cleansed.csv` when available, else redshift from the fits.
  - **Computed:**
    - **mB** from SALT2 (mB = −2.5 log10(x0) + 10.635).
    - **μ_obs** = mB - M + α x1 - β c (Tripp standardization; M, α, β fixed).
    - **μ_th** = theoretical distance modulus from FlatLambdaCDM(H0=70, Om0=0.3).
    - **resid** = μ_obs − μ_th (Hubble residual).

- **Produces:** `runs/<run>/distance_process.csv` - ztf_id, host_redshift, ncall, ndof, chisq, t0, x0, x1, c, mB, mu_obs, mu_th, resid.

### 6. Plot Hubble diagram

- **a. Run `6_plotHubbleDiagram`**  
  Enter the run name. Loads `runs/<run>/distance_process.csv`, plots μ_obs vs log(z) (and optionally μ_th(z)), reports combined H₀ or residual statistics, and saves `runs/<run>/hubble_diagram.png`.
