# MSc Project

## ZTF CSV workflow

1. Export/download your ZTF data as a CSV.
2. Put the file in the project root folder and name it **exactly** `ztf.csv`.

---

## Pipeline

### 1. Build catalogue

- **a. Download ztf.csv**  
  Place `ztf.csv` in the project root (see above).

- **b. Run `1_cleanseZTF`**
  - **Used:** Column **type** - filter to `type == 'SN Ia'`.
  - **Used:** Column **peakt** - convert to MJD: `peakt + 57999.5`.

- **c. Produces** `ztf_cleansed.csv` (project root).

### 2. Lasair light curves

- **a. Run `2_downloadLasair`**
  - Lasair API: `client.object(ztf_id)` per object (from `ztf_cleansed.csv`).

- **b. Enter folder name for current run**  
  Creates `runs/<run_name>/` and writes one light-curve CSV per object.

- **c. Download light curve data**  
  Per object: MJD, filter (g/r), unforced_mag, unforced_mag_error, forced_ujy, forced_ujy_error (from Lasair).

- **d. Produce forced_ujy if not in data**  
  `processLasairData()` fills **forced_ujy** and **forced_ujy_error** from unforced mag when empty, using AB mag → uJy: `F_uJy = 10^(-0.4*(mag - 23.9))`.

**Output:** `runs/<run>/<ZTFID>_lightcurve.csv` (MJD, filter, forced_ujy, forced_ujy_error, etc.).

### 3. Sncosmo (SALT2 fits)

- **a. Run `3_generateSNCosmo`**  
  Enter the same run folder name.

- **b. For each light curve, build a SALT2 model**
  - **From light-curve CSV:**
    - MJD → `time`
    - filter (g/r) → band names (ztfg / ztfr)
    - forced_ujy → `flux`
    - forced_ujy_error → `fluxerr`
    - zp = 23.9, zpsys = ab
  - **From `ztf_cleansed.csv`:**
    - **redshift** - fixed in model per object.
    - **A_V** - converted to E(B−V) as `A_V / 3.1` and set as **mwebv** (MW dust).
  - **Fitted parameters:** t0, x0, x1, c (mwebv and z fixed per object).
  - SALT2 + MW dust gives standardized light-curve parameters (t0, x0, x1, c) and redshift needed for distance and H₀.

- **Produces:** `runs/<run>/sncosmo_parameters.csv`: object_id, chisq, ndof, z, t0, x0, x1, c, mwebv, t0_err, x0_err, x1_err, c_err.  
  Step 4 needs (object_id, z, t0, x0, x1, c) to compute apparent/absolute peak mag and thus distance and H₀.

### 4. Find distance

- **a. Run `4_processDistance`**
  - **Input:** User run folder name; file: `runs/<run>/sncosmo_parameters.csv`.
  - **Used per row:** z, t0, x0, x1, c - set on a sncosmo SALT2 model (no MW dust here; extinction already in the fit).
  - **Computed:**
    - **m_B** = `model.source_peakmag("bessellb", "ab")` - apparent B magnitude at peak.
    - **M_B** = `model.source_peakabsmag("bessellb", "ab")` - absolute B at peak.
    - **μ** = m_B − M_B (distance modulus).
    - **d_L_Mpc** = 10^(μ/5 + 1) pc then ÷ 10^6.
    - **H₀** = c×z/d_L_Mpc (c in km/s, d_L in Mpc).
  - **Why:** μ and d_L are the distance; H₀ per SN is the low-z Hubble relation.

- **Produces:** `runs/<run>/distances.csv`: object_id, z, t0, x0, x1, c, m_B, μ, d_L_Mpc, H₀.

### 5. Plot Hubble diagram

- **a. Run `5_plotHubbleDiagram`**  
  Enter the run name. Loads `runs/<run>/distances.csv`, plots magnitude vs z and distance modulus vs z, reports combined H₀ (mean ± std), and saves `runs/<run>/hubble_diagram.png`.
