# MSc Project

## ZTF CSV workflow

1. Export/download your ZTF data as a CSV.
2. Put the file in the project `data/` folder and name it **exactly**:
   - `data/ztf.csv`
3. Open and run the **main Jupyter notebook** (the project’s primary `.ipynb`).
4. When prompted in the notebook:
   - **Run name**: this determines where outputs are stored.
     - If you enter a name, data will be stored under a run folder for that name.
     - If you leave it blank, the run folder defaults to **today’s date**.
   - **Process selection**:
     - Enter **`all`** to process everything in `ztf.csv`
     - Or enter a specific **ID** to process only that target/object from `ztf.csv`
   - **Overwrite**:
     - You’ll be asked `y/n` whether to overwrite existing data for the same run/ID.
