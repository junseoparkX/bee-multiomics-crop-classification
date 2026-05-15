import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN

# ------------------------------------------------------------
# Paths (Windows-safe)
# This script lives in: demographics/code/filtering.py
# Project root is:       demographics/
# Data folder is:        demographics/data/
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]   # .../demographics
DATA = ROOT / "data"

# Input Excel (already in your data folder)
XLSX_PATH = DATA / "meta+expression metrics.xlsx"

# Output CSV for the diagram
OUT_CSV = DATA / "diagram_csv.csv"

SHEET_NAME = "meta_location_pestcide_pathogen"

# ------------------------------------------------------------
# Pathogen feature selection helpers
# ------------------------------------------------------------
def norm_colname(c: str) -> str:
    """Normalize column names: strip + collapse spaces to underscores."""
    c = str(c).strip()
    c = re.sub(r"\s+", "_", c)
    return c

def pick_pathogen_columns(df: pd.DataFrame) -> list[str]:
    """
    Automatically select pathogen/mites related feature columns.
    Rules (case-insensitive):
      - contains 'norm_cop'  (virus copy normalization columns)
      - contains BOTH 'nosema' and 'spore'
      - contains 'mites_per'
    """
    cols = []
    for c in df.columns:
        cl = str(c).lower()
        if "norm_cop" in cl:
            cols.append(c)
        elif ("nosema" in cl) and ("spore" in cl):
            cols.append(c)
        elif "mites_per" in cl:
            cols.append(c)

    # De-duplicate while preserving order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def coerce_numeric(df: pd.DataFrame, cols: list[str], fillna_zero: bool = True) -> pd.DataFrame:
    """
    Convert selected columns to numeric, optionally fill NaN with 0,
    and clip negatives to 0 (many pathogen metrics are non-negative).
    """
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if fillna_zero:
            df[c] = df[c].fillna(0.0)
        df[c] = df[c].clip(lower=0)
    return df

# ------------------------------------------------------------
# Geo clustering helpers (DBSCAN + haversine distance)
# ------------------------------------------------------------
def add_geo_cluster_dbscan(
    df: pd.DataFrame,
    eps_km: float = 100.0,
    min_samples: int = 1,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    out_col: str = "geo_cluster_100km",
) -> pd.DataFrame:
    """
    Add geographic cluster labels using DBSCAN with haversine distance.

    Behaviors:
    - Treat latitude/longitude == 0 as missing (NaN).
    - Treat out-of-range coordinates as missing.
    - eps_km sets the neighborhood radius (kilometers).
    - min_samples=1 makes isolated points their own cluster (no noise label -1).
    """
    out = df.copy()

    # If coordinates are missing, create the column and return
    if lat_col not in out.columns or lon_col not in out.columns:
        out[out_col] = np.nan
        return out

    # Convert to numeric; invalid parses become NaN
    lat = pd.to_numeric(out[lat_col], errors="coerce")
    lon = pd.to_numeric(out[lon_col], errors="coerce")

    # Treat (0, 0) as missing
    lat = lat.mask(lat == 0, np.nan)
    lon = lon.mask(lon == 0, np.nan)

    # Treat invalid coordinate ranges as missing
    lat = lat.mask((lat < -90) | (lat > 90), np.nan)
    lon = lon.mask((lon < -180) | (lon > 180), np.nan)

    out[lat_col] = lat
    out[lon_col] = lon

    valid_mask = lat.notna() & lon.notna()
    out[out_col] = np.nan

    # If nothing is valid, return as-is
    if valid_mask.sum() == 0:
        return out

    # DBSCAN with haversine expects radians
    coords_deg = np.vstack([lat[valid_mask].to_numpy(), lon[valid_mask].to_numpy()]).T
    coords_rad = np.radians(coords_deg)

    # Convert eps from km to radians on Earth's surface
    earth_km = 6371.0088
    eps_rad = eps_km / earth_km

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = db.fit_predict(coords_rad)

    out.loc[valid_mask, out_col] = labels.astype(int)
    return out

def add_cluster_centroids(
    df: pd.DataFrame,
    cluster_col: str,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """
    Add per-cluster centroid latitude/longitude back onto each row.
    Centroids are computed only on rows with valid (cluster, lat, lon).
    """
    out = df.copy()

    needed = [cluster_col, lat_col, lon_col]
    if any(c not in out.columns for c in needed):
        return out

    tmp = out.dropna(subset=[cluster_col, lat_col, lon_col]).copy()
    if tmp.empty:
        out[f"{cluster_col}_lat_center"] = np.nan
        out[f"{cluster_col}_lon_center"] = np.nan
        return out

    cent = (
        tmp.groupby(cluster_col)[[lat_col, lon_col]]
        .mean()
        .rename(
            columns={
                lat_col: f"{cluster_col}_lat_center",
                lon_col: f"{cluster_col}_lon_center",
            }
        )
    )

    out = out.merge(cent, left_on=cluster_col, right_index=True, how="left")
    return out

# ------------------------------------------------------------
# Pesticide summary helpers
# ------------------------------------------------------------
def cols_ending(df: pd.DataFrame, suffix: str):
    """Return all columns whose names end with the given suffix."""
    return [c for c in df.columns if str(c).endswith(suffix)]

def add_pest_summaries(df: pd.DataFrame, prefix: str, cols: list[str]):
    """
    For a set of pesticide columns, compute:
    - sum
    - count_detected (values > 0)
    - max
    """
    if len(cols) == 0:
        return df
    X = df[cols].fillna(0)
    df[f"{prefix}_sum"] = X.sum(axis=1)
    df[f"{prefix}_count_detected"] = (X > 0).sum(axis=1)
    df[f"{prefix}_max"] = X.max(axis=1)
    return df

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if not XLSX_PATH.exists():
        raise FileNotFoundError(f"Excel file not found: {XLSX_PATH}")

    # Load sheet
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME).copy()

    # Optional: normalize column names (helps consistent matching)
    df.columns = [norm_colname(c) for c in df.columns]

    # Create sample_id (safe even if sample is missing)
    if "sample" in df.columns:
        df["sample_id"] = df["sample"].astype(str)
    else:
        df["sample_id"] = df.index.astype(str)

    # Keep only crop in {HBB, CRA, CAC, CAS} (case/space-safe)
    if "crop" in df.columns:
        df["crop"] = df["crop"].astype(str).str.strip().str.upper()
        df = df[df["crop"].isin(["HBB", "CRA", "CAC", "CAS"])].copy()
    else:
        raise KeyError("Column 'crop' not found in the meta sheet.")

    # Keep only timepoint == t2 (handles "t2", "T2", and numeric 2)
    if "timepoint" in df.columns:
        tp = df["timepoint"].astype(str).str.strip().str.lower()
        df = df[tp.isin(["t2", "2"])].copy()
    else:
        raise KeyError("Column 'timepoint' not found in the meta sheet.")

    # Create combined label for sorting/grouping (if tissue exists)
    if "tissue" in df.columns:
        df["label_crop_tissue"] = df["crop"].astype(str) + "_" + df["tissue"].astype(str)

    # Train/valid split by year
    if "year" in df.columns:
        df["split"] = np.where(
            df["year"] == 2020, "train",
            np.where(df["year"] == 2021, "valid", "other")
        )

    # Clean location (treat 0 as missing)
    if "location" in df.columns:
        df["location_clean"] = df["location"].replace(0, np.nan)

    # Exposure bin (optional annotation)
    if "exposure" in df.columns:
        df["exposure_bin"] = df["exposure"].map({"u": 0, "e": 1})

    # ------------------------------------------------------------
    # Lat/Lon cleaning + geo clustering (100 km)
    # ------------------------------------------------------------
    df = add_geo_cluster_dbscan(
        df,
        eps_km=100.0,
        min_samples=1,
        lat_col="latitude",
        lon_col="longitude",
        out_col="geo_cluster_100km",
    )

    df = add_cluster_centroids(
        df,
        cluster_col="geo_cluster_100km",
        lat_col="latitude",
        lon_col="longitude",
    )

    # ------------------------------------------------------------
    # Pesticide summary features only
    # ------------------------------------------------------------
    p_bees   = cols_ending(df, "_bees")
    p_nectar = cols_ending(df, "_nectar")
    p_pollen = cols_ending(df, "_pollen")
    p_wax    = cols_ending(df, "_wax")

    df = add_pest_summaries(df, "pest_bees", p_bees)
    df = add_pest_summaries(df, "pest_nectar", p_nectar)
    df = add_pest_summaries(df, "pest_pollen", p_pollen)
    df = add_pest_summaries(df, "pest_wax", p_wax)

    # Overall pesticide aggregates across all matrices
    sum_cols = [c for c in df.columns if str(c).startswith("pest_") and str(c).endswith("_sum")]
    if len(sum_cols) > 0:
        df["pest_sum_all"] = df[sum_cols].sum(axis=1)

    pest_cols_all = p_bees + p_nectar + p_pollen + p_wax
    if len(pest_cols_all) > 0:
        Xp = df[pest_cols_all].fillna(0)
        df["pest_count_all_detected"] = (Xp > 0).sum(axis=1)
        df["pest_max_all"] = Xp.max(axis=1)

    # Pest burden: value-based 6 bins (interpretable)
    if "pest_sum_all" in df.columns and df["pest_sum_all"].notna().any():
        bins = [-np.inf, 1, 10, 100, 1000, 5000, np.inf]
        labels = ["≤ 1", "1–10", "10–100", "100–1k", "1k–5k", "> 5k"]

        df["pest_sum_all_bin6_label"] = pd.cut(
            df["pest_sum_all"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        codes = df["pest_sum_all_bin6_label"].cat.codes  # -1 for NaN
        df["pest_sum_all_bin6"] = codes.where(codes >= 0, np.nan) + 1

    # ------------------------------------------------------------
    # NEW: Pathogen (+mites) feature selection + cleaning
    # ------------------------------------------------------------
    path_cols = pick_pathogen_columns(df)
    if len(path_cols) > 0:
        df = coerce_numeric(df, path_cols, fillna_zero=True)

    # ------------------------------------------------------------
    # Select final columns:
    # core meta + pesticide summaries + geo cluster outputs + pathogen features
    # ------------------------------------------------------------
    core_meta = [
        "sample_id", "crop", "tissue", "location", "year",
        "label_crop_tissue", "split",
        "province", "location_clean", "month",
        "exposure", "exposure_bin",
        "plate", "replicate",
        "latitude", "longitude",
        "geo_cluster_100km",
        "geo_cluster_100km_lat_center", "geo_cluster_100km_lon_center",
    ]

    # Keep all derived pesticide summary columns
    pest_summary_cols = [c for c in df.columns if str(c).startswith("pest_")]

    # Final keep list (only keep columns that actually exist)
    keep_cols = [c for c in core_meta if c in df.columns] + pest_summary_cols

    # Add pathogen features (only if found)
    if len(path_cols) > 0:
        keep_cols += path_cols

    out_df = df[keep_cols].copy()

    # Save
    out_df.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV)
    print("Shape:", out_df.shape)

    # Quick sanity checks
    if "geo_cluster_100km" in out_df.columns:
        n_valid = out_df["geo_cluster_100km"].notna().sum()
        n_clusters = out_df["geo_cluster_100km"].dropna().nunique()
        print(f"[INFO] geo_cluster_100km: valid_rows={n_valid}, num_clusters={n_clusters}")

    if "pest_sum_all_bin6_label" in out_df.columns:
        print("[INFO] pest_sum_all_bin6_label value counts:")
        print(out_df["pest_sum_all_bin6_label"].value_counts(dropna=False))

    if len(path_cols) > 0:
        print(f"[INFO] Pathogen/Mites features included: {len(path_cols)}")
        for c in path_cols:
            print("  -", c)
    else:
        print("[WARN] No pathogen/mites feature columns found by patterns.")
        print("       Expected: '*norm_cop*', '*nosema*spore*', '*mites_per*'.")

if __name__ == "__main__":
    main()
