# ============================================================
# plot_geo_panels_rect_units.py
#   Paper-style geo figure (RECT layout using UNIT coordinates):
#     - Rectangles are (x, y, w, h) in "units"
#       with origin at bottom-left, y increases upward.
#     - Code converts them to Matplotlib add_axes fractions.
#
#   Output: exactly 2 PNGs (2020 + 2021)
#   Labels ONLY: BC, AB (N), AB (S), MB, QC
#
# Requirements:
#   conda install -c conda-forge cartopy
#   (or) pip install cartopy
# ============================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature


# -----------------------------
# Auto paths
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / "data"
DEFAULT_OUTDIR = ROOT / "figures" / "geo_panels"

CSV_CANDIDATES = [
    DATA_DIR / "diagram_csv.csv",
    DATA_DIR / "diagram_csv.csv.csv",
]
DEFAULT_CSV = next((p for p in CSV_CANDIDATES if p.exists()), CSV_CANDIDATES[0])


# -----------------------------
# Cluster -> Region label mappings
# -----------------------------
CLUSTER_REGION_SHORT = {
    0: "1. BC",
    4: "2. AB (N)",
    3: "3. AB (S)",
    2: "4. MB",
    1: "5. QC",
}
PANEL_ORDER = [0, 4, 3, 2, 1]
CLUSTER_NUM = {0: "1", 4: "2", 3: "3", 2: "4", 1: "5"}


# ============================================================
# Layout (YOU EDIT THESE)
# 핵심: H_ROW를 원하는 값으로 "직접" 넣고, CANVAS_H를 자동 계산해서
#      H_ROW를 키워도 실제로 패널이 커지게 만든다.
# ============================================================
CANVAS_W = 100

M = 1.0   # outer margin
G = 0.5   # gap between panels

H_MAIN = 36.0   # main height (원하면 이것도 바꿔)
H_ROW  = 50.0   # <-- 여기! 네가 원하는 row 높이로 마음대로 늘려 (예: 50)

# CANVAS_H를 고정 100으로 두면 H_ROW를 크게 못 늘림(잘림).
# 그래서 H_ROW/H_MAIN에 맞춰 CANVAS_H를 자동으로 키움.
CANVAS_W = 100
M = 1.0
G = 0.5

H_MAIN = 44.0      # main을 더 크게 (원하는 만큼)
H_MID  = 46.0      # p1,p2,p3 높이
H_BOT  = 34.0      # p4,p5 높이 (이걸 줄이면 됨)

# CANVAS_H는 자동 계산 (잘림 방지)
CANVAS_H = 2*M + 2*G + H_MAIN + H_MID + H_BOT

# widths
W_ALL = CANVAS_W - 2*M
W3 = (W_ALL - 2*G) / 3
W2 = (W_ALL - 1*G) / 2

# y positions (아래에서 위로)
Y_BOT = M
Y_MID = M + H_BOT + G
Y_TOP = M + H_BOT + G + H_MID + G

RECTS_U = {
    "main": (M, Y_TOP, W_ALL, H_MAIN),

    "p1": (M,             Y_MID, W3, H_MID),
    "p2": (M + W3 + G,    Y_MID, W3, H_MID),
    "p3": (M + 2*(W3+G),  Y_MID, W3, H_MID),

    "p4": (M,          Y_BOT, W2, H_BOT),
    "p5": (M + W2 + G, Y_BOT, W2, H_BOT),
}

DEBUG_LAYOUT_ONLY = False


# -----------------------------
# Helpers
# -----------------------------
def urect_to_axes(rect_u, canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    """(x,y,w,h) units -> [left,bottom,width,height] fractions (0..1)."""
    x, y, w, h = rect_u
    return [x / canvas_w, y / canvas_h, w / canvas_w, h / canvas_h]


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _clean_latlon(df, lat_col="latitude", lon_col="longitude"):
    out = df.copy()

    if lat_col not in out.columns:
        out[lat_col] = np.nan
    if lon_col not in out.columns:
        out[lon_col] = np.nan

    out[lat_col] = _to_num(out[lat_col])
    out[lon_col] = _to_num(out[lon_col])

    out.loc[out[lat_col] == 0, lat_col] = np.nan
    out.loc[out[lon_col] == 0, lon_col] = np.nan

    out.loc[(out[lat_col] < -90) | (out[lat_col] > 90), lat_col] = np.nan
    out.loc[(out[lon_col] < -180) | (out[lon_col] > 180), lon_col] = np.nan
    return out


def _make_cluster_color_map(unique_cluster_ids):
    cmap = plt.get_cmap("tab10")
    ids = sorted(int(x) for x in unique_cluster_ids)
    colors = [cmap(i) for i in range(len(ids))]
    return {cid: colors[i] for i, cid in enumerate(ids)}


def _add_basemap(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.55, edgecolor="0.25")
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.50, edgecolor="0.25")

    admin1 = NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(admin1, linewidth=0.35, edgecolor="0.35", alpha=0.85)

    ax.add_feature(cfeature.LAKES.with_scale("10m"), alpha=0.10)
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), alpha=0.10)


def _cluster_centers(d, cluster_col, lat_col, lon_col):
    return (
        d.groupby(cluster_col)
        .agg(mean_lat=(lat_col, "mean"), mean_lon=(lon_col, "mean"), n=(cluster_col, "size"))
        .reset_index()
        .sort_values(cluster_col)
    )


def _extent_from_points(lon, lat, pad_lon=1.0, pad_lat=1.0):
    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
    return [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat]


def _panel_border(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(0.9)
        sp.set_edgecolor("0.35")


def _label_box(ax, text, x=0.02, y=0.98, fs=12):
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=fs, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.35", lw=0.7, alpha=0.98),
        zorder=10,
        clip_on=True,
    )


def _compute_global_inset_extents(d_all, cluster_col, lat_col, lon_col):
    extents = {}
    for cid in PANEL_ORDER:
        sub = d_all[d_all[cluster_col] == cid]
        if sub.empty:
            continue
        lon = sub[lon_col].to_numpy()
        lat = sub[lat_col].to_numpy()
        extents[cid] = _extent_from_points(lon, lat, pad_lon=0.9, pad_lat=0.7)
    return extents


def _validate_rects():
    """Rectangles should stay inside canvas (optional warnings)."""
    for k, (x, y, w, h) in RECTS_U.items():
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            print(f"[WARN] bad rect {k}: {RECTS_U[k]}")
        if x + w > CANVAS_W or y + h > CANVAS_H:
            print(f"[WARN] rect {k} exceeds canvas: {RECTS_U[k]} over {CANVAS_W}x{CANVAS_H}")


def plot_year(
    df_year: pd.DataFrame,
    out_path: Path,
    year: int,
    d_all_clean: pd.DataFrame,
    cluster_col: str = "geo_cluster_100km",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    point_size_main: float = 46.0,
    point_size_inset: float = 58.0,
    alpha: float = 0.92,
):
    d = _clean_latlon(df_year, lat_col=lat_col, lon_col=lon_col)
    d = d.dropna(subset=[lat_col, lon_col]).copy()
    if d.empty:
        print(f"[WARN] year={year}: valid (lat,lon) rows = 0 -> skip")
        return

    if cluster_col not in d.columns:
        print(f"[WARN] year={year}: missing cluster column '{cluster_col}' -> skip")
        return

    d[cluster_col] = _to_num(d[cluster_col])
    d = d[np.isfinite(d[cluster_col])].copy()
    if d.empty:
        print(f"[WARN] year={year}: cluster ids all NaN -> skip")
        return
    d[cluster_col] = d[cluster_col].astype(int)

    unique_ids = sorted(d_all_clean[cluster_col].unique().tolist())
    color_map = _make_cluster_color_map(unique_ids)
    inset_extents = _compute_global_inset_extents(d_all_clean, cluster_col, lat_col, lon_col)

    x = d[lon_col].to_numpy()
    y = d[lat_col].to_numpy()
    cids = d[cluster_col].to_numpy(dtype=int)
    colors = [color_map[int(cid)] for cid in cids]

    centers = _cluster_centers(d, cluster_col, lat_col, lon_col)

    proj = ccrs.PlateCarree()
    transform = ccrs.PlateCarree()

    # 캔버스 비율이 바뀌면 figsize도 같이 늘려주는 게 보기 좋음(선택)
    # 기존 7.0 기준으로 CANVAS_H가 커지면 세로도 조금 키움
    fig = plt.figure(figsize=(8.4, 7.0), facecolor="white")


    # --- MAIN MAP ---
    ax_main = fig.add_axes(urect_to_axes(RECTS_U["main"]), projection=proj)
    ax_main.set_aspect("auto")
    _add_basemap(ax_main)

    main_extent = _extent_from_points(x, y, pad_lon=2.0, pad_lat=2.0)
    ax_main.set_extent(main_extent, crs=transform)

    if not DEBUG_LAYOUT_ONLY:
        ax_main.scatter(
            x, y,
            s=point_size_main,
            c=colors,
            alpha=alpha,
            linewidths=0.0,
            edgecolors="none",
            transform=transform,
            zorder=3,
        )

        ax_main.text(
            0.985, 0.985, f"{year}",
            transform=ax_main.transAxes,
            ha="right", va="top",
            fontsize=15, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="0.35", lw=0.8, alpha=0.98),
            zorder=10,
            clip_on=True,
        )
                
        # --- main: numeric labels only (1..5) ---
        for _, r in centers.iterrows():
            cid = int(r[cluster_col])
            lab = CLUSTER_NUM.get(cid, "")
            if not lab:
                continue
            ax_main.annotate(
                lab,
                xy=(r["mean_lon"], r["mean_lat"]),
                xytext=(6, 6),
                textcoords="offset points",
                ha="left", va="bottom",
                fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.25", lw=0.6, alpha=0.95),
                transform=transform,
                zorder=6,
                clip_on=True,
            )


    ax_main.set_xticks([])
    ax_main.set_yticks([])
    _panel_border(ax_main)

    # --- INSETS (3 + 2) ---
    panel_keys = ["p1", "p2", "p3", "p4", "p5"]
    for key, cid in zip(panel_keys, PANEL_ORDER):
        ax = fig.add_axes(urect_to_axes(RECTS_U[key]), projection=proj)
        ax.set_aspect("auto")  # <-- 오타 수정: ax_main 말고 ax
        _add_basemap(ax)

        if cid in inset_extents:
            ax.set_extent(inset_extents[cid], crs=transform)

        sub = d[d[cluster_col] == cid].copy()
        if (not DEBUG_LAYOUT_ONLY) and (not sub.empty):
            subx = sub[lon_col].to_numpy()
            suby = sub[lat_col].to_numpy()
            sub_colors = [color_map[int(v)] for v in sub[cluster_col].to_numpy(dtype=int)]
            ax.scatter(
                subx, suby,
                s=point_size_inset,
                c=sub_colors,
                alpha=alpha,
                linewidths=0.0,
                edgecolors="none",
                transform=transform,
                zorder=3,
            )

        lab = CLUSTER_REGION_SHORT.get(cid, f"{cid}")
        _label_box(ax, lab, fs=12)

        ax.set_xticks([])
        ax.set_yticks([])
        _panel_border(ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, pad_inches=0.0)  # tight 제거 유지
    plt.close(fig)
    print("Saved:", out_path)


def main():
    print(f"[INFO] CANVAS_W={CANVAS_W}, CANVAS_H={CANVAS_H:.2f} (auto from H_ROW/H_MAIN)")
    _validate_rects()

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--cluster_col", type=str, default="geo_cluster_100km")
    ap.add_argument("--lat_col", type=str, default="latitude")
    ap.add_argument("--lon_col", type=str, default="longitude")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    need_cols = ["year", args.cluster_col, args.lat_col, args.lon_col]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        print("[ERROR] Missing columns:", missing)
        return

    d_all = _clean_latlon(df, lat_col=args.lat_col, lon_col=args.lon_col)
    d_all = d_all.dropna(subset=[args.lat_col, args.lon_col]).copy()
    d_all[args.cluster_col] = _to_num(d_all[args.cluster_col])
    d_all = d_all[np.isfinite(d_all[args.cluster_col])].copy()
    if d_all.empty:
        print("[ERROR] After cleaning, no valid rows remain.")
        return
    d_all[args.cluster_col] = d_all[args.cluster_col].astype(int)

    for y in [2020, 2021]:
        dy = df[df["year"] == y].copy()
        if dy.empty:
            print(f"[WARN] year {y} not found -> skip")
            continue
        out_path = outdir / f"geo_panels_{y}.png"
        plot_year(
            dy,
            out_path=out_path,
            year=y,
            d_all_clean=d_all,
            cluster_col=args.cluster_col,
            lat_col=args.lat_col,
            lon_col=args.lon_col,
        )


if __name__ == "__main__":
    main()
