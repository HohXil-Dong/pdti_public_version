"""
plot_potency.py: Visualize sub-fault slip and potency tensor distribution.

Reads PDTI inversion results (fort.40, pdtdis.dat, pddis.dat) and generates
a map-view plot with slip contours and beach-balls for each sub-fault.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as mpl_cm
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pyrocko import moment_tensor as pmt
from pyrocko.plot import beachball

# Scientific colormap (fallback to viridis)
try:
    from cmcrameri import cm as crameri_cm
    CMAP = crameri_cm.bilbao_r
except ImportError:
    print("Warning: cmcrameri not found, using viridis.")
    CMAP = mpl_cm.viridis

# Global plot style
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.fontset'] = 'custom'


def load_fort40_info(infile):
    """Parse fort.40 header for hypocenter and model geometry."""
    with open(infile, 'r') as f:
        lines = f.readlines()
    
    # Line 2: hypocenter info
    p = lines[1].split()
    rigid, ref_lat, ref_lon, ref_depth = float(p[2]), float(p[3]), float(p[4]), float(p[5])
    
    # Line 6: grid dimensions
    d = lines[5].split()
    dx, dy, mn, nn = float(d[0]), float(d[1]), int(d[2]), int(d[3])
    
    return ref_lat, ref_lon, ref_depth, mn, nn, dx, dy, rigid


def configure_map_axes(ax, extent):
    """Set map extent and configure tick formatters."""
    lon_min, lon_max, lat_min, lat_max = extent
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    xticks = np.arange(np.ceil(lon_min), np.floor(lon_max) + 0.1, 1.0)
    yticks = np.arange(np.ceil(lat_min), np.floor(lat_max) + 0.1, 1.0)
    
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())


def plot_potency(fort40_path, pdtdis_path, pddis_path, output_file='potency_distribution.png'):
    """Main plotting routine for slip and potency tensor distribution."""
    
    # --- Load metadata ---
    ref_lat, ref_lon, ref_depth, mn, nn, dx, dy, rigid = load_fort40_info(fort40_path)
    print(f"Hypocenter: {ref_lat:.3f}N, {ref_lon:.3f}E, {ref_depth:.1f}km")
    
    # scaling: dx*dy (km^2 -> m^2) * rigidity (GPa -> Pa)
    moment_scale = dx * dy * rigid * 1.0e15  # constant rigidity
    
    # --- Load slip grid (pddis.dat) ---
    col_names = ['lat', 'lon', 'x', 'y', 'slip', 'depth']
    try:
        df_grid = pd.read_csv(pddis_path, sep=r'\s+', names=col_names, header=None)
    except FileNotFoundError:
        print(f"Error: {pddis_path} not found.")
        return

    # Reshape for contour plotting
    try:
        X = df_grid.pivot(index='y', columns='x', values='lon').values
        Y = df_grid.pivot(index='y', columns='x', values='lat').values
        Z = df_grid.pivot(index='y', columns='x', values='slip').values
    except Exception as e:
        print(f"Error reshaping grid data: {e}")
        return

    max_slip = df_grid['slip'].max()
    
    # Map extent with padding
    pad = 0.2
    extent = [df_grid['lon'].min() - pad, df_grid['lon'].max() + pad,
              df_grid['lat'].min() - pad, df_grid['lat'].max() + pad]

    # --- Load potency tensors (pdtdis.dat) ---
    try:
        df_mt = pd.read_csv(
            pdtdis_path, sep=r'\s+', header=None,
            usecols=[0, 1, 4, 5, 7, 8, 9, 10, 11, 12, 13],
            names=['n', 'm', 'lat', 'lon', 'slip', 
                   'm_dd', 'm_nn', 'm_ee', 'm_nd', 'm_ed_neg', 'm_ne_neg']
        )
    except FileNotFoundError:
        print(f"Error: {pdtdis_path} not found.")
        return

    if len(df_mt) != mn * nn:
        print(f"Warning: Loaded {len(df_mt)} subfaults, expected {mn*nn}.")

    # --- Initialize figure ---
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=ccrs.Mercator())
    configure_map_axes(ax, extent)
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    
    # --- Plot slip contours ---
    levels = np.arange(0, max_slip + 1.0, 1.0)
    print(f"Contour interval: 1.0 m (Max Slip: {max_slip:.2f} m)")
    
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=CMAP, 
                     transform=ccrs.PlateCarree(), alpha=0.8)
    
    # Colorbar
    ax_cb = ax.inset_axes([0.65, 0.05, 0.3, 0.02])
    cbar = plt.colorbar(cf, cax=ax_cb, orientation='horizontal')
    cbar.set_label('Slip (m)', fontsize=8, labelpad=2)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0, max_slip / 2.0, max_slip])
    
    # --- Compute moment for each subfault ---
    tensors = []
    max_moment = 0.0
    
    for _, row in df_mt.iterrows():
        mt_mat = pmt.symmat6(
            row['m_dd'], row['m_nn'], row['m_ee'], 
            row['m_nd'], row['m_ed_neg'], row['m_ne_neg']
        )
        mt_scaled = pmt.MomentTensor(m_up_south_east=mt_mat * moment_scale)
        moment = mt_scaled.moment
        max_moment = max(max_moment, moment)
        
        tensors.append({
            'row': row,
            'mt_orientation': pmt.MomentTensor(m_up_south_east=mt_mat),
            'moment': moment
        })

    if max_moment == 0:
        print("No beachballs to plot (Max Moment = 0).")
        return

    # --- Plot beachballs ---
    # Size scaling: linear in (moment / max_moment), capped to TARGET_MW_MAX
    TARGET_MW_MAX = 8.0
    SIZE_FACTOR = 2.4
    norm = mpl.colors.Normalize(vmin=0, vmax=max_slip)
    
    for item in tensors:
        row = item['row']
        mt = item['mt_orientation']
        mw_fake = max((item['moment'] / max_moment) * TARGET_MW_MAX, 0.1)
        size = SIZE_FACTOR * mw_fake
        color = CMAP(norm(row['slip']))
        
        proj_pt = ax.projection.transform_point(row['lon'], row['lat'], src_crs=ccrs.Geodetic())
        
        try:
            # Deviatoric fill
            beachball.plot_beachball_mpl(
                mt, ax, beachball_type='deviatoric', position=proj_pt, size=size,
                color_t=color, color_p='white', edgecolor='none', linewidth=0, zorder=10
            )
            # DC nodal planes
            beachball.plot_beachball_mpl(
                mt, ax, beachball_type='dc', position=proj_pt, size=size,
                color_t='none', color_p='none', edgecolor='black', linewidth=0.3, zorder=11
            )
        except Exception:
            pass  

    # --- Plot hypocenter ---
    ax.scatter(ref_lon, ref_lat, transform=ccrs.PlateCarree(), marker='*', s=300, 
               c='yellow', edgecolors='black', zorder=20, label='Hypocenter')
    
    ax.set_title(f"Potency Tensor Distribution\nHypocenter: {ref_lat}N {ref_lon}E {ref_depth}km")

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Moment Tensor Potency Distribution")
    parser.add_argument("fort40", help="Path to fort.40 file")
    parser.add_argument("pdtdis", help="Path to pdtdis.dat file")
    parser.add_argument("pddis", help="Path to pddis.dat file")
    parser.add_argument("--output", default="potency_distribution.png", help="Output filename")
    
    args = parser.parse_args()
    
    if all(os.path.exists(f) for f in [args.fort40, args.pdtdis, args.pddis]):
        plot_potency(args.fort40, args.pdtdis, args.pddis, args.output)
    else:
        print("Error: One or more input files not found.")
