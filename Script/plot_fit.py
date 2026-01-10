import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import math

def read_obssyn_file(filepath):
    """
    Read *BHZ.obs01 or *BHZ.syn01 file.
    Format: Time  Value  StationName  ...
    Returns:
        times (numpy array): time series
        values (numpy array): amplitude values
        station_name (str): station name (from filename or content)
    """
    try:
        if not os.path.exists(filepath):
            return None, None, None

        data = []
        times = []
        station_name_in_file = "Unknown"

        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            return None, None, None

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    t = float(parts[0])
                    val = float(parts[1])
                    times.append(t)
                    data.append(val)
                    if len(parts) >= 3:
                        station_name_in_file = parts[2]
                except ValueError:
                    continue
        
        return np.array(times), np.array(data), station_name_in_file
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None, None

def load_station_info(filepath):
    """
    Load station information from a text file.
    Expected columns: Station (0), Azimuth (4), Distance (5)
    Returns: dict {station_name: {'az': azimuth, 'dist': distance}}
    """
    station_info = {}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Skip header assuming first line is header if it contains "Station"
        start_idx = 0
        if len(lines) > 0 and "Station" in lines[0]:
            start_idx = 1
            
        for line in lines[start_idx:]:
            parts = line.strip().split()
            if len(parts) >= 6:
                name = parts[0]
                try:
                    az = float(parts[4])
                    dist = float(parts[5])
                    station_info[name] = {'az': az, 'dist': dist}
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error reading station info file: {e}")
        return None
        
    return station_info

def plot_comparison(obs_path, syn_path, output_dir, station_info=None):
    """
    Plot comparison between observed and synthetic values.
    """
    basename = os.path.basename(obs_path)
    # Remove extension and channel to get station name
    station_name = basename.replace("BHZ.obs01", "")
    
    t_obs, d_obs, _ = read_obssyn_file(obs_path)
    t_syn, d_syn, _ = read_obssyn_file(syn_path)
    
    if t_obs is None or d_obs is None or len(t_obs) == 0:
        print(f"Skipping {station_name}: No valid observation data.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(t_obs, d_obs, 'k-', linewidth=1.5, label='Observed', alpha=0.8)
    
    if t_syn is not None and d_syn is not None and len(t_syn) > 0:
        ax.plot(t_syn, d_syn, 'r-', linewidth=1.5, label='Synthetic', alpha=0.8)
    else:
        print(f"Warning: No synthetic data found for {station_name}")

    title_str = f"Station: {station_name}"
    if station_info and station_name in station_info:
        info = station_info[station_name]
        title_str += f" | Az: {info['az']:.1f}° Dist: {info['dist']:.1f}°"
    
    ax.set_title(title_str, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Velocity (um/s)", fontsize=12)
    ax.legend(loc='upper right')
    
    # Tick optimization: 20s labels, 10s ticks
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    
    if len(t_obs) > 0:
        ax.set_xlim(0, max(t_obs))
    
    output_filename = os.path.join(output_dir, f"{station_name}_fit.png")
    fig.savefig(output_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_filename}")

    # Return data for summary plot
    az = -1
    dist = -1
    if station_info and station_name in station_info:
        az = station_info[station_name]['az']
        dist = station_info[station_name]['dist']

    return {
        'station': station_name,
        'times': t_obs,
        'obs': d_obs,
        't_syn': t_syn,
        'syn': d_syn,
        'az': az,
        'dist': dist,
        'plot_limit': max(t_obs) if len(t_obs) > 0 else 0
    }

def plot_summary(all_data, output_dir, rows_per_page=6, cols_per_page=6):
    """
    Plot all stations in multi-page summary figures, sorted by azimuth.
    Values are from Obs (black) and Syn (red).
    """
    if not all_data:
        return

    # Sort by azimuth
    sorted_data = sorted(all_data, key=lambda x: x['az'])

    # Determine global X max (keep time axis uniform across all pages)
    max_plot_limit = 0
    for d in sorted_data:
        max_plot_limit = max(max_plot_limit, d['plot_limit'])

    # Split into pages 
    total_stations = len(sorted_data)
    stations_per_page = rows_per_page * cols_per_page
    total_pages = (total_stations + stations_per_page - 1) // stations_per_page

    print(f"Total stations: {total_stations}. Splitting into {total_pages} pages.")

    for page in range(total_pages):
        start_idx = page * stations_per_page
        end_idx = min((page + 1) * stations_per_page, total_stations)
        page_data = sorted_data[start_idx:end_idx]
        
        n_page_stations = len(page_data)
        
        # Determine actual rows needed for this page
        current_rows = (n_page_stations + cols_per_page - 1) // cols_per_page
        
        # Figure height adjusted for actual rows
        fig, axes = plt.subplots(current_rows, cols_per_page, figsize=(cols_per_page * 4, current_rows * 2.0), sharex=True, squeeze=False)
        
        for i, data in enumerate(page_data):
            row = i // cols_per_page
            col = i % cols_per_page
            
            ax = axes[row, col]
            
            # Plot Obs
            if data['obs'] is not None:
                ax.plot(data['times'], data['obs'], 'k-', linewidth=0.6, label='Obs')
            
            # Plot Syn
            if data['syn'] is not None and data['t_syn'] is not None:
                ax.plot(data['t_syn'], data['syn'], 'r-', linewidth=0.6, label='Syn')
            
            # Calculate max amplitude for info and local scaling
            max_amp_obs = 0.0
            if data['obs'] is not None and len(data['obs']) > 0:
                max_amp_obs = np.max(np.abs(data['obs']))
            
            # Determine Local Y Limit (1.5x max amplitude)
            local_ylim = max_amp_obs * 1.5
            if local_ylim <= 0:
                local_ylim = 1.0 # Default fallback
                
            ax.set_ylim(-local_ylim, local_ylim)

            # Info text - Split into two lines
            line1 = f"{data['station']} (Dist:{data['dist']:.1f}°, Az:{data['az']:.0f}°)"
            line2 = f"Max: {max_amp_obs:.2f} um/s"
            
            ax.text(0.02, 0.93, line1, transform=ax.transAxes, va='top', fontsize=8, fontweight='bold')
            ax.text(0.02, 0.83, line2, transform=ax.transAxes, va='top', fontsize=8, color='blue')
            
            ax.set_yticks([]) # Hide Y ticks to reduce clutter
            ax.grid(False)
            ax.set_xlim(0, max_plot_limit)
            
            # Ticks
            ax.xaxis.set_major_locator(MultipleLocator(20)) 
            ax.xaxis.set_minor_locator(MultipleLocator(10)) 
            ax.tick_params(axis='x', labelsize=8) 
            
            if row == current_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=10)
            
        # Hide unused axes on this page
        for j in range(n_page_stations, current_rows * cols_per_page):
            row = j // cols_per_page
            col = j % cols_per_page
            axes[row, col].axis('off')

        plt.tight_layout()
        out_file = os.path.join(output_dir, f"all_stations_fit_summary_page_{page + 1}.png")
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved summary plot page {page + 1}: {out_file}")

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Plot observed vs synthetic waveforms (obs01/syn01).")
    parser.add_argument("wave_syn_dir", help="Path to wave.syn directory")
    parser.add_argument("--output", "-o", dest="output_dir", default=None, help="Output directory for plots (default: ../fits relative to wave_syn_dir)")
    parser.add_argument("--info", "-i", dest="station_info_file", default=None, help="Path to station info file (for Az/Dist in title)")
    
    args = parser.parse_args()
    
    wave_dir = args.wave_syn_dir
    
    if args.output_dir is None:
        abs_wave = os.path.abspath(wave_dir)
        parent_dir = os.path.dirname(abs_wave)
        output_dir = os.path.join(parent_dir, "fits")
    else:
        output_dir = args.output_dir
    
    if not os.path.exists(wave_dir):
        print(f"Error: wave.syn directory not found: {wave_dir}")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    s_info = None
    if args.station_info_file:
        print(f"Loading station info from {args.station_info_file}...")
        s_info = load_station_info(args.station_info_file)
    
    obs_files = sorted(glob.glob(os.path.join(wave_dir, "*BHZ.obs01")))
    
    if not obs_files:
        print(f"No *BHZ.obs01 files found in {wave_dir}")
    else:
        print(f"Found {len(obs_files)} observation files in {wave_dir}")
        print(f"Output directory: {output_dir}")
        
        all_stations_data = []
        for obs_file in obs_files:
            dirname = os.path.dirname(obs_file)
            basename = os.path.basename(obs_file)
            syn_filename = basename.replace("BHZ.obs01", "BHZ.syn01")
            syn_file = os.path.join(dirname, syn_filename)
            
            res = plot_comparison(obs_file, syn_file, output_dir, station_info=s_info)
            if res:
                all_stations_data.append(res)
        
        print("Generating summary plot...")
        plot_summary(all_stations_data, output_dir)
            
        print("All plots finished.")
