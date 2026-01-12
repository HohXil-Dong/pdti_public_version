#!/usr/bin/env python3
"""
Generate Final Inversion Input (Marked Data)

This script processes "Marked" SAC files (with manual P-picks) to generate
the final input data for inversion. It is similar to gen_inv.py but uses
manual picks (T3 header) rather than theoretical arrivals for alignment,
and includes some post-processing steps.

Steps:
1.  Read SAC files from 'Marked' directory.
2.  Align using Manual P-Pick (T3) or fallback to T1.
3.  Preprocess, Remove Response, Downsample.
4.  Cut waveform aligned on manual P-picks.
5.  Zero out data 300s after P-picks(similar to Okuwaki).
6.  Select best traces(From Different Kholes) and generate output.

Dependencies:
    common_utils.py
"""

import obspy
from obspy import UTCDateTime
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees, gps2dist_azimuth
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import argparse

# Import Shared Configuration and Utilities
from common_utils import (
    Config, 
    get_unique_arrivals, 
    preprocess_trace, 
    remove_instrument_response, 
    downsample_trace
)

# Local Constants
OUTPUT_DIR = "wave.obs"
STATION_INFO_FILE = "station_info_final.txt"
SAC_FILE_PATTERN = "Marked/*.SAC"

# ================= LOCAL HELPER FUNCTIONS =================

def calculate_geometry_and_theoretical_arrivals(tr, model, origin_time):
    """
    Calculate geometry and fetch Manual Pick (T3) or fallback (T1).
    """
    try:
        if not (hasattr(tr.stats, 'sac') and 'stla' in tr.stats.sac and 'stlo' in tr.stats.sac):
            print(f"    [Skipped] Missing SAC headers (stla/stlo)")
            return None

        stla = tr.stats.sac.stla
        stlo = tr.stats.sac.stlo
        
        # Geometry
        dist_m, az, baz = gps2dist_azimuth(Config.EVT_LAT, Config.EVT_LON, stla, stlo)
        dist_deg = locations2degrees(Config.EVT_LAT, Config.EVT_LON, stla, stlo)
        
        # use manual P-pick & theoretical PP 
        arrivals = get_unique_arrivals(model, Config.EVT_DEPTH, dist_deg, ["PP"])
        pp_arr = next((arr for arr in arrivals if arr.name == 'PP'), None)
        
        p_time_rel_origin = None
        
        # Check T3 (Manual)
        if 't3' in tr.stats.sac and tr.stats.sac.t3 is not None:
             pick_time = tr.stats.sac.t3
        # Fallback T1 (Obspy)
        elif 't1' in tr.stats.sac and tr.stats.sac.t1 is not None:
             pick_time = tr.stats.sac.t1
             print(f"    [Info] Using T1 (Obspy) fallback for {tr.stats.station}")
        else:
             print("    [Skipped] No T3 or T1 P-pick found.")
             return None

        # Calculate Travel Time relative to Origin
        if 'o' in tr.stats.sac and tr.stats.sac.o is not None:
            p_time_rel_origin = pick_time - tr.stats.sac.o
        else:
            # Calculate reference time (relative to start time)
            # 'b' header is start time relative to reference time
            ref_time = tr.stats.starttime - tr.stats.sac.b
            o_calc = origin_time - ref_time
            p_time_rel_origin = pick_time - o_calc
                
        return {
            "p_time": p_time_rel_origin,
            "pp_time": pp_arr.time if pp_arr else None,
            "az": az,
            "dist_deg": dist_deg,
            "stla": stla,
            "stlo": stlo
        }
    except Exception as e:
        print(f"    Error in geometry/arrival calc: {e}")
        return None

def cut_waveform_and_stats(tr, origin_time, p_time_rel_origin, pp_time_rel_origin):
    """
    Cut, scale, normalize, and specifically zero-out late data (300s post-P).
    """
    # 1. Unit scale first (m -> um)
    tr.data = tr.data * Config.SCALE_FACTOR
    
    # 2. Calculate Cut Absolute Times and Trim
    p_time_abs = origin_time + p_time_rel_origin
    start_time = p_time_abs - Config.CUT_PRE_P
    duration = (Config.NPTS_OUT - 1) * Config.TARGET_DT
    end_time = start_time + duration
    
    tr_cut = tr.copy()
    tr_cut.trim(start_time, end_time, nearest_sample=True, pad=True, fill_value=0)
    
    # Enforce NPTS
    if tr_cut.stats.npts > Config.NPTS_OUT:
        tr_cut.data = tr_cut.data[:Config.NPTS_OUT]
    elif tr_cut.stats.npts < Config.NPTS_OUT:
        pad_width = Config.NPTS_OUT - tr_cut.stats.npts
        tr_cut.data = np.pad(tr_cut.data, (0, pad_width), 'constant')

    # Indices
    expected_p_idx = int(round(Config.CUT_PRE_P / Config.TARGET_DT))
    p_rel_start = (p_time_abs - tr_cut.stats.starttime)
    p_idx = int(round(p_rel_start / Config.TARGET_DT))

    # Check if p_idx is close to expected_p_idx
    if abs(p_idx - expected_p_idx) <= 2:
        p_idx = expected_p_idx
        p_rel_start = float(p_idx) * Config.TARGET_DT
    else:
        print(f"    [Warning] P-arrival index mismatch: expected {expected_p_idx}, got {p_idx}")

    # Offset Removal: Zero at P-arrival
    if 0 <= p_idx < len(tr_cut.data):
        offset = tr_cut.data[p_idx]
        tr_cut.data = tr_cut.data - offset
        
    # Zero out data 300s after P-arrival (Specific to Final Inversion)
    samps_300s = int(round(300.0 / Config.TARGET_DT))
    idx_300s = p_idx + samps_300s
    
    if idx_300s < len(tr_cut.data):
        tr_cut.data[idx_300s:] = 0.0

    # Calculate Noise STD (Pre-P)
    if p_idx > 0:
        noise_std = np.std(tr_cut.data[:p_idx])
    else:
        noise_std = 0.0

    # PP Index
    pp_idx = None
    pp_time_abs = origin_time + pp_time_rel_origin
    pp_rel_start = (pp_time_abs - tr_cut.stats.starttime)
    pp_idx = int(round(pp_rel_start / Config.TARGET_DT))
        
    # Normalization
    if Config.NORMALIZE_WAVEFORM and pp_idx and pp_idx > p_idx:
        window = tr_cut.data[p_idx : pp_idx + 1]
        if len(window) > 0:
            max_amp = np.max(np.abs(window))
            if max_amp > 1e-9:
                tr_cut.data = tr_cut.data / max_amp
                
    return {
        "data": tr_cut.data,
        "npts": Config.NPTS_OUT,
        "dt": Config.TARGET_DT,
        "noise_std": noise_std,
        "p_idx": p_idx,
        "pp_idx": pp_idx,
        "p_rel_start": p_rel_start,
        "pp_rel_start": pp_rel_start
    }

def plot_waveform(times, data, p_sec, pp_sec, station, channel, noise_std, output_path):
    """Generate plotting for verification."""
    fig = plt.figure(figsize=(10, 4))
    plt.plot(times, data, 'k', linewidth=0.5, label='Waveform')
    
    plt.axvline(x=p_sec, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='P')
    plt.text(p_sec, np.max(np.abs(data)) * 0.8, 'P', color='red', rotation=90, verticalalignment='bottom')
    
    if pp_sec and pp_sec > 0 and pp_sec < times[-1]:
         plt.axvline(x=pp_sec, color='orange', linestyle='--', alpha=0.8, linewidth=1.5, label='PP')
         plt.text(pp_sec, np.max(np.abs(data)) * 0.8, 'PP', color='orange', rotation=90, verticalalignment='bottom')

    unit = "Velocity (um/s)" if Config.OUTPUT_TYPE == 'VEL' else "Displacement (um)"
    if Config.NORMALIZE_WAVEFORM: unit = "Normalized Amplitude"
    
    plt.title(f"{station}.{channel}\nNoise Std: {noise_std:.2e}")
    plt.xlabel("Time (s)")
    plt.ylabel(unit)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.grid(True, which='major', linestyle='-', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

# ================= MAIN EXECUTION =================

def process_file(sac_file, model, origin_time):
    """Pipeline for a single file. Returns (result, report_flags)."""
    report = {
        'station': 'Unknown', 
        'missing_t3': False, 
        'missing_pz': False,
        'resp_rem_fail': False,
        'resample_fail': False,
        'other_error': False,
        'success': False
    }
    
    try:
        st = obspy.read(sac_file)
        tr = st[0]
        report['station'] = tr.stats.station
        
        # Check variables for reporting
        if 't3' not in tr.stats.sac or tr.stats.sac.t3 is None:
            report['missing_t3'] = True
            
        # Check PZ existence
        net = tr.stats.network
        sta = tr.stats.station
        loc = tr.stats.location
        chan = tr.stats.channel
        loc_str = loc if loc and loc.strip() else "--"
        pz_filename = f"SACPZ.{net}.{sta}.{loc_str}.{chan}"
        if not os.path.exists(pz_filename):
             pz_fallback = f"SACPZ.{net}.{sta}..{chan}"
             if not os.path.exists(pz_fallback):
                 report['missing_pz'] = True

        # 1. Geometry & Arrivals
        arr_info = calculate_geometry_and_theoretical_arrivals(tr, model, origin_time)
        if not arr_info: 
            return None, report
        
        # 2. Preprocess
        tr = preprocess_trace(tr)
        
        # 3. Instrument Response
        tr, used_nyquist = remove_instrument_response(tr)
        if tr is None: 
            if not report['missing_pz']:
                report['resp_rem_fail'] = True
            return None, report
        
        # 4. Downsample
        if not downsample_trace(tr): 
            report['resample_fail'] = True
            return None, report
        
        # 5. Cut, Scale, Noise, Normalize
        cut_info = cut_waveform_and_stats(tr, origin_time, arr_info['p_time'], arr_info['pp_time'])
        
        result = {
            "station": tr.stats.station,
            "net": tr.stats.network,
            "loc": tr.stats.location.strip() if tr.stats.location else "--",
            "channel": tr.stats.channel,
            "filename": sac_file,
            "arr_info": arr_info,
            "cut_info": cut_info,
            "noise_std": cut_info['noise_std'],
            "final_data": cut_info['data']
        }
        
        report['success'] = True
        return result, report

    except Exception as e:
        print(f"Error processing {sac_file}: {e}")
        report['other_error'] = True
        return None, report

def main():
    parser = argparse.ArgumentParser(description="Generate Final Inversion Input (Marked)")
    parser.add_argument('datadir', nargs='?', default='.', help='Data directory containing Marked/ folder')
    args = parser.parse_args()
    
    if args.datadir != '.':
        try:
            os.chdir(args.datadir)
            print(f"Changed working directory to: {args.datadir}")
        except FileNotFoundError:
            print(f"Error: Directory '{args.datadir}' not found.")
            return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"--- Starting Processing (Marked Files) ---")
    print(f"Target DT: {Config.TARGET_DT}s | Output: {Config.OUTPUT_TYPE} | Norm: {Config.NORMALIZE_WAVEFORM}")
    
    model = TauPyModel(model="iasp91")
    origin_time = UTCDateTime(Config.ORIGIN_TIME_STR)
    
    sac_files = glob.glob(SAC_FILE_PATTERN)
    if not sac_files:
        print(f"No SAC files found in {SAC_FILE_PATTERN}.")
        return

    # Statistics Containers
    total_processed = 0
    perfect_count = 0
    
    # Error Lists
    err_missing_t3 = set()
    err_missing_pz = set()
    err_resp_rem = set()  
    err_resample = set()  
    err_other = set()     
    
    # 1. Processing Loop
    station_groups = {} 
    
    for f in sac_files:
        print(f"Processing {f}...")
        res, report = process_file(f, model, origin_time)
        
        total_processed += 1
        sta = report['station']
        
        if report['missing_t3']: err_missing_t3.add(sta)
        if report['missing_pz']: err_missing_pz.add(sta)
        if report['resp_rem_fail']: err_resp_rem.add(sta)
        if report['resample_fail']: err_resample.add(sta)
        if report['other_error']: err_other.add(sta)
        
        if report['success']:
            perfect_count += 1
            if res:
                sname = res['station']
                if sname not in station_groups:
                    station_groups[sname] = []
                station_groups[sname].append(res)
                print(f"  -> OK. Std: {res['noise_std']:.2e}")
        else:
            reasons = []
            if report['missing_pz']: reasons.append("Missing SACPZ")
            if report['missing_t3']: reasons.append("Missing T3")
            if report['resp_rem_fail']: reasons.append("Response Removal Failed")
            if report['resample_fail']: reasons.append("Resampling Failed")
            if report['other_error']: reasons.append("Other Error")
            print(f"  -> Failed: {', '.join(reasons)}")

    # 2. Selection Loop
    print("\n--- Selecting Best Trace per Station ---")
    final_list = []
    
    sorted_stations = sorted(station_groups.keys())
    for sta in sorted_stations:
        candidates = station_groups[sta]
        best = min(candidates, key=lambda x: x['noise_std'])
        final_list.append(best)
        
        out_name = f"{best['station']}{best['channel']}"
        out_txt_path = os.path.join(OUTPUT_DIR, out_name)
        out_img_path = os.path.join(OUTPUT_DIR, out_name + ".png")
        
        h = best['cut_info']
        a = best['arr_info']
        header = (f"{Config.CUT_PRE_P:.2f}   {h['dt']:.4f}    {h['npts']}    "
                  f"{a['stla']:.4f}  {a['stlo']:.4f}     "
                  f"{h['pp_rel_start']:.2f}      {h['pp_idx'] if h['pp_idx'] else 0}      {h['noise_std']:.6e}")
        
        with open(out_txt_path, 'w') as f_out:
            f_out.write(header + "\n")
            for val in best['final_data']:
                f_out.write(f"{val:.6e}\n")
                
        times = np.linspace(0, (h['npts'] - 1) * h['dt'], h['npts'])
        plot_waveform(times, best['final_data'], h['p_rel_start'], h['pp_rel_start'],
                      best['station'], best['channel'], h['noise_std'], out_img_path)
                      
        print(f"  {sta}: Selected {best['filename']} (Std: {best['noise_std']:.2e}) -> {out_name}")

    # 3. Summary Statistics
    if final_list:
        info_path = os.path.join(OUTPUT_DIR, STATION_INFO_FILE)
        with open(info_path, 'w') as f:
             f.write(f"{'Station':<8} {'Channel':<8} {'Latitude':<10} {'Longitude':<10} {'Azimuth':<10} {'Distance':<10} {'Khole':<6} {'StdDev':<12} {'P_time':<10} {'PP_time':<10} {'Net':<4} {'Filename'}\n")
             for item in final_list:
                 a = item['arr_info']
                 khole = item['loc']
                 p_str = f"{a['p_time']:.2f}"
                 pp_str = f"{a['pp_time']:.2f}" if a['pp_time'] else "N/A"
                 f.write(f"{item['station']:<8} {item['channel']:<8} "
                         f"{a['stla']:<10.4f} {a['stlo']:<10.4f} "
                         f"{a['az']:<10.2f} {a['dist_deg']:<10.2f} "
                         f"{khole:<6} {item['noise_std']:<12.4e} "
                         f"{p_str:<10} {pp_str:<10} "
                         f"{item['net']:<4} {item['filename']}\n")
                         
        stds = [x['noise_std'] for x in final_list]
        names = [f"{x['station']}{x['channel']}" for x in final_list]
        fig, ax = plt.subplots(figsize=(10, len(final_list)*0.3 + 2))
        y_pos = np.arange(len(names))
        ax.hlines(y=y_pos, xmin=0, xmax=stds, color='skyblue')
        ax.plot(stds, y_pos, 'o', color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel("Noise Std Dev")
        ax.set_title("Selected Station Noise Levels")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "noise_std.png"))
        plt.close(fig)

    # 4. Final Processing Report
    print("\n" + "="*45)
    print("          PROCESSING EXCEPTION REPORT")
    print("="*45)
    print(f"Total Files Processed: {total_processed}")
    print(f"Successfully Processed: {perfect_count}")
    print("-" * 45)
    
    def print_err_list(title, sta_set):
        print(f"{title} ({len(sta_set)} stations):")
        if sta_set:
            print(", ".join(sorted(list(sta_set))))
        else:
            print("None")
        print("-" * 45)

    print_err_list("Missing T3 Variable", err_missing_t3)
    print_err_list("Missing SACPZ File", err_missing_pz)
    print_err_list("Response Removal Failed (ObsPy Error)", err_resp_rem)
    print_err_list("Resampling Failed (Wiggins Error)", err_resample)
    print_err_list("Other Unhandled Exceptions", err_other)
    
    print("="*45 + "\n")

if __name__ == "__main__":
    main()
