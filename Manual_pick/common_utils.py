#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for the `script/pick` teleseismic preprocessing workflow.

This module centralizes:
1. Event and waveform-processing configuration.
2. TauP travel-time queries.
3. Whole-trace preprocessing before response removal.
4. SACPZ-based response removal.
5. Wiggins-style interpolation to the PDTI target sample interval.
6. Small geometry helpers reused by station-selection scripts.
"""

import os
import math
import numpy as np
import obspy
from obspy.io.sac.sacpz import attach_paz

# ================= CONFIGURATION =================

class Config:
    # ---------------- Earthquake Parameters ----------------
    # Event location and origin time
    EVT_LAT = 39.953           # Latitude (deg)
    EVT_LON = 143.046          # Longitude (deg)
    EVT_DEPTH = 25.0           # Depth (km)
    ORIGIN_TIME_STR = "2026-04-20T07:53:00.000"

    # ---------------- Processing Parameters ----------------
    # Pre-filter low frequency corners (f1, f2).
    # High frequency corners (f3, f4) are calculated dynamically based on Nyquist.
    # PDTI retains high-frequency components
    PRE_FILT_LOW = (0.002, 0.004)
    PRE_FILT_MODE = "AUTO_POLE"  # "AUTO_POLE" or "FIXED"
    # Keep f2 near instrument corner and within a broadband-informed range.
    AUTO_FILT_F2_RATIO = 1.10    # f2 = ratio * fpole
    # Smooth taper zone width on low-frequency side.
    AUTO_FILT_F1_RATIO = 0.50    # f1 = ratio * f2
    # Clip range referenced to common broadband low corners: 360 s to 120 s.
    AUTO_FILT_F2_MIN = 0.004   # Hz
    AUTO_FILT_F2_MAX = 1.0 / 120.0   # Hz

    # Preview-only pre-filter used by `view_sac.py` plots.
    PRE_FILT_VIEW = (0.002, 0.004, 0.8, 1.0)
    
    TARGET_DT = 0.05          # Target sampling interval (s) = 20 Hz
    SCALE_FACTOR = 1.0e6      # Scaling factor to convert units (m/s -> um/s)
    TAPER_PERCENTAGE = 0.05   # Taper length (percentage of trace)

    # ---------------- Processing Options ----------------
    OUTPUT_TYPE = 'DISP'             # 'VEL' (Velocity) or 'DISP' (Displacement)
    # Normalize by max amplitude between P and PP
    # In order to compare with Prof Yagi's pictures, Default FALSE 
    NORMALIZE_WAVEFORM = False      

    # ---------------- Windowing & Output ----------------
    # PP arrival offset for testing (s). Set to 0.0 for original TauP PP time.
    # This offset is used to match the pp_idx in Prof. Yagi's Kamchatka example data.
    # Positive value shifts PP later 
    PP_OFFSET = 0.0
    
    CUT_PRE_P = 10.0          # Time (s) before P-arrival to start cut(Align with Wave.obs)
    NOISE_WINDOW_LEN = 10.0   # Window length (s) for noise STD calculation (Pre-P)
    NPTS_OUT = 8400           # Number of points for output files

    # ------- Preview plotting (used by `view_sac.py`) --------
    PLOT_WIN_START = -10.0    # Plot window start relative to P (s)
    PLOT_WIN_END = 300.0      # Plot window end relative to P (s) 
    AZ_PLOT_SCALE = 6.0       # Azimuth scale factor for record section plots
    
    PHASE_CONFIG = {
        "P": "red",
        "PP": "orange",
        "S": "blue",
        "SS": "cyan"
    }


# ================= UTILITY FUNCTIONS =================

def get_unique_arrivals(model, depth, dist_deg, phase_list, use_latest=False):
    """
    Calculate theoretical travel times using the TauP model.
    
    Parameters
    ----------
    model : TauPyModel
        TauP velocity model instance.
    depth : float
        Source depth in km.
    dist_deg : float
        Epicentral distance in degrees.
    phase_list : list of str
        List of phase names to compute (e.g., ['P', 'PP', 'S']).
    use_latest : bool, optional
        If True, return the latest arrival for each phase type.
        If False (default), return the earliest arrival.
        Due to 410 km  & 660 km discontinuities, multiple arrivals for each phase type.
        
    Returns
    -------
    list
        List of unique Arrival objects (one per phase type).
    """
    try:
        arrivals = model.get_travel_times(source_depth_in_km=depth,
                                          distance_in_degree=dist_deg,
                                          phase_list=phase_list)
        
        # Sort by arrival time (ascending for earliest, descending for latest)
        arrivals.sort(key=lambda x: x.time, reverse=use_latest)
        
        # Filter duplicates (keep only the first encountered arrival of each phase type)
        unique_arrivals = []
        seen_phases = set()
        
        for arr in arrivals:
            if arr.name not in seen_phases:
                unique_arrivals.append(arr)
                seen_phases.add(arr.name)
        
        # Re-sort by time ascending for consistent output order
        unique_arrivals.sort(key=lambda x: x.time)
                
        return unique_arrivals
    except Exception as e:
        print(f"    Error calculating travel times: {e}")
        return []


def get_preferred_p_pick(tr):
    """
    Return the preferred SAC P-pick header using priority t3 -> t4 -> t1.
    Ignore undefined SAC values such as -12345.0.

    Returns
    -------
    dict
        {"pick_time": value or None, "pick_header": header name or None}
    """
    sac = getattr(tr.stats, "sac", None)
    if sac is None:
        return {"pick_time": None, "pick_header": None}

    for header in ("t3", "t4", "t1"):
        if header in sac:
            pick_time = sac[header]
            if pick_time is not None and pick_time != -12345.0:
                return {"pick_time": pick_time, "pick_header": header}

    return {"pick_time": None, "pick_header": None}


def preprocess_trace(tr, taper_percentage=Config.TAPER_PERCENTAGE):
    """
    Apply basic preprocessing: rmean rtrend taper.
    """
    tr.detrend("demean")
    tr.detrend("linear")
    tr.taper(max_percentage=taper_percentage, type="hann")
    return tr


def get_sac_file_mode(sac_path):
    """
    Return the strict SAC naming mode based on the exact filename suffix.

    Supported modes are:
      - legacy: ``.SAC`` paired with ``SACPZ*``
      - modern: ``.sac`` paired with same-basename ``.sacpz``
    """
    sac_name = os.fspath(sac_path)
    if sac_name.endswith(".SAC"):
        return "legacy"
    if sac_name.endswith(".sac"):
        return "modern"
    return None


def resolve_pz_for_sac_path(sac_path, net=None, sta=None, loc=None, chan=None, search_dirs=None):
    """
    Resolve the matching response file for one SAC filename under strict mode rules.

    Returns None when the SAC suffix is unsupported or when no same-mode
    response file exists.
    """
    mode = get_sac_file_mode(sac_path)
    if mode is None:
        return None

    sac_path = os.fspath(sac_path)
    if mode == "modern":
        pz_path = sac_path[:-4] + ".sacpz"
        return pz_path if os.path.exists(pz_path) else None

    if None in {net, sta, chan}:
        return None

    loc_str = loc if loc and str(loc).strip() else "--"
    search_dirs = tuple(search_dirs or (os.path.dirname(sac_path) or ".",))
    candidates = [f"SACPZ.{net}.{sta}.{loc_str}.{chan}"]
    if loc_str in {"", "--"}:
        candidates.append(f"SACPZ.{net}.{sta}..{chan}")
        candidates.append(f"SACPZ.{net}.{sta}.--.{chan}")

    for search_dir in search_dirs:
        for candidate in candidates:
            pz_path = os.path.join(search_dir, candidate)
            if os.path.exists(pz_path):
                return pz_path
    return None


def resolve_pz_filename(tr, search_dirs=None, sac_path=None):
    """
    Resolve the response filename from trace metadata and the current SAC path.
    Returns None if no file is found.
    """
    net = tr.stats.network
    sta = tr.stats.station
    loc = tr.stats.location
    chan = tr.stats.channel
    return resolve_pz_for_sac_path(
        sac_path or "",
        net=net,
        sta=sta,
        loc=loc,
        chan=chan,
        search_dirs=search_dirs,
    )


def parse_sacpz_poles(pz_filename):
    """
    Parse POLES section from a SACPZ file and return a complex ndarray.
    """
    poles = []
    in_poles = False
    expected_count = None

    def _to_float(token):
        return float(token.replace("D", "E").replace("d", "e"))

    with open(pz_filename, "r") as fin:
        for raw_line in fin:
            line = raw_line.strip()
            if not line or line.startswith("*"):
                continue

            upper = line.upper()
            if upper.startswith("POLES"):
                in_poles = True
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        expected_count = int(parts[1])
                    except ValueError:
                        expected_count = None
                continue

            if not in_poles:
                continue

            # End POLES section if another header starts.
            if upper.startswith("ZEROS") or upper.startswith("CONSTANT"):
                break

            parts = line.split()
            if len(parts) < 2:
                continue
            poles.append(complex(_to_float(parts[0]), _to_float(parts[1])))
            if expected_count is not None and len(poles) >= expected_count:
                break

    if not poles:
        raise ValueError(f"No poles parsed from {pz_filename}")
    return np.asarray(poles, dtype=np.complex128)


def estimate_low_corners_from_poles(poles):
    """
    Estimate low-frequency pre-filter corners from pole information.
    """
    magnitudes = np.abs(poles)
    magnitudes = magnitudes[magnitudes > 0.0]
    if magnitudes.size == 0:
        raise ValueError("No valid non-zero poles")

    fpole = float(np.min(magnitudes) / (2.0 * math.pi))
    f2 = Config.AUTO_FILT_F2_RATIO * fpole
    f2 = float(np.clip(f2, Config.AUTO_FILT_F2_MIN, Config.AUTO_FILT_F2_MAX))
    f1 = float(Config.AUTO_FILT_F1_RATIO * f2)
    return f1, f2, fpole


def _validate_prefilt(prefilt):
    """
    Validate that pre_filt satisfies 0 < f1 < f2 < f3 < f4.
    """
    if len(prefilt) != 4:
        raise ValueError(f"pre_filt must have 4 values, got {len(prefilt)}")

    f1, f2, f3, f4 = [float(x) for x in prefilt]
    if not (0.0 < f1 < f2 < f3 < f4):
        raise ValueError(
            f"Invalid pre_filt order: ({f1}, {f2}, {f3}, {f4}), "
            "must satisfy 0 < f1 < f2 < f3 < f4"
        )
    return (f1, f2, f3, f4)


def build_prefilt(tr, pz_filename, mode=Config.PRE_FILT_MODE, pre_filt=None, pre_filt_low=Config.PRE_FILT_LOW):
    """
    Build response-removal pre_filt tuple and return metadata.
    Priority:
      1) Explicit pre_filt argument (manual override)
      2) AUTO_POLE mode (with FIXED fallback)
      3) FIXED mode
    """
    nyquist = 0.5 * tr.stats.sampling_rate
    fixed_f1, fixed_f2 = [float(x) for x in pre_filt_low]
    mode_req = (mode or "FIXED").upper()

    if pre_filt is not None:
        final_filt = _validate_prefilt(tuple(float(x) for x in pre_filt))
        return final_filt, {
            "mode_requested": mode_req,
            "mode_used": "MANUAL",
            "used_fixed_fallback": False,
            "fallback_reason": None,
            "fpole": None,
            "nyquist": nyquist,
            "f1": final_filt[0],
            "f2": final_filt[1],
            "f3": final_filt[2],
            "f4": final_filt[3],
            "pre_filt": final_filt,
        }

    if mode_req not in {"AUTO_POLE", "FIXED"}:
        raise ValueError(f"Unsupported pre-filter mode: {mode_req}")

    fallback_reason = None
    used_fixed_fallback = False
    fpole = None

    if mode_req == "AUTO_POLE":
        try:
            poles = parse_sacpz_poles(pz_filename)
            low_f1, low_f2, fpole = estimate_low_corners_from_poles(poles)
        except Exception as exc:
            low_f1, low_f2 = fixed_f1, fixed_f2
            used_fixed_fallback = True
            fallback_reason = str(exc)
    else:
        low_f1, low_f2 = fixed_f1, fixed_f2

    final_filt = (low_f1, low_f2, nyquist, 2.0 * nyquist)
    try:
        final_filt = _validate_prefilt(final_filt)
    except Exception as exc:
        # Safety fallback for AUTO_POLE if ordering check fails.
        if mode_req == "AUTO_POLE":
            final_filt = _validate_prefilt((fixed_f1, fixed_f2, nyquist, 2.0 * nyquist))
            used_fixed_fallback = True
            fallback_reason = f"auto_prefilt_invalid: {exc}"
        else:
            raise

    mode_used = "FIXED" if (mode_req == "FIXED" or used_fixed_fallback) else "AUTO_POLE"
    return final_filt, {
        "mode_requested": mode_req,
        "mode_used": mode_used,
        "used_fixed_fallback": used_fixed_fallback,
        "fallback_reason": fallback_reason,
        "fpole": fpole,
        "nyquist": nyquist,
        "f1": final_filt[0],
        "f2": final_filt[1],
        "f3": final_filt[2],
        "f4": final_filt[3],
        "pre_filt": final_filt,
    }


def remove_instrument_response(
    tr,
    output_type='VEL',
    pre_filt=None,
    pre_filt_low=Config.PRE_FILT_LOW,
    mode=Config.PRE_FILT_MODE,
    search_dirs=None,
    sac_path=None
):
    """
    Remove instrument response using SAC PoleZero (SACPZ) files.
    Automates the search for SACPZ files based on trace metadata.
    """
    pz_filename = resolve_pz_filename(tr, search_dirs=search_dirs, sac_path=sac_path)
    if pz_filename is None:
        print("    [Skipped] PZ file not found for trace metadata.")
        return None, None

    try:
        final_filt, meta = build_prefilt(
            tr,
            pz_filename,
            mode=mode,
            pre_filt=pre_filt,
            pre_filt_low=pre_filt_low,
        )
    except Exception as e:
        print(f"    Error building pre_filt: {e}")
        return None, None
    
    # Determine conversion mode
    tovel = True if output_type == 'VEL' else False
    
    try:
        attach_paz(tr, pz_filename, tovel=tovel)
        # simulate removes the response
        tr.simulate(paz_remove=tr.stats.paz, pre_filt=final_filt)
        meta["pz_filename"] = pz_filename
        meta["output_type"] = output_type
        return tr, meta
    except Exception as e:
        print(f"    Error removing response: {e}")
        return None, None


def downsample_trace(tr, target_dt=Config.TARGET_DT):
    """
    Downsample trace to target sampling interval using the Wiggins method.
    
    The Wiggins method (weighted average slopes) is equivalent to the SAC
    interpolate command and is preferred for preserving waveform shape without
    applying a strict low-pass filter (anti-aliasing) that standard decimation uses.
    """
    # Check if already at target rate (allow small epsilon for float precision)
    if abs(tr.stats.delta - target_dt) < 1e-6:
        return True
        
    target_sr = 1.0 / target_dt
    
    try:
        tr.interpolate(sampling_rate=target_sr, method='weighted_average_slopes')
    except Exception as e:
        print(f"    [Warning] Wiggins interpolation failed: {e}, falling back to linear")
        try:
            tr.interpolate(sampling_rate=target_sr, method='linear')
        except Exception as e2:
             print(f"    [Error] Linear Fallback failed: {e2}")
             return False
            
    # Verify result
    if abs(tr.stats.delta - target_dt) > 1e-6:
        print(f"    [Error] Resampling failed for {tr.id}. Delta is {tr.stats.delta}")
        return False
        
    return True

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 
    return c * r
