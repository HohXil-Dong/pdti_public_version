# -*- coding: utf-8 -*-
"""
Common Utility Module for Teleseismic Data Processing

This module contains shared configuration constants and utility functions
used across multiple scripts (view_sac.py, gen_obs.py, etc.) to process
teleseismic waveforms.

It handles:
1.  Global Configuration (Earthquake Constants)
2.  Travel Time Calculation (TauP)
3.  Waveform Preprocessing (rmean rtrend taper)
4.  Instrument Response Removal
5.  Downsampling (Wiggins Method without applying anti-aliasing filter)
6.  Geodesic Distance Calculation
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
    EVT_LAT = 52.512           # Latitude (deg)
    EVT_LON = 160.324          # Longitude (deg)
    EVT_DEPTH = 33.0           # Depth (km)
    ORIGIN_TIME_STR = "2025-07-29T23:24:52.000"

    # ---------------- Processing Parameters ----------------
    # Pre-filter low frequency corners (f1, f2).
    # High frequency corners (f3, f4) are calculated dynamically based on Nyquist.
    # PDTI retains high-frequency components
    PRE_FILT_LOW = (0.0033, 0.02)

    # DO NOT USE BANDPASS FILTER IN PDTI
    # Specific Pre-filter for Viewing (Visual Inspection)
    # Filters out high frequencies for cleaner plots (f3=0.8, f4=1.0)
    PRE_FILT_VIEW = (0.0033, 0.02, 0.8, 1.0)
    
    TARGET_DT = 0.05          # Target sampling interval (s) = 20 Hz
    SCALE_FACTOR = 1.0e6      # Scaling factor to convert units (m/s -> um/s)
    TAPER_PERCENTAGE = 0.05   # Taper length (percentage of trace)

    # ---------------- Processing Options ----------------
    OUTPUT_TYPE = 'VEL'             # 'VEL' (Velocity) or 'DISP' (Displacement)
    # Normalize by max amplitude between P and PP
    # In order to compare with Prof Yagi's pictures, Default FALSE 
    NORMALIZE_WAVEFORM = False      

    # ---------------- Windowing & Output ----------------
    CUT_PRE_P = 10.0          # Time (s) before P-arrival to start cut(Align with Wave.obs)
    NOISE_WINDOW_LEN = 10.0   # Window length (s) for noise STD calculation (Pre-P)
    NPTS_OUT = 8192           # Number of points for output files

    # ---------------- Visualization ----------------
    PLOT_WIN_START = -10.0    # Plot window start relative to P (s)
    PLOT_WIN_END = 200.0      # Plot window end relative to P (s) 
    AZ_PLOT_SCALE = 6.0       # Azimuth scale factor for record section plots
    
    PHASE_CONFIG = {
        "P": "red",
        "PP": "orange",
        "S": "blue",
        "SS": "cyan"
    }


# ================= UTILITY FUNCTIONS =================

def get_unique_arrivals(model, depth, dist_deg, phase_list):
    """
    Calculate theoretical travel times using the TauP model.
    """
    try:
        arrivals = model.get_travel_times(source_depth_in_km=depth,
                                          distance_in_degree=dist_deg,
                                          phase_list=phase_list)
        
        # Sort by arrival time
        arrivals.sort(key=lambda x: x.time)
        
        # Filter duplicates (keep only the first arrival of each phase type)
        unique_arrivals = []
        seen_phases = set()
        
        for arr in arrivals:
            if arr.name not in seen_phases:
                unique_arrivals.append(arr)
                seen_phases.add(arr.name)
                
        return unique_arrivals
    except Exception as e:
        print(f"    Error calculating travel times: {e}")
        return []


def preprocess_trace(tr, taper_percentage=Config.TAPER_PERCENTAGE):
    """
    Apply basic preprocessing: rmean rtrend taper.
    """
    tr.detrend("demean")
    tr.detrend("linear")
    tr.taper(max_percentage=taper_percentage, type="hann")
    return tr


def remove_instrument_response(tr, output_type='VEL', pre_filt=None, pre_filt_low=Config.PRE_FILT_LOW):
    """
    Remove instrument response using SAC PoleZero (SACPZ) files.
    Automates the search for SACPZ files based on trace metadata.
    """
    sr = tr.stats.sampling_rate
    nyquist = 0.5 * sr
    
    if pre_filt:
        final_filt = pre_filt
    else:
        # Construct Pre-filter: (f1, f2, f3=Nyquist, f4=2*Nyquist)
        f3 = nyquist
        f4 = 2.0 * nyquist
        final_filt = (pre_filt_low[0], pre_filt_low[1], f3, f4)
    
    # Construct SACPZ Filename
    net = tr.stats.network
    sta = tr.stats.station
    loc = tr.stats.location
    chan = tr.stats.channel
    # Handle empty location codes which might appear as spaces or empty strings
    loc_str = loc if loc and loc.strip() else "--"
    
    pz_filename = f"SACPZ.{net}.{sta}.{loc_str}.{chan}"
    if not os.path.exists(pz_filename):
        # Try fallback for empty location code '..' which sometimes occurs
        pz_fallback = f"SACPZ.{net}.{sta}..{chan}"
        if os.path.exists(pz_fallback):
             pz_filename = pz_fallback
        else:
             print(f"    [Skipped] PZ file not found: {pz_filename}")
             return None, None

    # Determine conversion mode
    tovel = True if output_type == 'VEL' else False
    
    try:
        attach_paz(tr, pz_filename, tovel=tovel)
        # simulate removes the response
        tr.simulate(paz_remove=tr.stats.paz, pre_filt=final_filt)
        return tr, nyquist
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
