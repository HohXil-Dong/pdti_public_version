import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys

def load_fort40_params(fort40_path):
    # Load parameters from fort.40
    with open(fort40_path, 'r') as f:
        lines = f.readlines()
    
    parts = lines[1].split()
    ref_lat = float(parts[3])
    ref_lon = float(parts[4])
    ref_depth = float(parts[5])
    
    parts = lines[3].split()
    strike_model = float(parts[0])
    dip_model = float(parts[1])
    
    parts = lines[5].split()
    dx = float(parts[0]) 
    dy = float(parts[1])
    mn = int(parts[2])
    nn = int(parts[3])
    
    return ref_lat, ref_lon, ref_depth, strike_model, dip_model, dx, dy, mn, nn

def load_pdtdis_slip(pdtdis_path, mn, nn):
    # Load slip data from pdtdis.dat
    data_slip = np.zeros((nn, mn))
    
    min_str = 1e9
    max_str = -1e9
    min_dip = 1e9
    max_dip = -1e9
    
    with open(pdtdis_path, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts: continue
            
            i_dip = int(parts[0]) - 1
            i_str = int(parts[1]) - 1
            
            d_dip = float(parts[2])
            d_str = float(parts[3])
            slip = float(parts[7])
            
            if i_dip < nn and i_str < mn:
                data_slip[i_dip, i_str] = slip
                
            min_dip = min(min_dip, d_dip)
            max_dip = max(max_dip, d_dip)
            min_str = min(min_str, d_str)
            max_str = max(max_str, d_str)
                
    return data_slip, min_dip, max_dip, min_str, max_str

def get_deglat_per_km(alat):
    # Calculate degrees latitude per km
    rad = 0.017453292
    delta = 0.1
    a = 6378.137
    b = 6356.752
    e = np.sqrt(1. - (b/a)**2)
    
    sin_val = np.sin((alat + delta/2.0) * rad)
    rc = a * (1. - e**2) / np.sqrt((1. - (e * sin_val)**2)**3)
    dist = rc * delta * rad
    return delta / dist

def get_deglon_per_km(alat):
    # Calculate degrees longitude per km
    rad = 0.017453292
    delta = 0.1
    a = 6378.137
    b = 6356.752
    e = np.sqrt(1. - (b/a)**2)
    
    sin_lat = np.sin(alat * rad)
    cos_lat = np.cos(alat * rad)
    rc = (a / np.sqrt(1. - (e * sin_lat)**2)) * cos_lat
    dist = rc * delta * rad
    return delta / dist

def xy2geo(olat, olon, xd, yd, str_deg, dip_deg):
    # Convert local Cartesian to geographic coordinates
    rad = 0.017453292
    
    deglat = get_deglat_per_km(olat)
    deglon = get_deglon_per_km(olat)
    
    dip1 = dip_deg * rad
    str1 = (str_deg - 90.0) * rad
    
    sin_str1 = np.sin(str1)
    cos_str1 = np.cos(str1)
    cos_dip1 = np.cos(dip1)
    
    work1 = xd * cos_str1 + yd * sin_str1 * cos_dip1
    work2 = -xd * sin_str1 + yd * cos_str1 * cos_dip1
    
    wlon = olon + deglon * work1
    wlat = olat + deglat * work2
    
    if wlon > 180.0:
        wlon -= 360.0
    if wlon < -180.0:
        wlon += 360.0
        
    return wlat, wlon

def calculate_point_geometry(lat0, lon0, depth0, strike, dip, x_dist, y_dist):
    # Calculate geometry for a single point
    lat, lon = xy2geo(lat0, lon0, x_dist, y_dist, strike, dip)
    
    d_depth = -y_dist * np.sin(np.radians(dip))
    depth = depth0 + d_depth
    
    return lat, lon, depth

def convert(fort40_path, pdtdis_path, out_pddis):
    # Output columns:
    # 1: Latitude (deg)
    # 2: Longitude (deg)
    # 3: Distance along Strike (km)
    # 4: Distance along Up-Dip (km)
    # 5: Total interpolated slip (m)
    # 6: Depth (km)
    
    ref_lat, ref_lon, ref_depth, strike, dip, dx, dy, mn, nn = load_fort40_params(fort40_path)
    g_slip, min_dip, max_dip, min_str, max_str = load_pdtdis_slip(pdtdis_path, mn, nn)
    
    out_x_start = min_str - dx
    out_x_end = max_str + dx
    out_y_start = min_dip - dy
    out_y_end = max_dip + dy 
    
    step = 0.5
    
    x_coords = np.arange(out_x_start, out_x_end + step/1000.0, step)
    y_coords = np.arange(out_y_start, out_y_end + step/1000.0, step)
    
    src_dip_axis = np.linspace(min_dip, max_dip, nn)
    src_str_axis = np.linspace(min_str, max_str, mn)
    
    pad_dip_axis = np.concatenate(([src_dip_axis[0]-dy], src_dip_axis, [src_dip_axis[-1]+dy]))
    pad_str_axis = np.concatenate(([src_str_axis[0]-dx], src_str_axis, [src_str_axis[-1]+dx]))
    
    pad_slip = np.zeros((nn+2, mn+2))
    pad_slip[1:-1, 1:-1] = g_slip
    
    interp_slip = RegularGridInterpolator((pad_dip_axis, pad_str_axis), pad_slip, bounds_error=False, fill_value=0.0)
    
    with open(out_pddis, 'w') as f:
        for y in y_coords: 
            pts = np.zeros((len(x_coords), 2))
            pts[:, 0] = y
            pts[:, 1] = x_coords
            slips = interp_slip(pts)
            
            for i in range(len(x_coords)):
                x = x_coords[i]
                slip = slips[i]
                
                lat, lon, depth = calculate_point_geometry(ref_lat, ref_lon, ref_depth, strike, dip, x, y)
                
                f.write(f"{lat:9.3f} {lon:8.3f} {x:8.3f} {y:8.3f} {slip:12.6f} {depth:8.3f}\n")

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pdtdis.dat to pddis.dat")
    parser.add_argument("fort40", help="Path to fort.40 file")
    parser.add_argument("pdtdis", help="Path to pdtdis.dat file")
    parser.add_argument("--output", default="pddis.dat", help="Output filename (default: pddis.dat)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.fort40):
        print(f"Error: Input file '{args.fort40}' not found.")
        sys.exit(1)
    if not os.path.exists(args.pdtdis):
        print(f"Error: Input file '{args.pdtdis}' not found.")
        sys.exit(1)
        
    convert(args.fort40, args.pdtdis, args.output)
