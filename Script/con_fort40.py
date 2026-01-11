import numpy as np
import sys
from pyrocko import moment_tensor as pmt

def _read_grid(lines, start_idx, mn, nn):
    # Helper to read subfault grid data
    grid = np.zeros((nn, mn))
    current_line = start_idx
    
    for n_idx in range(nn):
        vals = []
        while len(vals) < mn and current_line < len(lines):
            line_content = lines[current_line].replace('D', 'E').strip()
            if line_content:
                parts = line_content.split()
                for x in parts:
                    try:
                        vals.append(float(x))
                    except ValueError:
                        pass
            current_line += 1
        
        if len(vals) >= mn:
            grid[nn - 1 - n_idx, :] = vals[:mn]
            
    return grid, current_line

def load_fort40_full(infile):
    # Parse fort.40 file
    with open(infile, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 6:
        raise ValueError("File too short (less than 6 lines).")
    
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
    m0 = int(parts[4])
    n0 = int(parts[5])
    icmn = int(parts[8])
    
    vectors = np.zeros((icmn, nn, mn))
    total_slip = np.zeros((nn, mn))
    
    # Parse Slip Vectors
    idx_start = -1
    for i, line in enumerate(lines):
        if "Total Slip" in line and "Vec" in line:
            idx_start = i
            break
            
    if idx_start != -1:
        current_line = idx_start + 1
        for ic in range(icmn):
            while current_line < len(lines):
                if "Vector" in lines[current_line] or "vector" in lines[current_line]:
                    current_line += 1
                    break
                current_line += 1
            
            grid, current_line = _read_grid(lines, current_line, mn, nn)
            vectors[ic, :, :] = grid

    # Parse Total Slip
    idx_start = -1
    for i, line in enumerate(lines):
        if "Total Slip for each sub-fault" in line:
            idx_start = i
            break
            
    if idx_start != -1:
         current_line = idx_start + 1
         while current_line < len(lines):
             if lines[current_line].strip() and "Slip" not in lines[current_line]:
                 break
             current_line += 1
                 
         total_slip, _ = _read_grid(lines, current_line, mn, nn)
            
    return {
        'ref_lat': ref_lat, 'ref_lon': ref_lon, 'ref_depth': ref_depth,
        'strike': strike_model, 'dip': dip_model,
        'dx': dx, 'dy': dy,
        'mn': mn, 'nn': nn, 'm0': m0, 'n0': n0,
        'icmn': icmn,
        'vectors': vectors,
        'total_slip': total_slip
    }

# Reference Fortran Code:
    # sub.c_location_lib.f90: - calculate the location of the space knot point

def get_deglat_per_km(alat):
    # Calculate degrees latitude per km
    rad = np.radians(1.0)
    delta = 0.1
    a = 6378.137
    b = 6356.752
    e = np.sqrt(1. - (b/a)**2)
    
    sin_val = np.sin(np.radians(alat + delta/2.0))
    rc = a * (1. - e**2) / np.sqrt((1. - (e * sin_val)**2)**3)
    dist = rc * delta * rad
    return delta / dist

def get_deglon_per_km(alat):
    # Calculate degrees longitude per km
    rad = np.radians(1.0)
    delta = 0.1
    a = 6378.137
    b = 6356.752
    e = np.sqrt(1. - (b/a)**2)
    
    sin_lat = np.sin(np.radians(alat))
    cos_lat = np.cos(np.radians(alat))
    rc = (a / np.sqrt(1. - (e * sin_lat)**2)) * cos_lat
    dist = rc * delta * rad
    return delta / dist

def xy2geo(olat, olon, xd, yd, str_deg, dip_deg):
    # Convert local Cartesian to geographic coordinates
    deglat = get_deglat_per_km(olat)
    deglon = get_deglon_per_km(olat)
    
    dip_rad = np.radians(dip_deg)
    str_rad = np.radians(str_deg - 90.0)
    
    sin_str = np.sin(str_rad)
    cos_str = np.cos(str_rad)
    cos_dip = np.cos(dip_rad)
    
    work1 = xd * cos_str + yd * sin_str * cos_dip
    work2 = -xd * sin_str + yd * cos_str * cos_dip
    
    wlon = olon + deglon * work1
    wlat = olat + deglat * work2
    
    if wlon > 180.0: wlon -= 360.0
    if wlon < -180.0: wlon += 360.0
        
    return wlat, wlon

def calculate_subfault_geometry(lat0, lon0, depth0, strike, dip, xx, yy, m, n, m0, n0):
    # Calculate subfault geometry
    xd = (m - m0) * xx 
    yd = (n - n0) * yy
    
    lat, lon = xy2geo(lat0, lon0, xd, yd, strike, dip)
    
    d_depth = -yd * np.sin(np.radians(dip))
    depth = depth0 + d_depth
    
    return lat, lon, depth, yd, xd

def get_basis_mt(strike_b, dip_b, rake_b):
    # Calculate basis moment tensor components
    s = np.radians(strike_b)
    d = np.radians(dip_b)
    r = np.radians(rake_b)
    
    sin_d = np.sin(d); cos_d = np.cos(d)
    sin_r = np.sin(r); cos_r = np.cos(r)
    sin_s = np.sin(s); cos_s = np.cos(s)
    sin_2s = np.sin(2*s); cos_2s = np.cos(2*s)
    sin_2d = np.sin(2*d); cos_2d = np.cos(2*d)

    m_nn = -(sin_d*cos_r*sin_2s + sin_2d*sin_r*sin_s**2)
    m_ee =  (sin_d*cos_r*sin_2s - sin_2d*sin_r*cos_s**2)
    m_dd =  (sin_2d*sin_r)
    m_ne =  (sin_d*cos_r*cos_2s + 0.5*sin_2d*sin_r*sin_2s)
    m_nd = -(cos_d*cos_r*cos_s + cos_2d*sin_r*sin_s)
    m_ed = -(cos_d*cos_r*sin_s - cos_2d*sin_r*cos_s)
    
    return np.array([m_dd, m_nn, m_ee, m_nd, -m_ed, -m_ne])

def read_rotation_basis(filepath):
    """Load 5 elementary basis (strike, dip, rake) from file or use STANDARD basis."""
    if filepath is None or filepath == "STANDARD":
        # Kikuchi/PDTI default basis
        return [
            (0., 90., 0.),
            (135., 90., 0.),
            (180., 90., 90.),
            (90., 90., 90.),
            (90., 45., 90.)
        ]
    
    data = np.loadtxt(filepath)
    # Rotation_basis_DC.dat: 3×5 matrix, rows = (strike, dip, rake)
    return [(data[0, i], data[1, i], data[2, i]) for i in range(5)]

def vec_to_azi_plunge_lower(vec):
    # Vector to Azimuth/Plunge (lower hemisphere)
    n, e, d = vec
    
    if d < 0:
        n, e, d = -n, -e, -d
        
    pl = np.degrees(np.arcsin(d))
    az = np.degrees(np.arctan2(e, n))
    if az < 0: az += 360
    
    return az, pl

def convert(fort40_path, out_pdtdis, rotation_basis_file=None):
    data = load_fort40_full(fort40_path)
    
    basis_defs = read_rotation_basis(rotation_basis_file)
    basis_mts = np.array([get_basis_mt(s, d, r) for s, d, r in basis_defs])
    
    # Output columns:
    # 1: Row Index (n)
    # 2: Col Index (m)
    # 3: Up-dip Distance (km)
    # 4: Along-strike Distance (km)
    # 5: Latitude (deg)
    # 6: Longitude (deg)
    # 7: Depth (km)
    # 8: Total Slip (m)
    # 9-14: Moment Tensor Components (M_dd, M_nn, M_ee, M_nd, -M_ed, -M_ne)
    # 15: Constant Placeholder (20)
    # 16-17: Strike 1, Strike 2 (deg)
    # 18-19: Dip 1, Dip 2 (deg)
    # 20-21: Rake 1, Rake 2 (deg)
    # 22-24: P, T, B Axis Azimuth (deg)
    # 25-27: P, T, B Axis Plunge (deg)
    # 28: CLVD Percentage (%)
    
    with open(out_pdtdis, 'w') as f:
        for n in range(data['nn']):
             for m in range(data['mn']):
                 
                 lat, lon, depth, up_dip_dist, strike_dist = calculate_subfault_geometry(
                     data['ref_lat'], data['ref_lon'], data['ref_depth'],
                     data['strike'], data['dip'],
                     data['dx'], data['dy'],
                     m+1, n+1, data['m0'], data['n0']
                 )
                 
                 coeffs = data['vectors'][:, n, m]
                 slip_val = data['total_slip'][n, m]
                 mt_sum = np.zeros(6)
                 for k in range(min(5, data['icmn'])):
                     mt_sum += coeffs[k] * basis_mts[k]
                 
                 mt_mat = pmt.symmat6(*mt_sum)
                 mt = pmt.MomentTensor(m_up_south_east=mt_mat)
                 planes = mt.both_strike_dip_rake()
                 
                 p1, p2 = sorted(planes, key=lambda x: x[1])
                 
                 t_vec = mt.t_axis()
                 p_vec = mt.p_axis()
                 b_vec = mt.null_axis()
                 if b_vec is None: b_vec = np.cross(p_vec, t_vec)
                 
                 t_az, t_pl = vec_to_azi_plunge_lower(t_vec)
                 p_az, p_pl = vec_to_azi_plunge_lower(p_vec)
                 b_az, b_pl = vec_to_azi_plunge_lower(b_vec)
                 
                 m_devi = mt.deviatoric().m6_up_south_east()
                 evals = np.linalg.eigvalsh(pmt.symmat6(*m_devi))
                 abs_evals = np.abs(evals)
                 sort_idx = np.argsort(abs_evals)
                 
                 e_min_abs = evals[sort_idx[0]]
                 e_max_abs = evals[sort_idx[2]]
                 
                 if abs(e_max_abs) > 1e-16:
                     clvd_perc = 2.0 * (abs(e_min_abs) / abs(e_max_abs)) * 100.0
                     if e_max_abs < 0: clvd_perc = -clvd_perc
                 else:
                     clvd_perc = 0.0
                 
                 line_data = [
                      n+1, m+1, up_dip_dist, strike_dist, lat, lon, depth, slip_val,
                      mt_sum[0], mt_sum[1], mt_sum[2], mt_sum[3], mt_sum[4], mt_sum[5],
                      20,
                      p1[0], p2[0], p1[1], p2[1], p1[2], p2[2],
                      p_az, t_az, b_az, p_pl, t_pl, b_pl,
                      clvd_perc
                 ]
                 
                 fmt = ("%4d %4d %10.4f %10.4f %10.5f %10.5f %10.3f %10.4f "
                        "%12.5e %12.5e %12.5e %12.5e %12.5e %12.5e "
                        "%4d "
                        "%7.2f %7.2f %7.2f %7.2f %7.2f %7.2f "
                        "%7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f")
                 f.write(fmt % tuple(line_data) + "\n")

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert fort.40 to pdtdis.dat")
    parser.add_argument("fort40", help="Path to fort.40 file")
    parser.add_argument("--output", "-o", default="pdtdis.dat", help="Output filename (default: pdtdis.dat)")
    parser.add_argument("--rotation-basis", "-r", dest="rotation_basis", default=None,
                        help="Path to Rotation_basis_DC.dat. If omitted, STANDARD basis is used.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.fort40):
        print(f"Error: Input file '{args.fort40}' not found.")
        sys.exit(1)
        
    convert(args.fort40, args.output, args.rotation_basis)
