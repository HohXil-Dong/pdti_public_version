#!/usr/bin/env python3
"""
gen_mech_gmt.py: Generate GMT meca input files from PDTI inversion results.

Reads fort.40 coefficients, combines with rotation basis, and outputs
full moment tensor (.dat) and best DC (.dat_dc) for GMT psmeca.
"""
import sys
import os
import numpy as np
from pyrocko import moment_tensor as prmt


def read_rotation_basis(filepath):
    """Load 5 elementary basis moment tensors from file or use Rotation_basis_DC."""
    if filepath == "STANDARD":
        # Kikuchi/PDTI default basis (strike, dip, rake)
        params = [(0., 90., 0.), (135., 90., 0.), (180., 90., 90.),
                  (90., 90., 90.), (90., 45., 90.)]
        return [prmt.MomentTensor(strike=s, dip=d, rake=r) for s, d, r in params]

    try:
        data = np.loadtxt(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        sys.exit(1)
    
    return [prmt.MomentTensor(strike=data[0, i], dip=data[1, i], rake=data[2, i]) 
            for i in range(5)]


def read_knot_dat(knot_file):
    """Parse subfault geometry from knot.dat_in."""
    geometry = {}
    with open(knot_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                n, m = int(parts[0]), int(parts[1])
                geometry[(m, n)] = {
                    'lat': float(parts[2]),
                    'lon': float(parts[3]),
                    'depth': float(parts[4])
                }
            except ValueError:
                continue
    return geometry


def read_fort40_header_info(lines):
    """Extract grid dimensions (mn, nn) from fort.40 header (line 6)."""
    try:
        l6 = lines[5].strip().split()
        return int(l6[2]), int(l6[3])
    except (IndexError, ValueError):
        print("Error: Failed to parse header from fort.40")
        sys.exit(1)


def parse_block_data(lines, start_idx, mn, nn):
    """Parse a block of matrix data starting at given line index."""
    matrix = np.zeros((mn, nn))
    current_line = start_idx
    for n_idx in range(nn - 1, -1, -1):
        vals = []
        while len(vals) < mn:
            if current_line >= len(lines):
                print("Error: Unexpected EOF while parsing block data")
                sys.exit(1)
            vals.extend([float(x) for x in lines[current_line].strip().split()])
            current_line += 1
        matrix[:, n_idx] = vals[:mn]
    return matrix, current_line


def get_coefficients_from_fort40(lines, mn, nn):
    """Extract 5-component weighting coefficients from fort.40."""
    coeffs = np.zeros((mn, nn, 5))
    for i, line in enumerate(lines):
        if "Total Slip Vecoter for each sub-fault" in line:
            current_idx = i + 1
            for vec_idx in range(5):
                while current_idx < len(lines) and "Vector" not in lines[current_idx]:
                    current_idx += 1
                if current_idx >= len(lines):
                    break
                mat, next_idx = parse_block_data(lines, current_idx + 1, mn, nn)
                coeffs[:, :, vec_idx] = mat
                current_idx = next_idx
            return coeffs
    
    print("Error: Could not find coefficients in fort.40")
    sys.exit(1)


def ned_to_gmt_use(m_ned):
    """
    Convert NED moment tensor matrix to GMT USE convention.
    
    NED: (N, E, D) -> GMT USE: (Up, South, East)
    Mrr=Mdd, Mtt=Mnn, Mff=Mee, Mrt=Mnd, Mrf=-Med, Mtf=-Mne
    """
    return [m_ned[2, 2], m_ned[0, 0], m_ned[1, 1],  # Mrr, Mtt, Mff
            m_ned[0, 2], -m_ned[1, 2], -m_ned[0, 1]]  # Mrt, Mrf, Mtf


def generate_mech_file(knot_file, fort40_file, basis_file, output_file):
    """
    Main routine: generate GMT mechanism files from inversion results.
    
    Scaling: Map [0, max_moment] linearly to [0, TARGET_MW_MAX=8.0] for visualization.
    """
    geometry = read_knot_dat(knot_file)
    basis_mts = read_rotation_basis(basis_file)
    
    with open(fort40_file, 'r') as f:
        lines = f.readlines()
    
    mn, nn = read_fort40_header_info(lines)
    coeffs = get_coefficients_from_fort40(lines, mn, nn)
    
    # Compute moment tensors and find max moment
    tensors_data = []
    max_moment = 0.0
    
    for m in range(1, mn + 1):
        for n in range(1, nn + 1):
            if (m, n) not in geometry:
                continue
            
            m_accum = sum(basis_mts[k].m() * coeffs[m-1, n-1, k] for k in range(5))
            mt_final = prmt.MomentTensor(m=m_accum)
            moment = mt_final.moment
            
            if moment == 0:
                continue
            
            max_moment = max(max_moment, moment)
            tensors_data.append({
                'm': m, 'n': n, 'moment': moment,
                'mt_obj': mt_final, 'geo': geometry[(m, n)]
            })

    if max_moment == 0:
        print("Warning: No non-zero moments found.")
        return

    # Scaling: linear mapping to fake Mw 
    TARGET_MW_MAX = 8.0
    
    # --- Full tensor output (-Sm format) ---
    fmt_geo, fmt_depth = "{:<12.5f}", "{:<10.2f}"
    fmt_comp, fmt_exp = "{:<12.5f}", "{:<5d}"
    
    with open(output_file, 'w') as out:
        for item in tensors_data:
            moment, mt, geo = item['moment'], item['mt_obj'], item['geo']
            
            m_norm_ned = mt.m() / moment
            mw_fake = max((moment / max_moment) * TARGET_MW_MAX, 0.1)
            
            # GMT Mw-M0 relation: log10(M0_dyne) = 1.5 * (Mw + 10.7)
            power = (mw_fake + 10.7) * 1.5
            exponent = int(np.floor(power))
            mantissa = 10 ** (power - exponent)
            
            m_gmt_scaled = [x * mantissa for x in ned_to_gmt_use(m_norm_ned)]
            vals_str = " ".join([fmt_comp.format(v) for v in m_gmt_scaled])
            
            out.write(f"{fmt_geo.format(geo['lon'])} {fmt_geo.format(geo['lat'])} "
                      f"{fmt_depth.format(geo['depth'])} {vals_str} {fmt_exp.format(exponent)} "
                      f"{fmt_geo.format(geo['lon'])} {fmt_geo.format(geo['lat'])}\n")
    
    print(f"Generated {output_file}")

    # --- DC output (-Sd format) ---
    dc_output_file = output_file.replace('.dat', '_dc.dat')
    if dc_output_file == output_file:
        dc_output_file += '_dc.dat'
    
    fmt_angle, fmt_mag = "{:<8.1f}", "{:<8.3f}"
    
    with open(dc_output_file, 'w') as out_dc:
        for item in tensors_data:
            moment, mt, geo = item['moment'], item['mt_obj'], item['geo']
            s1, d1, r1 = mt.both_strike_dip_rake()[0]
            mw_fake = max((moment / max_moment) * TARGET_MW_MAX, 0.1)
            
            out_dc.write(f"{fmt_geo.format(geo['lon'])} {fmt_geo.format(geo['lat'])} "
                         f"{fmt_depth.format(geo['depth'])} {fmt_angle.format(s1)} "
                         f"{fmt_angle.format(d1)} {fmt_angle.format(r1)} "
                         f"{fmt_mag.format(mw_fake)} {fmt_geo.format(geo['lon'])} "
                         f"{fmt_geo.format(geo['lat'])}\n")
    
    print(f"Generated {dc_output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 generate_gmt_mech.py <knot.dat_in> <fort.40> <meca.dat> [Rotation_basis_DC.dat]")
        print("       If Rotation_basis_DC.dat is omitted, STANDARD basis is used.")
        sys.exit(1)
    
    knot_f = sys.argv[1]
    fort_f = sys.argv[2]
    out_f = sys.argv[3]
    basis_f = sys.argv[4] if len(sys.argv) > 4 else "STANDARD"
    
    generate_mech_file(knot_f, fort_f, basis_f, out_f)
