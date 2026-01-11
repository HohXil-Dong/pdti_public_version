import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
import argparse

def parse_fort40(filename):
    """
    Parses the fort.40 file to extract necessary parameters and rupture coefficients.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 1. Header Information
    # Line 2: Moment, Mw, Rigid, Lat, Lon, Depth, Vr, nsurface
    parts = lines[1].split()
    Mw = float(parts[1])
    
    # Line 6: xx, yy, mn, nn, m0, n0, Raisetime, jtn, icmn
    parts = lines[5].split()
    dx = float(parts[0]) # km
    dy = float(parts[1]) # km
    mn = int(parts[2])
    nn = int(parts[3])
    m0 = int(parts[4])
    n0 = int(parts[5])
    rise_time = float(parts[6])
    jtn = int(parts[7])
    icmn = int(parts[8])
    
    print(f"Model Params: Mw={Mw}, dx={dx}km, dy={dy}km")
    print(f"Grid: mn={mn}, nn={nn}, m0={m0}, n0={n0}, RiseTime={rise_time}s")

    # Locate sections
    start_time_idx = -1
    jt_icm_start_idx = -1
    
    for i, line in enumerate(lines):
        if "Start Time for each sub-fault" in line:
            start_time_idx = i + 1
        if "JT, ICM :" in line and jt_icm_start_idx == -1:
            jt_icm_start_idx = i
            
    # 2. Extract Start Times (Tr)
    Tr = np.zeros((mn, nn))
    if start_time_idx != -1:
        current_line = start_time_idx
        for n_idx in range(nn): # n from nn down to 1
            vals = [float(x) for x in lines[current_line].split()]
            n = nn - 1 - n_idx
            Tr[:, n] = vals
            current_line += 1
    else:
        raise ValueError("Start Time block not found in fort.40")

    # 3. Extract PDT_rate coefficients
    coeffs = np.zeros((mn, nn, icmn, jtn))
    current_line = jt_icm_start_idx
    while current_line < len(lines):
        line = lines[current_line].strip()
        if not line.startswith("JT, ICM :"):
            current_line += 1
            continue
            
        try:
            parts = line.split(':')
            indices = parts[1].split(',')
            jt_val = int(indices[0])
            icm_val = int(indices[1])
        except IndexError: 
             break
             
        current_line += 1 # Go to data
        for n_idx in range(nn):
            n = nn - 1 - n_idx
            vals = []
            while len(vals) < mn:
                vals.extend([float(x) for x in lines[current_line].split()])
                current_line += 1
            coeffs[:, n, icm_val-1, jt_val-1] = vals[:mn] # 0-indexed
            
    return ((Mw, dx, dy, rise_time, m0, n0), Tr, coeffs)


def basis_function(t, rise_time):
    """
    Triangular basis function for source time function.
    
    Shape: Isosceles triangle.
    Duration: 2 * rise_time.
    Peak Time: rise_time.
    Peak Value: 1.0 / rise_time.
    Integral (Area): 1.0.
    
    Args:
        t (np.array): Time array relative to the start of the basis function.
        rise_time (float): Rise time parameter (half-duration).
        
    Returns:
        np.array: Amplitude of the basis function at times t.
    """
    res = np.zeros_like(t)
    # Rising phase: 0 to rise_time
    mask1 = (t >= 0) & (t < rise_time)
    res[mask1] = t[mask1] / rise_time
    # Falling phase: rise_time to 2*rise_time
    mask2 = (t >= rise_time) & (t < 2 * rise_time)
    res[mask2] = 2.0 - t[mask2] / rise_time
    
    # Scale by 1/rise_time to ensure Unit Area
    return res / rise_time 

import os
import sys

def main():

    parser = argparse.ArgumentParser(description="Parse and visualize rupture spacetime from fort.40.")
    parser.add_argument("fort40", type=str, help="Path to the fort.40 input file")
    parser.add_argument("--output", default="rupture_history.png", help="Output filename (default: rupture_history.png)")
    
    args = parser.parse_args()
    fort_file = args.fort40
    
    if not os.path.exists(fort_file):
        print(f"Error: Input file '{fort_file}' not found.")
        sys.exit(1)
    
    # 1. Parse Data
    params, Tr, coeffs = parse_fort40(fort_file)
    Mw, dx, dy, rise_time, m0, n0 = params
    mn, nn, icmn, jtn = coeffs.shape
    
    # 2. Setup Time and Distance Grid
    # Calculate on the physical grid defined by rise_time and subfault size.
    dt_calc = rise_time
    t_max = np.max(Tr) + jtn * rise_time
    t_grid = np.arange(0, t_max, dt_calc)
    dist_grid = (np.arange(mn) - (m0 - 1)) * dx # Distance in km relative to epicenter
    
    # 3. Calculate Potency Rate Distribution
    # Accumulate moment rate from all subfaults overlapping in time.
    
    coeffs_unscaled = coeffs 
    potency_rate_mn_t = np.zeros((mn, len(t_grid)))
    
    for m in range(mn):
        col_rate = np.zeros_like(t_grid)
        for n in range(nn):
            t0 = Tr[m, n]
            amp = 1.0 
            
            subfault_tensor_rate = np.zeros((len(t_grid), 6))
            
            for jt in range(jtn):
                t_shift = t0 + (jt) * rise_time
                coeffs_vec = coeffs_unscaled[m, n, :, jt]
                
                # Transformation(from src/KIKUCHI/sub.focalM_lib.f90 subroutine Mtrx):
                # M11, M22, M33, M12, M13, M23
                # v0=NE, v1=NN, v2=ED, v3=ND, v4=DD
                M_vec = np.zeros(6)
                v = coeffs_vec
                M_vec[0] = v[1] - v[4] # M11
                M_vec[1] = -v[1]       # M22
                M_vec[2] = v[4]        # M33
                M_vec[3] = v[0]        # M12
                M_vec[4] = v[3]        # M13
                M_vec[5] = v[2]        # M23
                
                # Basis function addition
                t_start_basis = t_shift
                t_end_basis = t_shift + 2 * rise_time
                
                # Vectorized check for overlap with t_grid
                # t_grid points within [t_shift, t_shift + 2*rise_time]
                # To be efficient and precise with coarse grid:
                mask = (t_grid >= t_start_basis) & (t_grid <= t_end_basis)
                if np.any(mask):
                    basis = basis_function(t_grid[mask] - t_shift, rise_time)
                    subfault_tensor_rate[mask, :] += np.outer(basis, M_vec)
                    
            # Calculate Scalar Rate from the Summed Tensor Rate using Eigenvalues (Best Double Couple Moment)
            # Based on src/KIKUCHI/sub.focalM_lib.f90 subroutine d_cp.
            # The scalar moment M0 is defined as the average of the major and minor deviatoric eigenvalues:
            # M0 = (lambda_1 - lambda_3) / 2.0  (where lambda_1 >= lambda_2 >= lambda_3)
            M_t = np.zeros((len(t_grid), 3, 3))
            rates = subfault_tensor_rate
            M_t[:, 0, 0] = rates[:, 0]
            M_t[:, 1, 1] = rates[:, 1]
            M_t[:, 2, 2] = rates[:, 2]
            M_t[:, 0, 1] = M_t[:, 1, 0] = rates[:, 3]
            M_t[:, 0, 2] = M_t[:, 2, 0] = rates[:, 4]
            M_t[:, 1, 2] = M_t[:, 2, 1] = rates[:, 5]
            
            # Calculate eigenvalues
            eigvals = np.linalg.eigvalsh(M_t) 
            
            # Best Double Couple Approximation: M0 = (lambda_max - lambda_min) / 2
            scalar_vals = (eigvals[:, -1] - eigvals[:, 0]) / 2.0
            
            col_rate += scalar_vals * amp
            
        potency_rate_mn_t[m, :] = col_rate

    # 5. Plotting with interpolation
    
    # Average slip rate along dip direction
    slip_rate_m_t = potency_rate_mn_t / nn
    
    # Interpolation Strategy:
    # 1. Zero-pad the spatial boundaries to prevent edge artifacts.
    # 2. Use Linear Interpolation (RegularGridInterpolator).
    
    # Pad Spatial Dimension (Distance)
    pad_mn = mn + 2
    slip_pad = np.zeros((pad_mn, len(t_grid)))
    slip_pad[1:-1, :] = slip_rate_m_t
    
    # Update Distance Grid with padding
    dx_km = dx
    dist_pad = np.concatenate(([dist_grid[0] - dx_km], dist_grid, [dist_grid[-1] + dx_km]))
    
    x = dist_pad
    y = t_grid
    z = slip_pad
    
    # Perform Linear Interpolation
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((x, y), z, bounds_error=False, fill_value=0)
    
    # Generate Fine Grid for Visualization
    x_fine = np.linspace(x.min(), x.max(), 500)
    y_fine = np.linspace(y.min(), y.max(), 500)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    
    # Evaluate Interpolator
    pts = np.array([X_fine.ravel(), Y_fine.ravel()]).T
    Z_fine_flat = interp(pts)
    Z_fine = Z_fine_flat.reshape(X_fine.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = cm.bilbao_r
    
    # 1. Filled Contours
    max_val = Z_fine.max()
    levels_f = np.linspace(0, max_val, 100)
    cf = ax.contourf(X_fine, Y_fine, Z_fine, levels=levels_f, cmap=cmap, extend='max')
    
    # 2. Line Contours (0.02 m/s interval)
    contour_step = 0.02
    if max_val > contour_step:
        levels_l = np.arange(contour_step, max_val + contour_step/2.0, contour_step)
        cs = ax.contour(X_fine, Y_fine, Z_fine, levels=levels_l, colors='k', linewidths=0.5)
    
    # 3. Colorbar (Inset bottom left)
    ax_cb = ax.inset_axes([0.05, 0.1, 0.3, 0.02])
    cbar = plt.colorbar(cf, cax=ax_cb, orientation='horizontal')
    cbar.set_label('Slip rate (m/s)', fontsize=9, labelpad=-35, y=0.5)
    
    # Colorbar Ticks: Start, Middle, End
    ticks = [0, max_val/2.0, max_val]
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([f'{t:.2f}' for t in ticks], fontsize=8)
    
    # 4. Epicenter (Star at Origin)
    ax.scatter(0, 0, s=400, marker='*', facecolor='None', edgecolor='k', linewidth=2, zorder=100, clip_on=False)
    
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Time (s)')
    ax.set_title(f'Rupture Process (Mw {Mw})')
    ax.invert_xaxis()
    ax.set_ylim(0, t_max)
    
    outname = args.output
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {outname}")


if __name__ == "__main__":
    main()
