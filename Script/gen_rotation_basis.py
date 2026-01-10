#!/usr/bin/env python3
"""
Generate Rotation_basis_DC.dat for PDTI

This script generates 6 basis double-couple component definitions (Strike, Dip, Rake).
It rotates a standard set of basis tensors so that the first component aligns exactly
with a user-provided GCMT solution.

Method:
1. Calculates the P, T, and B (Null) axes analytically from the Strike/Dip/Rake
   angles for both the Reference Basis (0, 90, 0) and the Target GCMT.
   This avoids sign ambiguities inherent in numerical eigenvalue decomposition.
2. Computes a rotation matrix R that aligns the Reference frame to the Target frame.
3. Applies R to all 6 basis moment tensors.
4. Converts the rotated tensors back to Strike/Dip/Rake.
   - Handles the conjugate plane ambiguity by selecting the solution closest
     to the expected target for Basis 1.

Usage:
    python3 generate_rotation_basis.py <Strike> <Dip> <Rake>
"""

import sys
import numpy as np
from pyrocko import moment_tensor as prmt

def sdr2vectors(strike, dip, rake):
    """
    Convert Strike, Dip, Rake to slip vector (u) and fault normal (n).
    
    Coordinate system (NED): x=North, y=East, z=Down.
    Conventions follow Aki & Richards.
    """
    s_rad = np.deg2rad(strike)
    d_rad = np.deg2rad(dip)
    r_rad = np.deg2rad(rake)

    sn_phi = np.sin(s_rad)
    cs_phi = np.cos(s_rad)
    sn_delta = np.sin(d_rad)
    cs_delta = np.cos(d_rad)
    sn_lam = np.sin(r_rad)
    cs_lam = np.cos(r_rad)

    # Fault normal vector n
    n = np.array([
        -sn_delta * sn_phi,
         sn_delta * cs_phi,
        -cs_delta
    ])

    # Slip vector u
    u = np.array([
        cs_lam * cs_phi + cs_delta * sn_lam * sn_phi,
        cs_lam * sn_phi - cs_delta * sn_lam * cs_phi,
        -sn_delta * sn_lam
    ])
    
    return u, n

def get_ptb_axes(strike, dip, rake):
    """
    Calculate P (Pressure), T (Tension), and B (Null) axes vectors deterministically from SDR.
    
    Returns:
        3x3 matrix where columns are [P, T, B].
    
    Note:
        Constructing axes from u and n ensures that the coordinate frame is physically
        tied to the specific definitions of Strike/Dip/Rake, avoiding the random
        sign flips that can occur if using numerical eigenvector decomposition (eigh).
    """
    u, n = sdr2vectors(strike, dip, rake)
    
    p_vec = (n - u) / np.sqrt(2.0)
    t_vec = (n + u) / np.sqrt(2.0)
    b_vec = np.cross(p_vec, t_vec)
    
    # Normalize (for numerical stability)
    p_vec /= np.linalg.norm(p_vec)
    t_vec /= np.linalg.norm(t_vec)
    b_vec /= np.linalg.norm(b_vec)
    
    return np.column_stack((p_vec, t_vec, b_vec))

def get_rotation_matrix(source_sdr, target_sdr):
    """
    Calculate the rotation matrix that transforms the Source frame to the Target frame.
    
    Args:
        source_sdr: Tuple (Strike, Dip, Rake) of the reference basis.
        target_sdr: Tuple (Strike, Dip, Rake) of the target GCMT.
        
    Returns:
        3x3 Rotation Matrix R.
    """
    # Get analytical principal axes frames
    # V = [P, T, B] (orthogonal matrices)
    vecs_s = get_ptb_axes(*source_sdr)
    vecs_t = get_ptb_axes(*target_sdr)

    # Calculate Rotation Matrix R such that R * vecs_s = vecs_t.
    # Since V is orthogonal, V_inv = V_transpose.
    # R = V_target * V_source^T
    R = np.dot(vecs_t, vecs_s.T)
    return R

def rotate_tensor(mt, R):
    """
    Apply rotation R to a Pyrocko MomentTensor object.
    
    Formula: M_new = R * M_old * R^T
    """
    M_old = mt.m() # Get 3x3 matrix representation
    M_new = np.dot(R, np.dot(M_old, R.T))
    return prmt.MomentTensor(m=M_new)

def get_closest_sdr(candidates, target_sdr):
    """
    Select the SDR from a list of candidates that is closest to target_sdr.
    Used to disambiguate conjugate planes.
    
    Args:
        candidates: List of (Strike, Dip, Rake) tuples.
        target_sdr: (Strike, Dip, Rake) tuple to match against.
        
    Returns:
        best_sdr: The candidate closest to target_sdr in Euclidean space.
    """
    ts, td, tr = target_sdr
    best_diff = float('inf')
    best_sdr = candidates[0]
    
    for s, d, r in candidates:
        # Simple Euclidean distance.
        # Note: Does not account for circularity (360=0), but sufficient for
        # distinguishing disparate conjugate planes (e.g., Strike 0 vs 90).
        diff = (s - ts)**2 + (d - td)**2 + (r - tr)**2
        if diff < best_diff:
            best_diff = diff
            best_sdr = (s, d, r)
            
    return best_sdr

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 generate_rotation_basis.py <strike> <dip> <rake>")
        sys.exit(1)

    t_strike = 0.0
    t_dip = 0.0
    t_rake = 0.0

    try:
        t_strike = float(sys.argv[1])
        t_dip = float(sys.argv[2])
        t_rake = float(sys.argv[3])
    except ValueError:
        print("Error: Strike, Dip, and Rake must be numbers.")
        sys.exit(1)

    # 1. Define the 6 Standard Basis Components
    # These correspond to the default hardcoded values in GreenPointSources.f90
    basis_sdr = [
        (0.0, 90.0, 0.0),    # Basis 1: Reference (Target for rotation)
        (135.0, 90.0, 0.0),  # Basis 2
        (180.0, 90.0, 90.0), # Basis 3
        (90.0, 90.0, 90.0),  # Basis 4
        (90.0, 45.0, 90.0),  # Basis 5
        (0.0, 500.0, 0.0)    # Basis 6: Dummy (Isotropic flag)
    ]

    # Convert descriptions to Pyrocko MomentTensors
    basis_mts = []
    for s, d, r in basis_sdr:
        if d > 180: # Handle isotropic dummy case strictly
            basis_mts.append(None)
        else:
            basis_mts.append(prmt.MomentTensor(strike=s, dip=d, rake=r))

    # 2. Compute Rotation Matrix
    # Rotate the coordinate system so that Basis 1 (0, 90, 0)
    # aligns with the User's Input (target).
    R = get_rotation_matrix(basis_sdr[0], (t_strike, t_dip, t_rake))

    # 3. Apply Rotation and Generate Output
    rotated_basis_sdr = []
    
    print(f"Generating basis for Target: Strike={t_strike}, Dip={t_dip}, Rake={t_rake}")
    print("-" * 50)
    
    for i, mt in enumerate(basis_mts):
        if mt is None:
             # Preserve isotropic dummy component exactly as is
             rotated_basis_sdr.append(basis_sdr[i]) 
             print(f"Basis {i+1}: ({basis_sdr[i][0]:.1f}, {basis_sdr[i][1]:.1f}, {basis_sdr[i][2]:.1f}) (Dummy)")
             continue
        
        # Apply rotation
        rotated_mt = rotate_tensor(mt, R)
        
        # Get SDR candidates (Standard and Conjugate plane)
        sdr_pair = rotated_mt.both_strike_dip_rake()
        
        # Select the best representation
        if i == 0:
            # For Basis 1, we MUST pick the plane that matches the user input.
            final_sdr = get_closest_sdr(sdr_pair, (t_strike, t_dip, t_rake))
        else:
            # For others, picking the first solution is sufficient.
            final_sdr = sdr_pair[0]
            
        rotated_basis_sdr.append(final_sdr)
        print(f"Basis {i+1}: ({final_sdr[0]:.1f}, {final_sdr[1]:.1f}, {final_sdr[2]:.1f})")

    # 4. Write output file
    filename = "Rotation_basis_DC.dat"
    with open(filename, "w") as f:
        # Format requirements from GreenPointSources.f90:
        # Line 1: Strikes (space separated)
        # Line 2: Dips
        # Line 3: Rakes
        f.write("  ".join(f"{val[0]:6.1f}" for val in rotated_basis_sdr) + "\n")
        f.write("  ".join(f"{val[1]:6.1f}" for val in rotated_basis_sdr) + "\n")
        f.write("  ".join(f"{val[2]:6.1f}" for val in rotated_basis_sdr) + "\n")

    print("-" * 50)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    main()
