
import numpy as np
import os
import argparse
import sys
import matplotlib.pyplot as plt

class PDTI_MRF_Generator:
    def __init__(self, fort40_path, rigid_path, output_path):
        self.fort40_path = fort40_path
        self.rigid_path = rigid_path
        self.output_path = output_path
        
        # Parameters to be read from fort.40
        self.mn = 0
        self.nn = 0
        self.jtn = 0
        self.icmn = 0
        self.rtime = 0.0
        self.xx = 0.0
        self.yy = 0.0
        self.mw = 0.0
        self.moment = 0.0
        
        # Data containers
        self.start_times = None 
        self.coeff_data = None 
        self.rigidity = None    
        self.amplification = None
        
    def read_fort40(self):
        """
        Reads fort.40 to extract:
        1. Subfault grid parameters (mn, nn, xx, yy).
        2. Reference Time (rtime) and number of time knots (jtn).
        3. Rupture Start Time (Tr) for each subfault.
        4. Coefficient data for the 5-component basis at each time knot.
        """
        print(f"Reading {self.fort40_path}...")
        with open(self.fort40_path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
            
        # Parse Line 2: Scalar Moment, Mw, Reference Rigidity, Hypocenter (Lat, Lon, Depth), Rupture Velocity
        parts = lines[1].split()
        self.moment = float(parts[0])
        self.mw = float(parts[1])
        
        # Parse Line 6: Subfault dimensions (xx, yy), Grid Size (mn, nn), Raisetime (rtime), Time Knots (jtn), Basis Components (icmn)
        parts = lines[5].split()
        self.xx = float(parts[0])
        self.yy = float(parts[1])
        self.mn = int(parts[2])
        self.nn = int(parts[3])
        self.rtime = float(parts[6])
        self.jtn = int(parts[7])
        self.icmn = int(parts[8])
        
        print(f"Grid: mn={self.mn}, nn={self.nn}, jtn={self.jtn}, icmn={self.icmn}, rtime={self.rtime}")
        print(f"Subfault size: {self.xx} x {self.yy} km")

        # Initialize data arrays
        # fort.40 blocks are ordered from n=nn (Shallowest) down to n=1 (Deepest).
        # store them in 0-based arrays where index 0 corresponds to n=1 (Deepest) and index nn-1 to n=nn.
        self.start_times = np.zeros((self.nn, self.mn))
        self.coeff_data = np.zeros((self.nn, self.mn, self.icmn, self.jtn))
        
        # Locate "Start Time for each sub-fault" block
        st_idx = -1
        for i, line in enumerate(lines):
            if "Start Time for each sub-fault" in line:
                st_idx = i + 1
                break
        
        if st_idx == -1:
            raise ValueError("Start Time block not found in fort.40")
            
        # Read Start Times
        # File Order: n=nn, n=nn-1, ..., n=1
        # File Row i -> Internal Row n_idx = nn - 1 - i
        for i in range(self.nn):
            row_vals = []
            current_line_idx = st_idx
            # Consumes multiple lines if mn exceeds standard line width
            while len(row_vals) < self.mn:
                row_vals.extend([float(x) for x in lines[current_line_idx].split()])
                current_line_idx += 1
            
            n_idx = self.nn - 1 - i
            self.start_times[n_idx, :] = row_vals[:self.mn]
            st_idx = current_line_idx 

        # Parse Coefficients Block "JT, ICM : jt , icm"
        # Structure: Loop over Time (jt) -> Loop over Component (icm) -> Subfault Grid
        for i, line in enumerate(lines):
            if line.startswith("JT, ICM :"):
                # Parse indices jt and icm
                parts = line.replace("JT, ICM :", "").replace(",", "").split()
                jt = int(parts[0])
                icm = int(parts[1])
                
                # Read grid data for this (jt, icm) pair
                block_start = i + 1
                current_line_idx = block_start
                for row_k in range(self.nn):
                    row_vals = []
                    while len(row_vals) < self.mn:
                        row_vals.extend([float(x) for x in lines[current_line_idx].split()])
                        current_line_idx += 1
                    
                    n_idx = self.nn - 1 - row_k
                    self.coeff_data[n_idx, :, icm-1, jt-1] = row_vals[:self.mn]

    def read_rigid(self):
        """
        Reads rigid_amp.info to extract Rigidity (GPa) and Amplification Factors.
        Format: n m rigidity amplification depth
        Indices n and m are 1-based.
        """
        print(f"Reading {self.rigid_path}...")
        self.rigidity = np.zeros((self.nn, self.mn))
        self.amplification = np.zeros((self.nn, self.mn))
        
        with open(self.rigid_path, 'r') as f:
            lines = f.readlines()
            
        count = 0
        for line in lines:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 4: continue
            
            # Format: n, m, rigid, amp, depth
            n = int(parts[0])
            m = int(parts[1])
            rigid = float(parts[2])
            amp = float(parts[3])
            
            # Store values into 0-based arrays
            if 1 <= n <= self.nn and 1 <= m <= self.mn:
                self.rigidity[n-1, m-1] = rigid
                self.amplification[n-1, m-1] = amp
                count += 1
                
        print(f"Loaded rigidity for {count} subfaults.")

    def mtrx_transform(self, v):
        """
        Transforms the 5-component basis vector v to the full 6-component deviatoric moment tensor M.
        Coordinate System: North-East-Down (NED).
        Transformation Logic (from src/KIKUCHI/sub.focalM_lib.f90 subroutine Mtrx):
        M11 (NN) = v(2) - v(5)
        M12 (NE) = v(1)
        M13 (ND) = v(4)
        M22 (EE) = -v(2)
        M23 (ED) = v(3)
        M33 (DD) = v(5)
        """
        M = np.zeros(6)
        M[0] = v[1] - v[4] # M11 (North-North)
        M[1] = -v[1]       # M22 (East-East)
        M[2] = v[4]        # M33 (Down-Down)
        M[3] = v[0]        # M12 (North-East)
        M[4] = v[3]        # M13 (North-Down)
        M[5] = v[2]        # M23 (East-Down)
        return M

    def triangle_func(self, t, center_t, half_width):
        """
        Triangular basis function (Linear B-Spline of Order 2).
        (from src/sub.c_wave_lib.f90 subroutine stime_m)
        Defined as:
            T(t) = (1 - |t - center| / half_width) / half_width, for |t - center| < half_width
            T(t) = 0, otherwise.
        
        This function represents the temporal shape of the Moment Rate for a single time knot.
        The Area of this triangle is intended to be 1.0, so the peak height is 1/half_width.
        """
        dt_sample = abs(t - center_t)
        val = np.maximum(0.0, 1.0 - dt_sample / half_width)
        return val / half_width

    def generate_mrf(self):
        if self.start_times is None or self.coeff_data is None or self.rigidity is None or self.amplification is None:
            raise ValueError("Data not loaded. valid start_times, coeff_data, rigidity, and amplification are required.")

        # Time sampling
        # cover from 0 to max_start_time + jtn*rtime 
        max_t = np.max(self.start_times) + self.jtn * self.rtime 
        dt = self.rtime # Yagi's sampling (1.1s)
        
        time_points = np.arange(0.0, max_t, dt)
        num_steps = len(time_points)
        
        # M_stats: (num_steps, 6) -> M11, M22, M33, M12, M13, M23
        M_rate = np.zeros((num_steps, 6))
        scalar_rate = np.zeros(num_steps)
        
        # Area factor: xx * yy * 1e15 (m^2 * Pa)
        scaling_factor = self.xx * self.yy * 1.e15
        
        print(f"Calculating MRF on {num_steps} time steps...")
        
        for n in range(self.nn):
            for m in range(self.mn):
                mu = self.rigidity[n, m]
                amp = self.amplification[n, m]
                t0 = self.start_times[n, m]
                
                # Sum Moment Rate Contributions:
                # Rate(t) = Sum_{m,n,jt} [ Basis_Vector(coeffs) * Rigidity * Amplification * Logic_Area * Triangle_Basis(t) ]
                # Scaling Factor (Unit Conversion) = xx(km) * yy(km) * 1e15 -> converts GPa * km^2 to Nm.
                
                m_factor = mu * amp * scaling_factor
                
                for jt in range(self.jtn):
                    # Shift by one rtime to start causal triangle at t0
                    # jt=0: center at t0 + rtime. Non-zero from t0 to t0+2*rtime.
                    center_time = t0 + (jt + 1) * self.rtime
                    
                    # Get coefficients vector (5-component basis)
                    coeffs = self.coeff_data[n, m, :, jt]
                    
                    # Convert to tensor components
                    # M_ij for this triangle
                    M_comp = self.mtrx_transform(coeffs)
                    
                    # Add triangle to history
                    # Triangle is non-zero in [center - rtime, center + rtime]
                    t_min = center_time - self.rtime
                    t_max = center_time + self.rtime
                    
                    # Indices
                    idx_start = int(np.floor(t_min / dt))
                    idx_end = int(np.ceil(t_max / dt)) + 1
                    
                    idx_start = max(0, idx_start)
                    idx_end = min(num_steps, idx_end)
                    
                    if idx_start < idx_end:
                        t_vals = time_points[idx_start:idx_end]
                        # Rate shape (1/s)
                        rate_shape = self.triangle_func(t_vals, center_time, self.rtime)
                        
                        # Add contribution
                        # Shape: (time_slice, 6) += (time_slice, 1) * (6,) * scalar
                        tensor_contribution = M_comp[np.newaxis, :] * m_factor
                        weighted_rate = rate_shape[:, np.newaxis] * tensor_contribution
                        
                        M_rate[idx_start:idx_end, :] += weighted_rate

        # Calculate Scalar Rate using Eigenvalues (Best Double Couple Moment)
        # Based on src/KIKUCHI/sub.focalM_lib.f90 subroutine d_cp.
        # The scalar moment M0 is defined as the average of the major and minor deviatoric eigenvalues:
        # M0 = (lambda_1 - lambda_3) / 2.0  (where lambda_1 >= lambda_2 >= lambda_3)
        
        print("Calculating scalar rate via eigenvalues...")
        for i in range(num_steps):
            # Construct 3x3 symmetric tensor from the 6 components
            # stored as: [M11, M22, M33, M12, M13, M23]
            mt = np.array([
                [M_rate[i, 0], M_rate[i, 3], M_rate[i, 4]],
                [M_rate[i, 3], M_rate[i, 1], M_rate[i, 5]],
                [M_rate[i, 4], M_rate[i, 5], M_rate[i, 2]]
            ])
            
            # Compute eigenvalues (returns in ascending order: lambda_3 <= lambda_2 <= lambda_1)
            eigvals = np.linalg.eigvalsh(mt)
            
            # Apply Best Double Couple formula
            scalar_rate[i] = (eigvals[2] - eigvals[0]) / 2.0
        
        # Scaling to 10^18 and write
        # Output columns: Time, Scalar, M11, M22, M33, M12, M13, M23
        
        print(f"Writing {self.output_path}...")
        with open(self.output_path, 'w') as f:
            for i in range(num_steps):
                t = time_points[i]
                # Scale values by 1e-18 for output
                vals = [scalar_rate[i]] + list(M_rate[i, :])
                vals_scaled = [v * 1e-18 for v in vals]
                
                f.write(f"{t:10.3f}     ")
                f.write(" ".join([f"{v:14.6E}" for v in vals_scaled]))
                f.write("\n")
        
        # Store for plotting
        self.time_points = time_points
        self.scalar_rate = scalar_rate
    
    def plot_mrf(self, output_fig=None):
        """
        Plot the Moment Rate Function (MRF).
        """
        if not hasattr(self, 'time_points') or not hasattr(self, 'scalar_rate'):
            raise ValueError("MRF data not generated. Run generate_mrf() first.")
        
        # Determine scaling exponent 
        max_rate = np.max(self.scalar_rate)
        if max_rate > 0:
            exponent = int(np.floor(np.log10(max_rate)))
        else:
            exponent = 18  # fallback
        
        scale_factor = 10.0 ** (-exponent)
        scalar_rate_scaled = self.scalar_rate * scale_factor
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.fill_between(self.time_points, scalar_rate_scaled, alpha=0.4, color='gray')
        ax.plot(self.time_points, scalar_rate_scaled, color='black', linewidth=1.5)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(rf'Moment Rate ($\times 10^{{{exponent}}}$ N$\cdot$m/s)', fontsize=12)
        ax.set_title('Moment Rate Function', fontsize=14)
        
        ax.set_xlim(0, self.time_points[-1])
        ax.set_ylim(bottom=0)
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', labelsize=10)
        
        # Annotate total moment
        total_moment = np.trapezoid(self.scalar_rate, self.time_points)
        mw_calc = (2.0 / 3.0) * (np.log10(total_moment) - 9.1)
        
        m0_exponent = int(np.floor(np.log10(total_moment)))
        m0_mantissa = total_moment / (10.0 ** m0_exponent)
        
        info_text = f'$M_0$ = {m0_mantissa:.2f} $\\times 10^{{{m0_exponent}}}$ N$\\cdot$m\n$M_w$ = {mw_calc:.2f}'
        ax.text(0.97, 0.95, info_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        plt.tight_layout()
        
        if output_fig:
            plt.savefig(output_fig, dpi=150, bbox_inches='tight')
            print(f"MRF plot saved to: {output_fig}")
        else:
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Moment Rate Function (MRF) from PDTI inversion results."
    )
    parser.add_argument("fort40", help="Path to fort.40 file")
    parser.add_argument("rigid", help="Path to rigid_amp.info file")
    parser.add_argument("--output", "-o", default="mrf.dat", help="Path to output mrf.dat file (default: mrf.dat)")
    parser.add_argument("--plot", "-p", default="mrf.png", help="Path to output MRF figure (default: mrf.png)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.fort40):
        print(f"Error: Input file '{args.fort40}' not found.")
        sys.exit(1)
    if not os.path.exists(args.rigid):
        print(f"Error: Input file '{args.rigid}' not found.")
        sys.exit(1)
    
    gen = PDTI_MRF_Generator(args.fort40, args.rigid, args.output)
    gen.read_fort40()
    gen.read_rigid()
    gen.generate_mrf()
    gen.plot_mrf(output_fig=args.plot)
    
    print("Done.")
