"""
Finite Size Effect Analysis untuk Vertex Model Penrose Lattice
Analisis lokalisasi elektron di E=0 untuk berbagai ukuran sistem (N)
Menggunakan Lanczos Shift-Invert (Sparse Matrix)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from typing import Dict, List


class FiniteSizeAnalyzerLanczos:
    """Analisis finite size effect untuk vertex model dengan Lanczos"""
    
    def __init__(self):
        self.results = []  # List of dicts dengan data per iterasi
        self.ldos_data = None  # Data untuk LDOS plotting
    
    def load_and_analyze(self, iteration: int, k_steps: int = 50, sigma: float = 1e-9) -> Dict:
        """
        Load vertex model dan analisis untuk satu iterasi menggunakan Lanczos
        
        Parameters:
            iteration: iteration number
            k_steps: jumlah iterasi Lanczos
            sigma: shift parameter
        
        Returns:
            Dict dengan N, coordination distribution, dan localization
        """
        # Load data
        filename = 'data/penrose_lattice_data.npz'
        if iteration is not None:
            filename = f'data/penrose_lattice_iter{iteration}.npz'
        
        try:
            data = np.load(filename)
        except FileNotFoundError:
            print(f"  ✗ File not found: {filename}")
            return None
        
        edge_list = data['edge_list']
        N = int(data['N'])
        E = int(data['E'])
        
        print(f"\n[Iteration {iteration}] N={N}, E={E}")
        
        # Build Sparse Hamiltonian
        row_indices = []
        col_indices = []
        mat_data = []
        
        for i, j in edge_list:
            row_indices.extend([i, j])
            col_indices.extend([j, i])
            mat_data.extend([-1.0, -1.0])
        
        H = csr_matrix((mat_data, (row_indices, col_indices)), shape=(N, N))
        
        # Lanczos Shift-Invert
        I = eye(N, format='csr')
        M = H - sigma * I
        
        # Inisialisasi vektor awal
        np.random.seed(42)
        v_prev = np.zeros(N)
        v_curr = np.random.rand(N) - 0.5
        v_curr /= np.linalg.norm(v_curr)
        
        beta_prev = 0.0
        alphas = []
        betas = []
        krylov_basis = [v_curr]
        
        # Iterasi Lanczos
        for j in range(k_steps):
            w = spsolve(M, v_curr)
            alpha = np.dot(v_curr, w)
            alphas.append(alpha)
            
            w = w - alpha * v_curr - beta_prev * v_prev
            beta = np.linalg.norm(w)
            
            if beta < 1e-10:
                break
            
            v_next = w / beta
            betas.append(beta)
            krylov_basis.append(v_next)
            
            v_prev = v_curr
            v_curr = v_next
            beta_prev = beta
        
        # Matriks Tridiagonal
        k_actual = len(alphas)
        T_matrix = np.zeros((k_actual, k_actual))
        for i in range(k_actual):
            T_matrix[i, i] = alphas[i]
            if i < k_actual - 1:
                T_matrix[i, i+1] = betas[i]
                T_matrix[i+1, i] = betas[i]
        
        # Diagonalisasi T
        evals_T, evecs_T = np.linalg.eigh(T_matrix)
        
        # Pilih eigenvalue magnitude terbesar (shift-invert)
        idx_max = np.argmax(np.abs(evals_T))
        lambda_best = evals_T[idx_max]
        
        final_energy = sigma + (1.0 / lambda_best)
        
        # Rekonstruksi wavefunction
        y_vec = evecs_T[:, idx_max]
        final_wavefunction = np.zeros(N)
        for i in range(len(y_vec)):
            final_wavefunction += y_vec[i] * krylov_basis[i]
        final_wavefunction /= np.linalg.norm(final_wavefunction)
        
        # Probability density
        prob_density = np.abs(final_wavefunction)**2
        
        # Calculate coordination numbers
        degrees = np.zeros(N, dtype=int)
        for i, j in edge_list:
            degrees[i] += 1
            degrees[j] += 1
        
        # Calculate probability distribution by coordination
        prob_by_coord = {}
        for z in range(1, 10):
            mask = (degrees == z)
            if np.any(mask):
                prob_sum = np.sum(prob_density[mask])
                count = np.sum(mask)
                prob_by_coord[z] = {
                    'probability': prob_sum,
                    'count': count,
                    'percentage': (count / N) * 100
                }
        
        return {
            'iteration': iteration,
            'N': N,
            'E': E,
            'energy': final_energy,
            'prob_by_coord': prob_by_coord,
            'degrees': degrees
        }
    
    def analyze_multiple_iterations(self, iterations: List[int], k_steps: int = 50) -> None:
        """
        Analisis untuk multiple iterations
        
        Parameters:
            iterations: List of iteration numbers to analyze
            k_steps: Lanczos iteration steps
        """
        print("="*80)
        print("FINITE SIZE EFFECT ANALYSIS - VERTEX MODEL (LANCZOS)")
        print("Electron Localization at E = 0.0")
        print("="*80)
        
        for iteration in iterations:
            result = self.load_and_analyze(iteration, k_steps=k_steps)
            if result is not None:
                self.results.append(result)
    
    def print_comparison_table(self) -> None:
        """Print tabel perbandingan finite size effect"""
        if not self.results:
            print("\nNo results to display!")
            return
        
        print("\n")
        print("="*80)
        print("FINITE SIZE EFFECT: Localization by Coordination Number")
        print("="*80)
        
        # Collect all coordination numbers that appear
        all_coords = set()
        for result in self.results:
            all_coords.update(result['prob_by_coord'].keys())
        all_coords = sorted(all_coords)
        
        # Get N values
        N_values = [r['N'] for r in self.results]
        
        # Print header
        header = f"{'z':<3}"
        for N in N_values:
            header += f" | {N:>6}"
        print(header)
        print("-" * len(header))
        
        # Print data for each coordination number
        for z in all_coords:
            row = f"{z:<3}"
            for result in self.results:
                if z in result['prob_by_coord']:
                    prob = result['prob_by_coord'][z]['probability']
                    row += f" | {prob*100:>5.2f}%"
                else:
                    row += " |     - "
            print(row)
        
        print("="*80)
    
    def calculate_ldos_full(self, iteration: int, bins: int = 200) -> Dict:
        """
        Calculate full LDOS dengan diagonalisasi penuh
        Diperlukan untuk mendapatkan semua eigenvalues dan eigenvectors
        
        LDOS Formula:
        ρ_z(ω) = Σ_n Σ_{i∈z} |ψ_n(i)|² δ(ω - E_n)
        
        Parameters:
            iteration: iteration number
            bins: number of energy bins
            
        Returns:
            Dict dengan LDOS per coordination number
        """
        # Load data
        filename = 'data/penrose_lattice_data.npz'
        if iteration is not None:
            filename = f'data/penrose_lattice_iter{iteration}.npz'
        
        try:
            data = np.load(filename)
        except FileNotFoundError:
            print(f"  ✗ File not found: {filename}")
            return None
        
        edge_list = data['edge_list']
        N = int(data['N'])
        
        print(f"\n[LDOS Calculation] Iteration {iteration}, N={N}")
        print("  Building Hamiltonian...")
        
        # Build Hamiltonian (dense untuk full diagonalization)
        H = np.zeros((N, N), dtype=np.float64)
        for i, j in edge_list:
            H[i, j] = -1.0
            H[j, i] = -1.0
        
        print("  Diagonalizing (this may take a while)...")
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        # Calculate coordination numbers
        degrees = np.zeros(N, dtype=int)
        for i, j in edge_list:
            degrees[i] += 1
            degrees[j] += 1
        
        # Energy range untuk histogram
        e_min, e_max = eigenvalues.min(), eigenvalues.max()
        energy_bins = np.linspace(e_min, e_max, bins)
        energy_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
        
        # Calculate LDOS for each coordination number
        ldos_by_z = {}
        
        # Get unique coordination numbers
        unique_z = sorted(set(degrees))
        
        print(f"  Calculating LDOS for z = {unique_z}...")
        
        for z in unique_z:
            # Sites dengan coordination z
            sites_z = np.where(degrees == z)[0]
            
            if len(sites_z) == 0:
                continue
            
            # Initialize LDOS
            ldos = np.zeros(bins - 1)
            
            # Sum over all states
            for n in range(N):
                # Probability density di sites dengan coordination z
                prob_z = np.sum(np.abs(eigenvectors[sites_z, n])**2)
                
                # Find energy bin
                bin_idx = np.searchsorted(energy_bins, eigenvalues[n]) - 1
                if 0 <= bin_idx < len(ldos):
                    ldos[bin_idx] += prob_z
            
            # Normalize by bin width
            bin_width = (e_max - e_min) / (bins - 1)
            ldos /= bin_width
            
            # Normalize by N (per-site normalization, sesuai paper)
            ldos /= N
            
            ldos_by_z[z] = ldos
        
        return {
            'energy_centers': energy_centers,
            'ldos_by_z': ldos_by_z,
            'N': N,
            'iteration': iteration
        }
    
    def plot_ldos(self, iteration: int, bins: int = 200, filename: str = 'vertex_model/imgs/ldos_by_coordination.png'):
        """
        Plot LDOS by coordination number
        
        Parameters:
            iteration: iteration number to analyze
            bins: number of energy bins
            filename: output filename
        """
        ldos_data = self.calculate_ldos_full(iteration, bins)
        
        if ldos_data is None:
            return
        
        energy_centers = ldos_data['energy_centers']
        ldos_by_z = ldos_data['ldos_by_z']
        N = ldos_data['N']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color map untuk different z
        colors = {
            2: 'gray',
            3: 'black',
            4: 'green',
            5: 'red',
            6: 'blue',
            7: 'yellow',
            8: 'purple'
        }
        
        # Plot LDOS for each z (stacked bar)
        bottom = np.zeros(len(energy_centers))
        
        for z in sorted(ldos_by_z.keys()):
            ldos = ldos_by_z[z]
            color = colors.get(z, 'gray')
            ax.bar(energy_centers, ldos, width=(energy_centers[1]-energy_centers[0]),
                   bottom=bottom, label=f'z={z}', color=color, alpha=0.8, edgecolor='none')
            bottom += ldos
        
        ax.set_xlabel(r'$\hbar\omega/t$', fontsize=12)
        ax.set_ylabel(r'$\rho_z(\omega)$', fontsize=12)
        ax.set_title(f'Local Density of States (LDOS) by Coordination Number\nN={N}', 
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set y-limit seperti di paper (biar detail z=4,5,6 keliatan)
        # Puncak E=0 akan terpotong (seperti di Figure 12 paper)
        ax.set_ylim(0, 0.1)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n  ✓ LDOS plot saved: {filename}")
        plt.close()


def main():
    """Main program"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Finite Size Analysis for Vertex Model (E=0) with Lanczos'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        nargs='+',
        default=[2, 3, 4],
        help='List of iteration numbers to analyze (default: 2 3 4)'
    )
    parser.add_argument(
        '--k-steps',
        type=int,
        default=50,
        help='Number of Lanczos iterations (default: 50)'
    )
    parser.add_argument(
        '--plot-ldos',
        action='store_true',
        help='Generate LDOS plot (requires full diagonalization)'
    )
    parser.add_argument(
        '--ldos-iteration',
        type=int,
        default=4,
        help='Iteration to use for LDOS plot (default: 4)'
    )
    parser.add_argument(
        '--ldos-bins',
        type=int,
        default=200,
        help='Number of energy bins for LDOS (default: 200)'
    )
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FiniteSizeAnalyzerLanczos()
    
    print("ℹ️  Using Sparse Matrix + Lanczos Shift-Invert")
    
    # Analyze
    analyzer.analyze_multiple_iterations(args.iterations, k_steps=args.k_steps)
    
    # Print results
    analyzer.print_comparison_table()
    
    # Plot LDOS if requested
    if args.plot_ldos:
        print("\n" + "="*80)
        print("GENERATING LDOS PLOT")
        print("="*80)
        analyzer.plot_ldos(args.ldos_iteration, bins=args.ldos_bins)
    
    print("\n✅ Finite size analysis completed!")


if __name__ == "__main__":
    main()
