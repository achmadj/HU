"""
Magnetization vs Zeeman Field for Penrose Lattice

This script calculates the magnetization as a function of Zeeman field strength
for a Penrose lattice tight-binding model.

Magnetization M = (N_up - N_down) / N_total for occupied states
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import argparse
import time
import os


def load_penrose_data(filename):
    """Load Penrose lattice data from .npz file"""
    data = np.load(filename)
    vertex_coords = data['vertex_coords']
    edge_list = data['edge_list']
    N = int(data['N'])
    E = int(data['E'])
    return vertex_coords, edge_list, N, E


def build_hamiltonian_with_zeeman(edge_list, N, t=-1.0, zeeman=0.0):
    """
    Build tight-binding Hamiltonian with Zeeman effect
    
    Basis ordering: [up_0, up_1, ..., up_{N-1}, down_0, down_1, ..., down_{N-1}]
    """
    dim = 2 * N
    row_indices = []
    col_indices = []
    data = []
    
    # Hopping terms (spin-conserving)
    for i, j in edge_list:
        # Spin-up sector
        row_indices.extend([i, j])
        col_indices.extend([j, i])
        data.extend([t, t])
        
        # Spin-down sector
        row_indices.extend([i + N, j + N])
        col_indices.extend([j + N, i + N])
        data.extend([t, t])
    
    # Zeeman term: E_up = -zeeman/2, E_down = +zeeman/2
    for i in range(N):
        row_indices.append(i)
        col_indices.append(i)
        data.append(-zeeman / 2.0)
        
        row_indices.append(i + N)
        col_indices.append(i + N)
        data.append(+zeeman / 2.0)
    
    H = csr_matrix((data, (row_indices, col_indices)), shape=(dim, dim))
    return H


def calculate_magnetization(eigenvalues, eigenvectors, N, n_electrons=None, temperature=0.0):
    """
    Calculate magnetization from eigenstates
    
    Parameters:
        eigenvalues: array of eigenvalues (size 2N)
        eigenvectors: (2N, 2N) matrix of eigenvectors (columns)
        N: number of spatial sites
        n_electrons: number of electrons (default: half-filling = N)
        temperature: temperature for Fermi-Dirac distribution (0 = ground state)
    
    Returns:
        M: magnetization (N_up - N_down) / N_total
        n_up: number of spin-up electrons
        n_down: number of spin-down electrons
    """
    dim = 2 * N
    
    if n_electrons is None:
        n_electrons = N  # Half-filling
    
    if temperature == 0.0:
        # Ground state: fill lowest n_electrons states
        sorted_indices = np.argsort(eigenvalues)
        occupied_indices = sorted_indices[:n_electrons]
        
        # Calculate spin-up and spin-down occupation
        n_up = 0.0
        n_down = 0.0
        
        for idx in occupied_indices:
            psi = eigenvectors[:, idx]
            # Spin-up components: 0 to N-1
            # Spin-down components: N to 2N-1
            up_density = np.sum(np.abs(psi[:N])**2)
            down_density = np.sum(np.abs(psi[N:])**2)
            n_up += up_density
            n_down += down_density
    else:
        # Finite temperature: use Fermi-Dirac distribution
        # Find chemical potential for n_electrons
        kT = temperature
        
        def fermi(E, mu):
            x = (E - mu) / kT
            return 1.0 / (1.0 + np.exp(np.clip(x, -500, 500)))
        
        # Binary search for chemical potential
        mu_low, mu_high = eigenvalues.min() - 10*kT, eigenvalues.max() + 10*kT
        for _ in range(100):
            mu = (mu_low + mu_high) / 2
            n_total = np.sum(fermi(eigenvalues, mu))
            if n_total < n_electrons:
                mu_low = mu
            else:
                mu_high = mu
        
        mu = (mu_low + mu_high) / 2
        occupation = fermi(eigenvalues, mu)
        
        n_up = 0.0
        n_down = 0.0
        
        for idx in range(dim):
            psi = eigenvectors[:, idx]
            up_density = np.sum(np.abs(psi[:N])**2)
            down_density = np.sum(np.abs(psi[N:])**2)
            n_up += occupation[idx] * up_density
            n_down += occupation[idx] * down_density
    
    # Magnetization total (not normalized)
    M_tot = n_up - n_down
    # Magnetization normalized (per site)
    M_norm = (n_up - n_down) / N
    
    return M_tot, M_norm, n_up, n_down


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Magnetization vs Zeeman field for Penrose lattice'
    )
    parser.add_argument('--iteration', '-i', type=int, default=4,
                        help='Penrose lattice iteration (default: 4)')
    parser.add_argument('--hopping', '-t', type=float, default=-1.0,
                        help='Hopping parameter (default: -1.0)')
    parser.add_argument('--data-dir', type=str, default='../vertex_model/data',
                        help='Directory containing Penrose lattice data')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for plots (default: results)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for Fermi-Dirac distribution (default: 0.0)')
    
    args = parser.parse_args()
    
    # Zeeman field values: h^z = 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
    zeeman_values = np.arange(0, 22, 2)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    print("=" * 70)
    print("MAGNETIZATION vs ZEEMAN FIELD - PENROSE LATTICE")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Iteration: {args.iteration}")
    print(f"  Hopping parameter: {args.hopping}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Zeeman values: {zeeman_values}")
    print("=" * 70)
    
    # Load data
    print(f"\n[1/3] Loading Penrose lattice data...")
    filename = os.path.join(args.data_dir, f'penrose_lattice_iter{args.iteration}.npz')
    
    if not os.path.exists(filename):
        print(f"\n✗ ERROR: File not found: {filename}")
        print(f"\nPlease generate the data first using:")
        print(f"  cd ../vertex_model")
        print(f"  python penrose_tiling_fast.py --iteration {args.iteration} --save-all")
        return
    
    vertex_coords, edge_list, N, E = load_penrose_data(filename)
    print(f"  ✓ Loaded: {filename}")
    print(f"  ✓ Vertices: {N}")
    print(f"  ✓ Edges: {E}")
    
    # Calculate magnetization for each Zeeman value
    print(f"\n[2/3] Calculating magnetization for each Zeeman field...")
    print(f"  System dimension: {2*N} (spin-up + spin-down)")
    
    magnetization_tot_data = []
    magnetization_norm_data = []
    n_up_data = []
    n_down_data = []
    
    total_time = 0
    for i, Z in enumerate(zeeman_values):
        t0 = time.time()
        
        print(f"\n  [{i+1}/{len(zeeman_values)}] h^z = {Z:.2f}")
        
        # Build Hamiltonian
        H = build_hamiltonian_with_zeeman(edge_list, N, t=args.hopping, zeeman=Z)
        
        # Diagonalize (full spectrum needed for magnetization)
        H_dense = H.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
        
        # Calculate magnetization
        M_tot, M_norm, n_up, n_down = calculate_magnetization(
            eigenvalues, eigenvectors, N, 
            n_electrons=N,  # Half-filling
            temperature=args.temperature
        )
        
        magnetization_tot_data.append(M_tot)
        magnetization_norm_data.append(M_norm)
        n_up_data.append(n_up)
        n_down_data.append(n_down)
        
        t1 = time.time()
        total_time += (t1 - t0)
        
        print(f"      M_tot = {M_tot:.2f}, M_norm = {M_norm:.6f}")
        print(f"      N_up = {n_up:.2f}, N_down = {n_down:.2f}")
        print(f"      Time: {(t1-t0):.2f} s")
    
    magnetization_tot_data = np.array(magnetization_tot_data)
    magnetization_norm_data = np.array(magnetization_norm_data)
    n_up_data = np.array(n_up_data)
    n_down_data = np.array(n_down_data)
    
    print(f"\n  Total computation time: {total_time:.2f} s")
    
    # Plot results
    print(f"\n[3/3] Generating plots...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Total Magnetization vs Zeeman field
    ax1.plot(zeeman_values, magnetization_tot_data, 'o-', 
             linewidth=2.5, markersize=8, color='darkblue', 
             markerfacecolor='darkblue', markeredgecolor='darkblue',
             label=r'$M_{tot}$')
    
    ax1.set_xlabel(r'Zeeman Field $h^z$', fontsize=14)
    ax1.set_ylabel(r'Total Magnetization $M_{tot}$', fontsize=14)
    ax1.set_title(f'Magnetization vs Zeeman Field\n(Iteration {args.iteration})', 
                  fontsize=15, fontweight='bold', style='italic')
    
    ax1.set_xlim(-0.1, zeeman_values.max() + 0.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, loc='upper left')
    
    # Plot 2: Normalized Magnetization vs Zeeman field
    ax2.plot(zeeman_values, magnetization_norm_data, 'o-', 
             linewidth=2.5, markersize=8, color='darkred', 
             markerfacecolor='darkred', markeredgecolor='darkred',
             label=r'$M_{norm}$')
    
    ax2.set_xlabel(r'Zeeman Field $h^z$', fontsize=14)
    ax2.set_ylabel(r'Normalized Magnetization $M_{norm} = (N_\uparrow - N_\downarrow)/N$', fontsize=14)
    ax2.set_title(f'Normalized Magnetization vs Zeeman Field\n(Iteration {args.iteration})', 
                  fontsize=15, fontweight='bold', style='italic')
    
    ax2.set_xlim(-0.1, zeeman_values.max() + 0.1)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12, loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    filename_plot = f'magnetization_vs_zeeman_iter{args.iteration}.png'
    filepath = os.path.join(args.output_dir, filename_plot)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Plot saved: {filepath}")
    
    plt.show()
    
    # Save data to file
    data_filename = f'magnetization_data_iter{args.iteration}.npz'
    data_filepath = os.path.join(args.output_dir, data_filename)
    np.savez(data_filepath,
             zeeman_values=zeeman_values,
             magnetization_tot=magnetization_tot_data,
             magnetization_norm=magnetization_norm_data,
             n_up=n_up_data,
             n_down=n_down_data,
             iteration=args.iteration,
             N=N,
             hopping=args.hopping,
             temperature=args.temperature)
    print(f"  ✓ Data saved: {data_filepath}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'h^z':>6} | {'M_tot':>10} | {'M_norm':>10} | {'N_up':>8} | {'N_down':>8}")
    print("-" * 60)
    for z, mt, mn, nu, nd in zip(zeeman_values, magnetization_tot_data, magnetization_norm_data, n_up_data, n_down_data):
        print(f"{z:>6.2f} | {mt:>10.2f} | {mn:>10.6f} | {nu:>8.1f} | {nd:>8.1f}")
    print("-" * 60)
    
    print("\n" + "=" * 70)
    print("✅ CALCULATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  Plot: {filename_plot}")
    print(f"  Data: {data_filename}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
