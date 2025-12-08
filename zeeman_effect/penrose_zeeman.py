"""
Penrose Lattice Tight-Binding with Zeeman Effect

This script calculates the electronic properties of a Penrose lattice
with an applied magnetic field (Zeeman effect).

The Hamiltonian includes:
- Tight-binding hopping: H_hop = -t * sum_{<i,j>} (c_i^dagger c_j + h.c.)
- Zeeman splitting: H_Z = -g * mu_B * B * sum_i s_z * c_i^dagger c_i

where s_z = +1/2 for spin-up, -1/2 for spin-down
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import argparse
import time
import os


def load_penrose_data(filename):
    """
    Load Penrose lattice data from .npz file
    
    Returns:
        vertex_coords: (N, 2) array of vertex positions
        edge_list: (E, 2) array of edges
        N: number of vertices
        E: number of edges
    """
    data = np.load(filename)
    vertex_coords = data['vertex_coords']
    edge_list = data['edge_list']
    N = int(data['N'])
    E = int(data['E'])
    
    return vertex_coords, edge_list, N, E


def build_hamiltonian_with_zeeman(edge_list, N, t=-1.0, zeeman=0.0):
    """
    Build tight-binding Hamiltonian with Zeeman effect
    
    Parameters:
        edge_list: (E, 2) array of edges
        N: number of vertices
        t: hopping parameter (default: -1.0)
        zeeman: Zeeman energy splitting (default: 0.0)
    
    Returns:
        H: (2N, 2N) Hamiltonian matrix in spin space
           Basis ordering: [up_0, up_1, ..., up_{N-1}, down_0, down_1, ..., down_{N-1}]
    """
    # Total dimension is 2N (spin-up and spin-down)
    dim = 2 * N
    
    # Build sparse Hamiltonian
    row_indices = []
    col_indices = []
    data = []
    
    # Hopping terms (spin-conserving)
    for i, j in edge_list:
        # Spin-up sector: (i, j) -> (i, j)
        row_indices.extend([i, j])
        col_indices.extend([j, i])
        data.extend([t, t])
        
        # Spin-down sector: (i+N, j+N) -> (i+N, j+N)
        row_indices.extend([i + N, j + N])
        col_indices.extend([j + N, i + N])
        data.extend([t, t])
    
    # Zeeman term (on-site energy difference between spin-up and spin-down)
    # H_Z = -zeeman * s_z, where s_z = +1/2 for up, -1/2 for down
    # This gives: E_up = -zeeman/2, E_down = +zeeman/2
    for i in range(N):
        # Spin-up: -zeeman/2
        row_indices.append(i)
        col_indices.append(i)
        data.append(-zeeman / 2.0)
        
        # Spin-down: +zeeman/2
        row_indices.append(i + N)
        col_indices.append(i + N)
        data.append(+zeeman / 2.0)
    
    H = csr_matrix((data, (row_indices, col_indices)), shape=(dim, dim))
    
    return H


def calculate_dos(eigenvalues, energy_range=None, num_bins=500):
    """
    Calculate Density of States (DOS) using histogram
    
    Parameters:
        eigenvalues: array of eigenvalues
        energy_range: (E_min, E_max) or None for automatic
        num_bins: number of bins for histogram
    
    Returns:
        energies: energy values
        dos: density of states
    """
    if energy_range is None:
        E_min = eigenvalues.min() - 0.5
        E_max = eigenvalues.max() + 0.5
    else:
        E_min, E_max = energy_range
    
    dos, bin_edges = np.histogram(eigenvalues, bins=num_bins, 
                                   range=(E_min, E_max), density=True)
    energies = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return energies, dos


def calculate_idos(energies, dos):
    """
    Calculate Integrated Density of States (IDOS)
    
    Parameters:
        energies: energy values
        dos: density of states
    
    Returns:
        idos: integrated density of states
    """
    dE = energies[1] - energies[0]
    idos = np.cumsum(dos) * dE
    
    return idos


def plot_results(eigenvalues, zeeman, iteration, save_dir='results'):
    """
    Plot DOS, IDOS, and Energy Spectrum
    
    Parameters:
        eigenvalues: array of eigenvalues
        zeeman: Zeeman splitting parameter
        iteration: iteration number
        save_dir: directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate DOS and IDOS
    energies, dos = calculate_dos(eigenvalues)
    idos = calculate_idos(energies, dos)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Energy Spectrum
    ax1 = axes[0]
    n_states = len(eigenvalues)
    state_indices = np.arange(n_states)
    
    ax1.scatter(state_indices, eigenvalues, s=1, alpha=0.5, c='blue')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='E=0')
    ax1.set_xlabel('State Index', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title(f'Energy Spectrum\n(Iteration {iteration}, Z={zeeman:.3f})', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Density of States (DOS)
    ax2 = axes[1]
    ax2.plot(energies, dos, linewidth=2, color='darkblue')
    ax2.fill_between(energies, dos, alpha=0.3, color='blue')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='E=0')
    ax2.set_xlabel('Energy', fontsize=12)
    ax2.set_ylabel('DOS (states/energy)', fontsize=12)
    ax2.set_title(f'Density of States\n(Iteration {iteration}, Z={zeeman:.3f})', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Integrated Density of States (IDOS)
    ax3 = axes[2]
    ax3.plot(energies, idos, linewidth=2, color='darkgreen')
    ax3.fill_between(energies, idos, alpha=0.3, color='green')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='E=0')
    ax3.set_xlabel('Energy', fontsize=12)
    ax3.set_ylabel('IDOS (cumulative states)', fontsize=12)
    ax3.set_title(f'Integrated Density of States\n(Iteration {iteration}, Z={zeeman:.3f})', 
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    # Save figure
    filename = f'zeeman_Z{zeeman:.3f}_iter{iteration}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {filepath}")
    
    plt.show()


def analyze_spin_splitting(eigenvalues, zeeman, N):
    """
    Analyze spin-up and spin-down state splitting
    
    Parameters:
        eigenvalues: array of eigenvalues (size 2N)
        zeeman: Zeeman splitting parameter
        N: number of spatial sites
    """
    print("\n" + "=" * 70)
    print("SPIN SPLITTING ANALYSIS")
    print("=" * 70)
    
    n_states = len(eigenvalues)
    
    # Find states near E=0
    zero_tolerance = 1e-10
    zero_mask = np.abs(eigenvalues) < zero_tolerance
    n_zero = np.sum(zero_mask)
    
    print(f"\nTotal states: {n_states}")
    print(f"  Spin-up sector: {N}")
    print(f"  Spin-down sector: {N}")
    print(f"\nZeeman parameter: {zeeman:.6f}")
    print(f"Expected splitting: ±{zeeman/2:.6f}")
    
    print(f"\nStates near E=0 (|E| < {zero_tolerance}):")
    print(f"  Count: {n_zero}")
    print(f"  Percentage: {100*n_zero/n_states:.2f}%")
    
    # Energy range
    print(f"\nEnergy spectrum:")
    print(f"  Min energy: {eigenvalues.min():.6f}")
    print(f"  Max energy: {eigenvalues.max():.6f}")
    print(f"  Energy range: {eigenvalues.max() - eigenvalues.min():.6f}")
    
    # States around zero energy
    if n_zero > 0:
        zero_energies = eigenvalues[zero_mask]
        print(f"\nZero energy states (first 10):")
        for i, E in enumerate(zero_energies[:10]):
            print(f"    E_{i} = {E:.10f}")
        if n_zero > 10:
            print(f"    ... and {n_zero - 10} more")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Penrose lattice tight-binding with Zeeman effect'
    )
    parser.add_argument('--iteration', '-i', type=int, default=4,
                        help='Penrose lattice iteration (default: 4)')
    parser.add_argument('--zeeman', '-z', type=float, default=0.0,
                        help='Zeeman splitting parameter (default: 0.0)')
    parser.add_argument('--hopping', '-t', type=float, default=-1.0,
                        help='Hopping parameter (default: -1.0)')
    parser.add_argument('--data-dir', type=str, default='../vertex_model/data',
                        help='Directory containing Penrose lattice data')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for plots (default: results)')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print("PENROSE LATTICE TIGHT-BINDING WITH ZEEMAN EFFECT")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Iteration: {args.iteration}")
    print(f"  Zeeman splitting: {args.zeeman}")
    print(f"  Hopping parameter: {args.hopping}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 70)
    
    # Load data
    print(f"\n[1/4] Loading Penrose lattice data...")
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
    
    # Build Hamiltonian
    print(f"\n[2/4] Building Hamiltonian with Zeeman effect...")
    t0 = time.time()
    H = build_hamiltonian_with_zeeman(edge_list, N, t=args.hopping, zeeman=args.zeeman)
    t1 = time.time()
    
    dim = 2 * N
    print(f"  ✓ Hamiltonian size: {dim} × {dim} (spin-up + spin-down)")
    print(f"  ✓ Non-zero elements: {H.nnz:,}")
    print(f"  ✓ Sparsity: {100 * (1 - H.nnz / (dim * dim)):.2f}%")
    print(f"  ✓ Build time: {(t1-t0)*1000:.2f} ms")
    
    # Diagonalize
    print(f"\n[3/4] Diagonalizing Hamiltonian...")
    print(f"  (This may take a while for large systems...)")
    t0 = time.time()
    
    # For dense diagonalization (smaller systems)
    if dim < 5000:
        print(f"  Using dense diagonalization...")
        H_dense = H.toarray()
        eigenvalues = np.linalg.eigvalsh(H_dense)
    else:
        # For larger systems, use sparse (but get all eigenvalues is expensive)
        print(f"  Warning: System is large (dim={dim}). Using dense method may be slow.")
        print(f"  Consider using sparse methods for production runs.")
        H_dense = H.toarray()
        eigenvalues = np.linalg.eigvalsh(H_dense)
    
    t1 = time.time()
    
    print(f"  ✓ Diagonalization completed")
    print(f"  ✓ Number of eigenvalues: {len(eigenvalues)}")
    print(f"  ✓ Time: {(t1-t0):.3f} seconds")
    
    # Analyze spin splitting
    analyze_spin_splitting(eigenvalues, args.zeeman, N)
    
    # Plot results
    print(f"\n[4/4] Generating plots...")
    plot_results(eigenvalues, args.zeeman, args.iteration, save_dir=args.output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ CALCULATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  Plot: zeeman_Z{args.zeeman:.3f}_iter{args.iteration}.png")
    print("\nTo run with different Zeeman field:")
    print(f"  python penrose_zeeman.py --iteration {args.iteration} --zeeman 0.5")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
