"""
Local Spin Density Plot for Penrose Lattice with Zeeman Effect

This script calculates and plots the local spin density on each site
of the Penrose lattice under a given Zeeman field.

Local spin density: S_z(i) = (n_up(i) - n_down(i)) / 2
where n_up(i) and n_down(i) are the local occupation densities.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import argparse
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


def calculate_local_spin_density(eigenvalues, eigenvectors, N, n_electrons=None):
    """
    Calculate local spin density at each site
    
    Parameters:
        eigenvalues: array of eigenvalues (size 2N)
        eigenvectors: (2N, 2N) matrix of eigenvectors (columns)
        N: number of spatial sites
        n_electrons: number of electrons (default: half-filling = N)
    
    Returns:
        local_spin: local spin density S_z(i) at each site (size N)
        local_n_up: local spin-up occupation (size N)
        local_n_down: local spin-down occupation (size N)
    """
    if n_electrons is None:
        n_electrons = N  # Half-filling
    
    # Ground state: fill lowest n_electrons states
    sorted_indices = np.argsort(eigenvalues)
    occupied_indices = sorted_indices[:n_electrons]
    
    # Calculate local spin-up and spin-down occupation
    local_n_up = np.zeros(N)
    local_n_down = np.zeros(N)
    
    for idx in occupied_indices:
        psi = eigenvectors[:, idx]
        # Spin-up components: 0 to N-1
        # Spin-down components: N to 2N-1
        local_n_up += np.abs(psi[:N])**2
        local_n_down += np.abs(psi[N:])**2
    
    # Local spin density: S_z(i) = (n_up(i) - n_down(i)) / 2
    local_spin = (local_n_up - local_n_down) / 2.0
    
    return local_spin, local_n_up, local_n_down


def plot_local_spin_density(vertex_coords, edge_list, local_spin, zeeman, iteration, save_dir='results'):
    """
    Plot local spin density on the Penrose lattice
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw edges
    for i, j in edge_list:
        x = [vertex_coords[i, 0], vertex_coords[j, 0]]
        y = [vertex_coords[i, 1], vertex_coords[j, 1]]
        ax.plot(x, y, 'k-', linewidth=0.3, alpha=0.3)
    
    # Plot local spin density as colored scatter plot
    scatter = ax.scatter(vertex_coords[:, 0], vertex_coords[:, 1], 
                         c=local_spin, cmap='coolwarm', 
                         s=50, edgecolors='black', linewidths=0.5,
                         vmin=-0.5, vmax=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(r'Local Spin Density $S_z(i)$', fontsize=12)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Local Spin Density on Penrose Lattice\n' + 
                 r'$h^z = $' + f'{zeeman:.1f} (Iteration {iteration})', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'local_spin_density_hz{zeeman:.1f}_iter{iteration}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Plot saved: {filepath}")
    
    plt.show()
    
    return filename


def main():
    parser = argparse.ArgumentParser(
        description='Plot Local Spin Density for Penrose lattice with Zeeman effect'
    )
    parser.add_argument('--iteration', '-i', type=int, default=4,
                        help='Penrose lattice iteration (default: 4)')
    parser.add_argument('--zeeman', '-z', type=float, default=20.0,
                        help='Zeeman field strength (default: 20.0)')
    parser.add_argument('--hopping', '-t', type=float, default=-1.0,
                        help='Hopping parameter (default: -1.0)')
    parser.add_argument('--data-dir', type=str, default='../vertex_model/data',
                        help='Directory containing Penrose lattice data')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for plots (default: results)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LOCAL SPIN DENSITY - PENROSE LATTICE WITH ZEEMAN EFFECT")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Iteration: {args.iteration}")
    print(f"  Zeeman field h^z: {args.zeeman}")
    print(f"  Hopping parameter: {args.hopping}")
    print("=" * 70)
    
    # Load data
    print(f"\n[1/4] Loading Penrose lattice data...")
    filename = os.path.join(args.data_dir, f'penrose_lattice_iter{args.iteration}.npz')
    
    if not os.path.exists(filename):
        print(f"\n✗ ERROR: File not found: {filename}")
        return
    
    vertex_coords, edge_list, N, E = load_penrose_data(filename)
    print(f"  ✓ Loaded: {filename}")
    print(f"  ✓ Vertices: {N}")
    print(f"  ✓ Edges: {E}")
    
    # Build Hamiltonian
    print(f"\n[2/4] Building Hamiltonian with Zeeman effect...")
    H = build_hamiltonian_with_zeeman(edge_list, N, t=args.hopping, zeeman=args.zeeman)
    print(f"  ✓ Hamiltonian size: {2*N} × {2*N}")
    
    # Diagonalize
    print(f"\n[3/4] Diagonalizing Hamiltonian...")
    H_dense = H.toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
    print(f"  ✓ Diagonalization completed")
    
    # Calculate local spin density
    print(f"\n[4/4] Calculating local spin density...")
    local_spin, local_n_up, local_n_down = calculate_local_spin_density(
        eigenvalues, eigenvectors, N, n_electrons=N
    )
    
    print(f"  ✓ Total spin-up: {np.sum(local_n_up):.2f}")
    print(f"  ✓ Total spin-down: {np.sum(local_n_down):.2f}")
    print(f"  ✓ Total magnetization: {np.sum(local_spin)*2:.2f}")
    print(f"  ✓ Mean local spin: {np.mean(local_spin):.6f}")
    print(f"  ✓ Min local spin: {np.min(local_spin):.6f}")
    print(f"  ✓ Max local spin: {np.max(local_spin):.6f}")
    
    # Plot
    print(f"\n[5/4] Generating plot...")
    plot_local_spin_density(vertex_coords, edge_list, local_spin, 
                            args.zeeman, args.iteration, save_dir=args.output_dir)
    
    print("\n" + "=" * 70)
    print("✅ CALCULATION COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
