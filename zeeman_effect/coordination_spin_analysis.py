"""
Coordination Number Analysis for Zeeman Effect on Penrose Lattice

This script analyzes the relationship between coordination number (z=2,3,4,5,6,7)
and local spin density under different Zeeman field strengths.

Sites are classified into three categories based on spin density:
- Weak: Low spin polarization
- Intermediate: Medium spin polarization  
- Strong: High spin polarization
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
    """Build tight-binding Hamiltonian with Zeeman effect"""
    dim = 2 * N
    row_indices = []
    col_indices = []
    data = []
    
    for i, j in edge_list:
        row_indices.extend([i, j])
        col_indices.extend([j, i])
        data.extend([t, t])
        
        row_indices.extend([i + N, j + N])
        col_indices.extend([j + N, i + N])
        data.extend([t, t])
    
    for i in range(N):
        row_indices.append(i)
        col_indices.append(i)
        data.append(-zeeman / 2.0)
        
        row_indices.append(i + N)
        col_indices.append(i + N)
        data.append(+zeeman / 2.0)
    
    H = csr_matrix((data, (row_indices, col_indices)), shape=(dim, dim))
    return H


def get_coordination_numbers(edge_list, N):
    """
    Calculate coordination number (number of neighbors) for each site
    
    Returns:
        coord_numbers: Array of coordination numbers for each site
    """
    coord_numbers = np.zeros(N, dtype=int)
    
    for i, j in edge_list:
        coord_numbers[i] += 1
        coord_numbers[j] += 1
    
    return coord_numbers


def calculate_local_spin_density(eigenvectors, N, num_filled_states=None):
    """Calculate local spin density <S_z_i> at each site i"""
    if num_filled_states is None:
        num_filled_states = N
    
    local_Sz = np.zeros(N)
    
    for state_idx in range(num_filled_states):
        psi = eigenvectors[:, state_idx]
        
        for i in range(N):
            spin_up_prob = np.abs(psi[i])**2
            spin_down_prob = np.abs(psi[i + N])**2
            local_Sz[i] += 0.5 * (spin_up_prob - spin_down_prob)
    
    return local_Sz


def classify_spin_strength(spin_density, max_spin=0.5):
    """
    Classify spin density into weak, intermediate, strong categories
    
    Classification based on fraction of maximum possible spin (0.5):
    - Weak: |S_z| < 0.1 * max_spin (< 10% polarized)
    - Intermediate: 0.1 <= |S_z| < 0.4 * max_spin (10-40% polarized)
    - Strong: |S_z| >= 0.4 * max_spin (>= 40% polarized)
    
    Returns:
        'weak', 'intermediate', or 'strong'
    """
    abs_spin = np.abs(spin_density)
    
    if abs_spin < 0.1 * max_spin:
        return 'weak'
    elif abs_spin < 0.4 * max_spin:
        return 'intermediate'
    else:
        return 'strong'


def analyze_coordination_vs_spin(coord_numbers, local_Sz, zeeman):
    """
    Analyze relationship between coordination number and spin density
    
    Returns:
        analysis_dict: Dictionary with statistics for each coordination number
    """
    analysis = {}
    
    for z in range(2, 8):  # z = 2, 3, 4, 5, 6, 7
        mask = coord_numbers == z
        n_sites = np.sum(mask)
        
        if n_sites == 0:
            continue
        
        spins_z = local_Sz[mask]
        
        # Statistics
        mean_spin = np.mean(spins_z)
        std_spin = np.std(spins_z)
        min_spin = np.min(spins_z)
        max_spin = np.max(spins_z)
        
        # Classification counts
        classifications = [classify_spin_strength(s) for s in spins_z]
        n_weak = classifications.count('weak')
        n_intermediate = classifications.count('intermediate')
        n_strong = classifications.count('strong')
        
        analysis[z] = {
            'n_sites': n_sites,
            'mean_spin': mean_spin,
            'std_spin': std_spin,
            'min_spin': min_spin,
            'max_spin': max_spin,
            'n_weak': n_weak,
            'n_intermediate': n_intermediate,
            'n_strong': n_strong,
            'pct_weak': 100 * n_weak / n_sites,
            'pct_intermediate': 100 * n_intermediate / n_sites,
            'pct_strong': 100 * n_strong / n_sites
        }
    
    return analysis


def print_analysis_table(analysis, zeeman, N):
    """Print analysis results as formatted table"""
    print(f"\n{'='*90}")
    print(f"COORDINATION NUMBER vs SPIN DENSITY ANALYSIS (h^z = {zeeman})")
    print(f"{'='*90}")
    
    # Header
    print(f"\n{'z':>4} | {'Sites':>6} | {'<S_z>':>10} | {'σ(S_z)':>8} | "
          f"{'Weak':>8} | {'Interm':>8} | {'Strong':>8}")
    print(f"{'-'*4}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    
    total_weak = 0
    total_interm = 0
    total_strong = 0
    total_sites = 0
    
    for z in sorted(analysis.keys()):
        data = analysis[z]
        print(f"{z:>4} | {data['n_sites']:>6} | {data['mean_spin']:>10.6f} | "
              f"{data['std_spin']:>8.6f} | "
              f"{data['pct_weak']:>7.1f}% | {data['pct_intermediate']:>7.1f}% | "
              f"{data['pct_strong']:>7.1f}%")
        
        total_weak += data['n_weak']
        total_interm += data['n_intermediate']
        total_strong += data['n_strong']
        total_sites += data['n_sites']
    
    print(f"{'-'*4}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    print(f"{'Tot':>4} | {total_sites:>6} | {'-':>10} | {'-':>8} | "
          f"{100*total_weak/total_sites:>7.1f}% | {100*total_interm/total_sites:>7.1f}% | "
          f"{100*total_strong/total_sites:>7.1f}%")
    
    print(f"\nClassification thresholds:")
    print(f"  Weak:         |S_z| < 0.05 (< 10% of max)")
    print(f"  Intermediate: 0.05 <= |S_z| < 0.20 (10-40% of max)")
    print(f"  Strong:       |S_z| >= 0.20 (>= 40% of max)")


def plot_coordination_analysis(all_analyses, zeeman_values, iteration, save_dir):
    """Plot coordination number vs spin density analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    coord_numbers = sorted(all_analyses[zeeman_values[0]].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(coord_numbers)))
    
    # Plot 1: Mean spin density vs Zeeman field for each z
    ax1 = axes[0, 0]
    for i, z in enumerate(coord_numbers):
        mean_spins = [all_analyses[hz][z]['mean_spin'] if z in all_analyses[hz] else np.nan 
                      for hz in zeeman_values]
        ax1.plot(zeeman_values, mean_spins, 'o-', color=colors[i], 
                 label=f'z={z}', linewidth=2, markersize=6)
    
    ax1.set_xlabel(r'Zeeman Field $h^z$', fontsize=12)
    ax1.set_ylabel(r'Mean Local Spin Density $\langle S_z \rangle$', fontsize=12)
    ax1.set_title('Mean Spin Density vs Zeeman Field\nby Coordination Number', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: Percentage of strong spin sites vs Zeeman field for each z
    ax2 = axes[0, 1]
    for i, z in enumerate(coord_numbers):
        pct_strong = [all_analyses[hz][z]['pct_strong'] if z in all_analyses[hz] else np.nan 
                      for hz in zeeman_values]
        ax2.plot(zeeman_values, pct_strong, 'o-', color=colors[i], 
                 label=f'z={z}', linewidth=2, markersize=6)
    
    ax2.set_xlabel(r'Zeeman Field $h^z$', fontsize=12)
    ax2.set_ylabel('Strong Spin Sites (%)', fontsize=12)
    ax2.set_title('Percentage of Strong Spin Sites\nby Coordination Number', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Stacked bar chart for classification at specific Zeeman values
    ax3 = axes[1, 0]
    selected_hz = [zeeman_values[0], zeeman_values[len(zeeman_values)//2], zeeman_values[-1]]
    x = np.arange(len(coord_numbers))
    width = 0.25
    
    for idx, hz in enumerate(selected_hz):
        weak_pct = [all_analyses[hz][z]['pct_weak'] if z in all_analyses[hz] else 0 
                    for z in coord_numbers]
        ax3.bar(x + idx*width, weak_pct, width, label=f'h^z={hz} (Weak)', 
                color=plt.cm.Blues(0.3 + idx*0.2), edgecolor='black', linewidth=0.5)
    
    ax3.set_xlabel('Coordination Number z', fontsize=12)
    ax3.set_ylabel('Weak Spin Sites (%)', fontsize=12)
    ax3.set_title('Weak Spin Sites by Coordination Number', fontsize=13, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(coord_numbers)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Box plot of spin density distribution by coordination number (last Zeeman value)
    ax4 = axes[1, 1]
    last_hz = zeeman_values[-1]
    
    # We need the raw data, not just statistics - regenerate for this plot
    ax4.text(0.5, 0.5, f'Spin Density Distribution\nat h^z = {last_hz}\n(See table for details)', 
             transform=ax4.transAxes, ha='center', va='center', fontsize=14)
    ax4.set_xlabel('Coordination Number z', fontsize=12)
    ax4.set_ylabel(r'Local Spin Density $S_z$', fontsize=12)
    ax4.set_title(f'Spin Density Distribution (h^z = {last_hz})', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    filename = f'coordination_spin_analysis_iter{iteration}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze coordination number vs spin density for Penrose lattice with Zeeman effect'
    )
    parser.add_argument('--iteration', '-i', type=int, default=4,
                        help='Penrose lattice iteration (default: 4)')
    parser.add_argument('--zeeman-values', '-z', type=float, nargs='+',
                        default=[0.0, 0.1, 0.5, 1.0, 2.0, 3.0],
                        help='List of Zeeman values to analyze')
    parser.add_argument('--data-dir', type=str, default='../vertex_model/data',
                        help='Directory containing Penrose lattice data')
    parser.add_argument('--output-dir', type=str, default='results_coordination',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 90)
    print("COORDINATION NUMBER vs SPIN DENSITY ANALYSIS")
    print("=" * 90)
    print(f"\nParameters:")
    print(f"  Iteration: {args.iteration}")
    print(f"  Zeeman values: {args.zeeman_values}")
    print("=" * 90)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"\n[1/4] Loading Penrose lattice data...")
    filename = os.path.join(args.data_dir, f'penrose_lattice_iter{args.iteration}.npz')
    
    if not os.path.exists(filename):
        print(f"\n✗ ERROR: File not found: {filename}")
        return
    
    vertex_coords, edge_list, N, E = load_penrose_data(filename)
    print(f"  ✓ Vertices: {N}")
    print(f"  ✓ Edges: {E}")
    
    # Get coordination numbers
    print(f"\n[2/4] Calculating coordination numbers...")
    coord_numbers = get_coordination_numbers(edge_list, N)
    
    # Print coordination number distribution
    print(f"\n  Coordination Number Distribution:")
    for z in range(2, 8):
        n_z = np.sum(coord_numbers == z)
        if n_z > 0:
            print(f"    z={z}: {n_z:5d} sites ({100*n_z/N:5.2f}%)")
    
    # Analyze for each Zeeman value
    print(f"\n[3/4] Analyzing spin density for each Zeeman field...")
    
    all_analyses = {}
    
    for hz in args.zeeman_values:
        print(f"\n  Processing h^z = {hz}...")
        
        # Build Hamiltonian
        H = build_hamiltonian_with_zeeman(edge_list, N, t=-1.0, zeeman=hz)
        
        # Diagonalize
        t0 = time.time()
        H_dense = H.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
        t1 = time.time()
        
        # Calculate local spin density
        local_Sz = calculate_local_spin_density(eigenvectors, N, num_filled_states=N)
        
        # Analyze coordination vs spin
        analysis = analyze_coordination_vs_spin(coord_numbers, local_Sz, hz)
        all_analyses[hz] = analysis
        
        # Print table
        print_analysis_table(analysis, hz, N)
        
        print(f"  ✓ Done (diagonalization: {(t1-t0):.2f}s)")
    
    # Generate plots
    print(f"\n[4/4] Generating plots...")
    plot_coordination_analysis(all_analyses, args.zeeman_values, args.iteration, args.output_dir)
    
    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    print("\nKey observations:")
    
    # Compare z=3 vs z=7 at different Zeeman fields
    if 3 in all_analyses[args.zeeman_values[-1]] and 7 in all_analyses[args.zeeman_values[-1]]:
        last_hz = args.zeeman_values[-1]
        z3_strong = all_analyses[last_hz][3]['pct_strong']
        z7_strong = all_analyses[last_hz][7]['pct_strong']
        z3_mean = all_analyses[last_hz][3]['mean_spin']
        z7_mean = all_analyses[last_hz][7]['mean_spin']
        
        print(f"\n  At h^z = {last_hz}:")
        print(f"    z=3 (low coordination):  <S_z> = {z3_mean:.4f}, {z3_strong:.1f}% strong")
        print(f"    z=7 (high coordination): <S_z> = {z7_mean:.4f}, {z7_strong:.1f}% strong")
        
        if z3_strong > z7_strong:
            print(f"    → Low-coordination sites show STRONGER spin polarization")
        else:
            print(f"    → High-coordination sites show STRONGER spin polarization")
    
    print("\n" + "=" * 90)
    print("✅ ANALYSIS COMPLETED")
    print("=" * 90)
    print(f"\nResults saved to: {args.output_dir}/")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
