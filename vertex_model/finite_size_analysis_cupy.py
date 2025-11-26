"""
Finite Size Effect Analysis untuk Vertex Model Penrose Lattice
Analisis lokalisasi elektron di E=0 untuk berbagai ukuran sistem (N)
GPU Accelerated with CuPy
"""

import numpy as np
import pickle
import time
from typing import Dict, List
from collections import defaultdict

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️  CuPy not available - using CPU (will be slower)")


class FiniteSizeAnalyzer:
    """Analisis finite size effect untuk vertex model"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.results = []  # List of dicts dengan data per iterasi
    
    def load_and_analyze(self, iteration: int) -> Dict:
        """
        Load vertex model dan analisis untuk satu iterasi
        
        Returns:
            Dict dengan N, coordination distribution, dan localization
        """
        # Load data
        filename = 'vertex_model/data/penrose_lattice_data.npz'
        if iteration is not None:
            filename = f'vertex_model/data/penrose_lattice_iter{iteration}.npz'
        
        try:
            data = np.load(filename)
        except FileNotFoundError:
            print(f"  ✗ File not found: {filename}")
            return None
        
        vertices = {int(vid): coord for vid, coord in 
                   enumerate(data['vertex_coords'])}
        edge_list = data['edge_list']
        edges = {(int(e[0]), int(e[1])): 1 for e in edge_list}
        N = int(data['N'])
        E = int(data['E'])
        
        print(f"\n[Iteration {iteration}] N={N}, E={E}")
        
        # Build Hamiltonian
        H = np.zeros((N, N), dtype=np.float64)
        for (i, j) in edges.keys():
            H[i, j] = -1.0
            H[j, i] = -1.0
        
        # Diagonalize
        if self.use_gpu:
            H_gpu = cp.asarray(H)
            eigenvalues_gpu, eigenvectors_gpu = cp.linalg.eigh(H_gpu)
            eigenvalues = cp.asnumpy(eigenvalues_gpu)
            eigenvectors = cp.asnumpy(eigenvectors_gpu)
            del H_gpu, eigenvalues_gpu, eigenvectors_gpu
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        # Ambil state di tengah spektrum
        middle_idx = N // 2
        closest_energy = eigenvalues[middle_idx]
        
        # Get wavefunction
        psi = eigenvectors[:, middle_idx]
        prob_density = np.abs(psi)**2
        
        # Calculate coordination numbers
        degrees = np.zeros(N, dtype=int)
        for (i, j) in edges.keys():
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
            'energy': closest_energy,
            'prob_by_coord': prob_by_coord,
            'degrees': degrees
        }
    
    def analyze_multiple_iterations(self, iterations: List[int]) -> None:
        """
        Analisis untuk multiple iterations
        
        Parameters:
            iterations: List of iteration numbers to analyze
        """
        print("="*80)
        print("FINITE SIZE EFFECT ANALYSIS - VERTEX MODEL")
        print("Electron Localization at E = 0.0")
        print("="*80)
        
        for iteration in iterations:
            result = self.load_and_analyze(iteration)
            if result is not None:
                self.results.append(result)
        
        # GPU memory cleanup
        if self.use_gpu:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
    
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


def main():
    """Main program"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Finite Size Analysis for Vertex Model (E=0)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        nargs='+',
        default=[2, 3, 4],
        help='List of iteration numbers to analyze (default: 2 3 4)'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU computation (disable GPU)'
    )
    args = parser.parse_args()
    
    # Initialize analyzer
    use_gpu = not args.cpu
    analyzer = FiniteSizeAnalyzer(use_gpu=use_gpu)
    
    if analyzer.use_gpu:
        print("✓ Using GPU acceleration (CuPy)")
    else:
        print("ℹ️  Using CPU (NumPy)")
    
    # Analyze
    analyzer.analyze_multiple_iterations(args.iterations)
    
    # Print results
    analyzer.print_comparison_table()
    
    print("\n✅ Finite size analysis completed!")


if __name__ == "__main__":
    main()
