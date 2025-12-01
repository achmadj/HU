"""
Finite Size Effect Analysis untuk Center Model Penrose Lattice
Analisis lokalisasi elektron di E=2 untuk berbagai ukuran sistem (N)
"""

import numpy as np
import pickle
import time
from typing import Dict, List
from collections import defaultdict, Counter


def generate_center_model(vertex_coords, edge_list):
    """
    Generate center model dari vertex model
    
    Parameters:
        vertex_coords: numpy array of vertex coordinates
        edge_list: numpy array of edges
    
    Returns:
        centers: numpy array of center coordinates
        dual_edges: list of tuples (i, j) untuk dual edges
    """
    N = len(vertex_coords)
    
    # Build adjacency list
    adj = defaultdict(set)
    edges_set = set()
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
        edges_set.add(tuple(sorted((u, v))))
    
    # Detect rhombi (4-cycles)
    faces = []
    processed_quads = set()
    
    for u in range(N):
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                v = neighbors[i]
                w = neighbors[j]
                
                # Find common neighbor of v and w (excluding u)
                common = adj[v].intersection(adj[w])
                common.discard(u)
                
                if common:
                    x = common.pop()
                    
                    # Verify valid rhombus
                    quad_edges = [
                        tuple(sorted((u, v))),
                        tuple(sorted((v, x))),
                        tuple(sorted((x, w))),
                        tuple(sorted((w, u)))
                    ]
                    
                    if all(e in edges_set for e in quad_edges):
                        quad = tuple(sorted((u, v, w, x)))
                        if quad not in processed_quads:
                            faces.append(quad)
                            processed_quads.add(quad)
    
    # Compute center coordinates
    centers = []
    for quad in faces:
        coords = vertex_coords[list(quad)]
        center = np.mean(coords, axis=0)
        centers.append(center)
    
    centers = np.array(centers)
    
    # Build dual edges
    edge_to_faces = defaultdict(list)
    
    for idx, quad in enumerate(faces):
        for i in range(4):
            for j in range(i+1, 4):
                edge = tuple(sorted((quad[i], quad[j])))
                if edge in edges_set:
                    edge_to_faces[edge].append(idx)
    
    dual_edges = []
    for edge, face_indices in edge_to_faces.items():
        if len(face_indices) == 2:
            f1, f2 = face_indices
            dual_edges.append((f1, f2))
    
    return centers, dual_edges


class FiniteSizeAnalyzer:
    """Analisis finite size effect untuk center model"""
    
    def __init__(self):
        self.results = []  # List of dicts dengan data per iterasi
    
    def load_and_analyze(self, iteration: int) -> Dict:
        """
        Load vertex model, generate center model, dan analisis untuk satu iterasi
        
        Returns:
            Dict dengan N, coordination distribution, dan localization
        """
        # Load vertex model
        vertex_filename = f'vertex_model/data/penrose_lattice_iter{iteration}.npz'
        
        try:
            vertex_data = np.load(vertex_filename)
        except FileNotFoundError:
            print(f"  ✗ File not found: {vertex_filename}")
            return None
        
        vertex_coords = vertex_data['vertex_coords']
        edge_list = vertex_data['edge_list']
        
        print(f"\n[Iteration {iteration}] Generating center model...")
        
        # Generate center model
        centers, dual_edges = generate_center_model(vertex_coords, edge_list)
        
        N = len(centers)
        E = len(dual_edges)
        
        print(f"  Center model: N={N}, E={E}")
        
        # Build Hamiltonian
        H = np.zeros((N, N), dtype=np.float64)
        for (i, j) in dual_edges:
            H[i, j] = -1.0
            H[j, i] = -1.0
        
        # Diagonalize
        print(f"  Diagonalizing {N}×{N} matrix...")
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        # Find state closest to E = 2
        target_energy = 2.0
        energy_diff = np.abs(eigenvalues - target_energy)
        closest_idx = np.argmin(energy_diff)
        closest_energy = eigenvalues[closest_idx]
        
        print(f"  State at E = {closest_energy:.6f}")
        
        # Get wavefunction
        psi = eigenvectors[:, closest_idx]
        prob_density = np.abs(psi)**2
        prob_density = prob_density / np.sum(prob_density)  # Normalize
        
        # Calculate coordination numbers
        degrees = np.zeros(N, dtype=int)
        for (i, j) in dual_edges:
            degrees[i] += 1
            degrees[j] += 1
        
        # Calculate probability distribution by coordination
        prob_by_coord = {}
        site_by_coord = {}
        
        for z in range(1, 10):
            mask = (degrees == z)
            if np.any(mask):
                prob_sum = np.sum(prob_density[mask])
                count = np.sum(mask)
                prob_by_coord[z] = prob_sum
                site_by_coord[z] = {
                    'count': count,
                    'percentage': (count / N) * 100
                }
        
        return {
            'iteration': iteration,
            'N': N,
            'E': E,
            'energy': closest_energy,
            'prob_by_coord': prob_by_coord,
            'site_by_coord': site_by_coord,
            'degrees': degrees
        }
    
    def analyze_multiple_iterations(self, iterations: List[int]) -> None:
        """
        Analisis untuk multiple iterations
        
        Parameters:
            iterations: List of iteration numbers to analyze
        """
        print("="*80)
        print("FINITE SIZE EFFECT ANALYSIS - CENTER MODEL")
        print("Electron Localization at E ≈ 2.0")
        print("="*80)
        
        for iteration in iterations:
            result = self.load_and_analyze(iteration)
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
        print("\n1. ELECTRON DENSITY (%) by Coordination Number")
        print("-" * 80)
        header = f"{'z':<3}"
        for N in N_values:
            header += f" | N={N:>5}"
        print(header)
        print("-" * 80)
        
        # Print electron density data for each coordination number
        for z in all_coords:
            row = f"{z:<3}"
            for result in self.results:
                if z in result['prob_by_coord']:
                    prob = result['prob_by_coord'][z]
                    row += f" | {prob*100:>6.2f}%"
                else:
                    row += " |      - "
            print(row)
        
        print("\n")
        print("2. SITE POPULATION (%) by Coordination Number")
        print("-" * 80)
        header = f"{'z':<3}"
        for N in N_values:
            header += f" | N={N:>5}"
        print(header)
        print("-" * 80)
        
        # Print site population data for each coordination number
        for z in all_coords:
            row = f"{z:<3}"
            for result in self.results:
                if z in result['site_by_coord']:
                    site_pct = result['site_by_coord'][z]['percentage']
                    row += f" | {site_pct:>6.2f}%"
                else:
                    row += " |      - "
            print(row)
        
        print("\n")
        print("3. SUMMARY")
        print("-" * 80)
        print(f"{'Iteration':<12} | {'N':<8} | {'E':<8} | {'Energy':<10} | {'Dominant z':<12}")
        print("-" * 80)
        
        for result in self.results:
            # Find dominant coordination number
            max_z = max(result['prob_by_coord'].items(), key=lambda x: x[1])
            dominant_z, dominant_prob = max_z
            
            print(f"{result['iteration']:<12} | {result['N']:<8} | {result['E']:<8} | "
                  f"{result['energy']:<10.6f} | z={dominant_z} ({dominant_prob*100:.1f}%)")
        
        print("="*80)
    
    def print_detailed_analysis(self) -> None:
        """Print analisis detail untuk setiap iteration"""
        if not self.results:
            return
        
        print("\n")
        print("="*80)
        print("DETAILED ANALYSIS PER ITERATION")
        print("="*80)
        
        for result in self.results:
            print(f"\nIteration {result['iteration']}: N={result['N']}, E={result['E']}")
            print(f"State energy: E = {result['energy']:.6f}")
            print(f"\n{'z':<4} | {'Site Pop.':<15} | {'Electron Density':<20}")
            print("-" * 50)
            
            for z in sorted(result['prob_by_coord'].keys()):
                site_info = result['site_by_coord'][z]
                site_pct = site_info['percentage']
                site_count = site_info['count']
                elec_pct = result['prob_by_coord'][z] * 100
                
                print(f"{z:<4} | {site_pct:>5.2f}% ({site_count:>4} sites) | {elec_pct:>6.2f}%")
            
            # Interpretation
            max_z = max(result['prob_by_coord'].items(), key=lambda x: x[1])
            dominant_z, dominant_prob = max_z
            print(f"\n→ Electrons prefer z={dominant_z} sites ({dominant_prob*100:.1f}%)")


def main():
    """Main program"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Finite Size Analysis for Center Model (E≈2)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        nargs='+',
        default=[2, 3, 4],
        help='List of iteration numbers to analyze (default: 2 3 4)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Print detailed analysis for each iteration'
    )
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FiniteSizeAnalyzer()
    
    # Analyze
    analyzer.analyze_multiple_iterations(args.iterations)
    
    # Print results
    analyzer.print_comparison_table()
    
    if args.detailed:
        analyzer.print_detailed_analysis()
    
    print("\n✅ Finite size analysis completed!")


if __name__ == "__main__":
    main()
