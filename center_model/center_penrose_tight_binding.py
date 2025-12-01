"""
Tight Binding Model untuk Penrose Lattice Center Model
Diagonalisasi Hamiltonian dengan nearest-neighbor hopping
Plot wavefunction di E ≈ 2
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import time
from collections import Counter, deque


class CenterPenroseTightBinding:
    """
    Tight Binding Model untuk Penrose Lattice Center Model
    
    Attributes:
        vertices (Dict[int, NDArray[np.float64]]): Dictionary vertex_id → koordinat
        edges (Dict[Tuple[int, int], int]): Dictionary (i, j) → edge_type
        N (int): Jumlah vertices (sites)
        E (int): Jumlah edges (bonds)
        epsilon_0 (float): On-site energy (default: 0.0)
        t (float): Hopping parameter (default: 1.0)
        hamiltonian (NDArray[np.float64]): Hamiltonian matrix (N × N)
        eigenvalues (NDArray[np.float64]): Energy eigenvalues
        eigenvectors (NDArray[np.float64]): Eigenstates
    """
    
    def __init__(self, epsilon_0: float = 0.0, t: float = 1.0) -> None:
        """
        Inisialisasi Tight Binding Model
        
        Input:
            epsilon_0 (float): On-site energy (default: 0.0)
            t (float): Hopping parameter (default: 1.0)
        """
        self.vertices: Dict[int, NDArray[np.float64]] = {}
        self.edges: Dict[Tuple[int, int], int] = {}
        self.N: int = 0
        self.E: int = 0
        self.epsilon_0: float = epsilon_0
        self.t: float = t
        self.hamiltonian: NDArray[np.float64] = np.array([])
        self.eigenvalues: NDArray[np.float64] = np.array([])
        self.eigenvectors: NDArray[np.float64] = np.array([])
        self.phi: float = 0.0
        self.iteration: int = 0
    
    def load_from_pickle(self, filename: str = 'center_model/data/center_model_penrose_lattice.pkl') -> None:
        """
        Load data dari pickle file
        
        Input:
            filename (str): Nama file pickle
        
        Output:
            None (modifikasi attributes in-place)
        """
        print(f"\n[LOADING] Reading data from {filename}...")
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.vertices = data['vertices']
        self.edges = data['edges']
        self.N = data['N']
        self.E = data['E']
        self.phi = data.get('phi', (1 + np.sqrt(5)) / 2)
        self.iteration = data.get('iteration', 0)
        
        print(f"  ✓ Loaded N={self.N} vertices (center sites)")
        print(f"  ✓ Loaded E={self.E} edges (dual bonds)")
        print(f"  ✓ Iteration: {self.iteration}")
        print(f"  ✓ Golden ratio φ: {self.phi:.6f}")
    
    def load_from_numpy(self, filename: str = 'center_model/data/center_model_penrose_lattice.npz') -> None:
        """
        Load data dari numpy file
        
        Input:
            filename (str): Nama file npz
        
        Output:
            None (modifikasi attributes in-place)
        """
        print(f"\n[LOADING] Reading data from {filename}...")
        
        data = np.load(filename)
        
        # Reconstruct vertices dict
        vertex_ids = data['vertex_ids']
        vertex_coords = data['vertex_coords']
        self.vertices = {int(vid): coord for vid, coord in zip(vertex_ids, vertex_coords)}
        
        # Reconstruct edges dict
        edge_list = data['edge_list']
        self.edges = {(int(e[0]), int(e[1])): 1 for e in edge_list}
        
        self.N = int(data['N'])
        self.E = int(data['E'])
        self.phi = float(data.get('phi', (1 + np.sqrt(5)) / 2))
        self.iteration = int(data.get('iteration', 0))
        
        print(f"  ✓ Loaded N={self.N} vertices (center sites)")
        print(f"  ✓ Loaded E={self.E} edges (dual bonds)")
        print(f"  ✓ Iteration: {self.iteration}")
        print(f"  ✓ Golden ratio φ: {self.phi:.6f}")
    
    def build_hamiltonian(self) -> None:
        """
        Konstruksi Hamiltonian matrix untuk tight binding model
        
        H_ij = ε₀ δ_ij - t (jika i dan j tetangga terdekat)
        
        Output:
            None (modifikasi self.hamiltonian in-place)
        """
        print(f"\n[HAMILTONIAN] Building {self.N}×{self.N} Hamiltonian matrix...")
        
        # Inisialisasi matrix NxN dengan zeros
        H: NDArray[np.float64] = np.zeros((self.N, self.N), dtype=np.float64)
        
        # On-site energy (diagonal elements)
        for i in range(self.N):
            H[i, i] = self.epsilon_0
        
        # Hopping terms (off-diagonal)
        # Symmetrize the edges
        hopping_count: int = 0
        
        for (i, j), edge_type in self.edges.items():
            H[i, j] = -self.t
            H[j, i] = -self.t
            hopping_count += 1
        
        self.hamiltonian = H
        
        print(f"  ✓ Matrix size: {self.N}×{self.N}")
        print(f"  ✓ On-site energy ε₀: {self.epsilon_0}")
        print(f"  ✓ Hopping parameter t: {self.t}")
        print(f"  ✓ Total hopping terms: {hopping_count} (from {self.E} directed edges)")
        
        # Check if Hamiltonian is Hermitian
        is_hermitian: bool = np.allclose(H, H.T)
        print(f"  ✓ Hermiticity check: {'PASSED' if is_hermitian else 'FAILED'}")
    
    def diagonalize(self) -> None:
        """
        Diagonalisasi Hamiltonian untuk mendapatkan eigenvalues dan eigenvectors
        
        Output:
            None (modifikasi self.eigenvalues dan self.eigenvectors in-place)
        """
        print(f"\n[DIAGONALIZATION] Solving eigenvalue problem...")
        print(f"  Matrix size: {self.N}×{self.N} ({self.N**2:,} elements)")
        print(f"  Starting diagonalization...")
        
        # Timer start
        start_time = time.time()
        
        # Diagonalisasi menggunakan numpy.linalg.eigh (untuk Hermitian matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(self.hamiltonian)
        
        # Timer end
        elapsed_time = time.time() - start_time
        
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        print(f"  ✓ Diagonalization completed in {elapsed_time:.2f} seconds")
        print(f"  ✓ Found {len(self.eigenvalues)} eigenvalues")
        print(f"  ✓ Energy range: [{np.min(self.eigenvalues):.6f}, {np.max(self.eigenvalues):.6f}]")
        print(f"  ✓ Energy bandwidth: {np.max(self.eigenvalues) - np.min(self.eigenvalues):.6f}")
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Hitung statistik dari eigenvalues
        
        Output:
            Dict[str, float]: Dictionary berisi statistik
        """
        return {
            'mean_energy': float(np.mean(self.eigenvalues)),
            'std_energy': float(np.std(self.eigenvalues)),
            'min_energy': float(np.min(self.eigenvalues)),
            'max_energy': float(np.max(self.eigenvalues)),
            'bandwidth': float(np.max(self.eigenvalues) - np.min(self.eigenvalues)),
            'zero_energy_count': int(np.sum(np.abs(self.eigenvalues) < 1e-12))
        }
    
    def compute_bipartite_sublattices(self) -> Dict[int, str]:
        """
        Compute bipartite sublattice assignment menggunakan BFS
        
        Output:
            Dict[int, str]: mapping vertex_id → 'A' atau 'B'
        """
        # Build adjacency list
        adj = {i: [] for i in range(self.N)}
        for (i, j) in self.edges.keys():
            adj[i].append(j)
            adj[j].append(i)
        
        sublattice_map = {}
        queue = deque()
        
        for start in range(self.N):
            if start not in sublattice_map:
                sublattice_map[start] = 'A'
                queue.append(start)
                
                while queue:
                    u = queue.popleft()
                    lbl = sublattice_map[u]
                    opp = 'B' if lbl == 'A' else 'A'
                    
                    for v in adj[u]:
                        if v not in sublattice_map:
                            sublattice_map[v] = opp
                            queue.append(v)
        
        return sublattice_map
    
    def analyze_wavefunction(self, state_index: int) -> Dict[str, any]:
        """
        Analisis wavefunction untuk state tertentu
        
        Input:
            state_index (int): Index state (eigenvalue index)
        
        Output:
            Dict berisi analisis wavefunction
        """
        if state_index < 0 or state_index >= self.N:
            raise ValueError(f"state_index {state_index} out of range [0, {self.N})")
        
        psi = self.eigenvectors[:, state_index]
        energy = self.eigenvalues[state_index]
        
        # Probability density
        prob_density = np.abs(psi)**2
        
        # Participation ratio
        participation_ratio = 1.0 / np.sum(prob_density**2)
        
        # Localization measure
        max_amplitude = np.max(prob_density)
        
        return {
            'state_index': state_index,
            'energy': float(energy),
            'wavefunction': psi,
            'probability_density': prob_density,
            'participation_ratio': float(participation_ratio),
            'max_amplitude': float(max_amplitude),
            'normalization': float(np.sum(prob_density))
        }
    
    def plot_wavefunction(self, state_index: int, ax: plt.Axes, 
                         size_scale: float = 500.0) -> None:
        """
        Plot wavefunction pada axes
        
        Input:
            state_index (int): Index state
            ax (plt.Axes): Matplotlib axes
            size_scale (float): Scale untuk ukuran marker
        """
        # Analisis wavefunction
        analysis = self.analyze_wavefunction(state_index)
        
        psi = analysis['wavefunction']
        prob_density = analysis['probability_density']
        energy = analysis['energy']
        pr = analysis['participation_ratio']
        
        # Get coordinates
        coords = np.array([self.vertices[i] for i in range(self.N)])
        
        # Rotasi -18 derajat
        theta = -18.0
        theta_rad = np.deg2rad(theta)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        rotation_matrix = np.array([[cos_theta, -sin_theta], 
                                    [sin_theta, cos_theta]])
        coords_rotated = coords @ rotation_matrix.T
        x_coords = coords_rotated[:, 0]
        y_coords = coords_rotated[:, 1]
        
        # Compute sublattices
        sublattice_map = self.compute_bipartite_sublattices()
        
        # Plot edges (jika tidak terlalu banyak)
        if self.N <= 10000:
            from matplotlib.collections import LineCollection
            lines = []
            for (i, j) in self.edges.keys():
                lines.append([coords_rotated[i], coords_rotated[j]])
            lc = LineCollection(lines, colors='gray', linewidths=0.3, alpha=0.3, zorder=1)
            ax.add_collection(lc)
        
        # Size and color thresholds
        size_threshold = 1e-15
        color_threshold = 1e-9
        
        # Compute sizes
        sizes = np.where(prob_density < size_threshold, 0.01,
                        np.where(prob_density >= color_threshold, 20, 1.0))
        
        # Plot all centers with probability density
        scatter = ax.scatter(x_coords, y_coords, s=sizes, c=prob_density, 
                            cmap='hot', alpha=0.8, edgecolors='black', 
                            linewidth=0.3, zorder=2, vmin=0, vmax=np.max(prob_density))
        
        # Highlight high probability sites with sublattice colors
        high_prob_mask = prob_density >= color_threshold
        if np.any(high_prob_mask):
            indices_high = np.where(high_prob_mask)[0]
            indices_A = [i for i in indices_high if sublattice_map[i] == 'A']
            indices_B = [i for i in indices_high if sublattice_map[i] == 'B']
            
            if indices_A:
                ax.scatter(x_coords[indices_A], y_coords[indices_A], 
                          s=sizes[indices_A], c='red', 
                          alpha=1.0, edgecolors='black', linewidth=0.5, zorder=3, 
                          label='Sublattice A')
            if indices_B:
                ax.scatter(x_coords[indices_B], y_coords[indices_B], 
                          s=sizes[indices_B], c='blue', 
                          alpha=1.0, edgecolors='black', linewidth=0.5, zorder=3, 
                          label='Sublattice B')
        
        # Title
        ax.set_title(rf"Center Model Wavefunction at E = {energy:.2f}", 
                    fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Legend if sublattices shown
        if np.any(high_prob_mask):
            ax.legend(loc='upper right', fontsize=8)


def print_separator(char: str = "=", length: int = 80) -> None:
    """Print separator line"""
    print(char * length)


def main() -> None:
    """
    Main program untuk tight binding model center Penrose lattice
    """
    print_separator()
    print("PENROSE LATTICE CENTER MODEL - TIGHT BINDING")
    print("Nearest-Neighbor Hopping with ε₀=0, t=1")
    print_separator()
    
    # Inisialisasi model
    tb_model = CenterPenroseTightBinding(epsilon_0=0.0, t=1.0)
    
    # Load data (gunakan pickle)
    tb_model.load_from_pickle('center_model/data/center_model_penrose_lattice.pkl')
    
    # Build Hamiltonian
    tb_model.build_hamiltonian()
    
    # Diagonalisasi
    tb_model.diagonalize()
    
    # Statistik
    print("\n")
    print_separator()
    print("ENERGY STATISTICS")
    print_separator()
    
    stats = tb_model.get_statistics()
    print(f"Mean energy:          {stats['mean_energy']:12.6f}")
    print(f"Std deviation:        {stats['std_energy']:12.6f}")
    print(f"Min energy:           {stats['min_energy']:12.6f}")
    print(f"Max energy:           {stats['max_energy']:12.6f}")
    print(f"Bandwidth:            {stats['bandwidth']:12.6f}")
    print(f"Zero-energy states:   {stats['zero_energy_count']:12d}")
    
    # Find states near E ≈ 2
    print("\n")
    print_separator()
    print("FINDING STATES NEAR E ≈ 2")
    print_separator()
    
    target_energy = 2.0
    energy_threshold = 0.01  # Threshold untuk menghitung fraksi state di E=2
    
    # Hitung fraksi state di sekitar E=2
    states_near_E2 = np.sum(np.abs(tb_model.eigenvalues - target_energy) < energy_threshold)
    total_states = tb_model.N
    fraction_E2 = states_near_E2 / total_states
    
    print(f"\nState Fraction Analysis (E = {target_energy} ± {energy_threshold}):")
    print(f"  N (states near E=2):  {states_near_E2}")
    print(f"  N₀ (total states):    {total_states}")
    print(f"  f = N/N₀:             {fraction_E2:.6f} ({fraction_E2*100:.4f}%)")
    
    # Find closest state
    energy_diff = np.abs(tb_model.eigenvalues - target_energy)
    sorted_indices = np.argsort(energy_diff)
    
    closest_idx = sorted_indices[0]
    closest_energy = tb_model.eigenvalues[closest_idx]
    
    print(f"\nClosest state to E = {target_energy}:")
    print(f"  State {closest_idx}: E = {closest_energy:.6f}")
    
    # Plot wavefunction untuk state di E≈2
    print("\n")
    print_separator()
    print("WAVEFUNCTION VISUALIZATION")
    print_separator()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    tb_model.plot_wavefunction(closest_idx, ax)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    wavefunction_filename = f'center_model/imgs/center_wavefunctions_E2.png'
    plt.savefig(wavefunction_filename, dpi=500, bbox_inches='tight')
    print(f"\n  ✓ Saved wavefunction plots: {wavefunction_filename}")
    plt.close()
    
    print("\n")
    print_separator()
    print("✅ Center model tight binding analysis completed!")
    print("📊 Wavefunction plots at E≈2 generated")
    print_separator()
    print()


if __name__ == "__main__":
    main()
