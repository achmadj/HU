"""
Tight Binding Model untuk Penrose Lattice Center Model - GPU Accelerated
Menggunakan CuPy untuk diagonalisasi cepat pada sistem besar (N > 5k)
Plot wavefunction di E ≈ 2

Requirements:
    pip install cupy-cuda12x  # Sesuaikan dengan CUDA version
"""

import numpy as np
import pickle
import time
from typing import Dict, Tuple
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from collections import deque

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✓ CuPy detected - GPU acceleration enabled")
except ImportError:
    CUPY_AVAILABLE = False
    print("✗ CuPy not available - falling back to NumPy")
    print("  Install with: pip install cupy-cuda12x")


class CenterPenroseTightBindingGPU:
    """
    Tight Binding Model untuk Penrose Center Model dengan GPU acceleration
    
    Attributes:
        vertices (Dict[int, NDArray]): Dictionary vertex_id → koordinat
        edges (Dict[Tuple[int, int], int]): Dictionary (i, j) → edge_type
        N (int): Jumlah vertices
        E (int): Jumlah edges
        epsilon_0 (float): On-site energy
        t (float): Hopping parameter
        hamiltonian: Hamiltonian matrix (CPU atau GPU)
        eigenvalues: Energy eigenvalues
        eigenvectors: Eigenstates
        use_gpu (bool): Flag untuk GPU usage
    """
    
    def __init__(self, epsilon_0: float = 0.0, t: float = 1.0, use_gpu: bool = True):
        """
        Inisialisasi Tight Binding Model
        
        Input:
            epsilon_0 (float): On-site energy (default: 0.0)
            t (float): Hopping parameter (default: 1.0)
            use_gpu (bool): Gunakan GPU jika tersedia (default: True)
        """
        self.vertices: Dict[int, NDArray[np.float64]] = {}
        self.edges: Dict[Tuple[int, int], int] = {}
        self.N: int = 0
        self.E: int = 0
        self.epsilon_0: float = epsilon_0
        self.t: float = t
        self.hamiltonian = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.phi: float = 0.0
        self.iteration: int = 0
        self.use_gpu: bool = use_gpu and CUPY_AVAILABLE
        
        if use_gpu and not CUPY_AVAILABLE:
            print("  ⚠️  GPU requested but CuPy not available, using CPU")
    
    def load_from_pickle(self, filename: str = 'center_model/data/center_model_penrose_lattice.pkl') -> None:
        """Load center model data dari pickle file"""
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
        """Load center model data dari numpy file"""
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
        Konstruksi Hamiltonian matrix
        
        H_ij = ε₀ δ_ij - t (jika i dan j tetangga)
        """
        print(f"\n[HAMILTONIAN] Building {self.N}×{self.N} Hamiltonian matrix...")
        print(f"  Device: {'GPU (CuPy)' if self.use_gpu else 'CPU (NumPy)'}")
        
        start_time = time.time()
        
        # Build on CPU first (more efficient for construction)
        H = np.zeros((self.N, self.N), dtype=np.float64)
        
        # On-site energy (diagonal)
        for i in range(self.N):
            H[i, i] = self.epsilon_0
        
        # Hopping terms (off-diagonal)
        hopping_count = 0
        for (i, j), edge_type in self.edges.items():
            H[i, j] = -self.t
            H[j, i] = -self.t
            hopping_count += 1
        
        build_time = time.time() - start_time
        
        # Transfer to GPU if needed
        if self.use_gpu:
            transfer_start = time.time()
            self.hamiltonian = cp.asarray(H)
            transfer_time = time.time() - transfer_start
            memory_mb = H.nbytes / (1024**2)
            print(f"  ✓ Matrix built in {build_time:.3f} seconds")
            print(f"  ✓ Transferred to GPU in {transfer_time:.3f} seconds")
            print(f"  ✓ GPU memory used: {memory_mb:.2f} MB")
        else:
            self.hamiltonian = H
            print(f"  ✓ Matrix built in {build_time:.3f} seconds")
        
        print(f"  ✓ Matrix size: {self.N}×{self.N} ({self.N**2:,} elements)")
        print(f"  ✓ On-site energy ε₀: {self.epsilon_0}")
        print(f"  ✓ Hopping parameter t: {self.t}")
        print(f"  ✓ Total hopping terms: {hopping_count}")
        
        # Check Hermiticity
        if self.use_gpu:
            is_hermitian = bool(cp.allclose(self.hamiltonian, self.hamiltonian.T))
        else:
            is_hermitian = bool(np.allclose(self.hamiltonian, self.hamiltonian.T))
        print(f"  ✓ Hermiticity check: {'PASSED' if is_hermitian else 'FAILED'}")
    
    def diagonalize(self) -> None:
        """
        Diagonalisasi Hamiltonian
        Menggunakan cupy.linalg.eigh (GPU) atau numpy.linalg.eigh (CPU)
        """
        print(f"\n[DIAGONALIZATION] Solving eigenvalue problem...")
        print(f"  Matrix size: {self.N}×{self.N} ({self.N**2:,} elements)")
        print(f"  Device: {'GPU (CuPy)' if self.use_gpu else 'CPU (NumPy)'}")
        print(f"  Starting diagonalization...")
        
        start_time = time.time()
        
        if self.use_gpu:
            # GPU diagonalization
            eigenvalues_gpu, eigenvectors_gpu = cp.linalg.eigh(self.hamiltonian)
            
            # Transfer back to CPU
            transfer_start = time.time()
            self.eigenvalues = cp.asnumpy(eigenvalues_gpu)
            self.eigenvectors = cp.asnumpy(eigenvectors_gpu)
            transfer_time = time.time() - transfer_start
            
            elapsed_time = time.time() - start_time
            diag_time = elapsed_time - transfer_time
            
            print(f"  ✓ GPU diagonalization completed in {diag_time:.2f} seconds")
            print(f"  ✓ Transfer back to CPU in {transfer_time:.2f} seconds")
            print(f"  ✓ Total time: {elapsed_time:.2f} seconds")
            
        else:
            # CPU diagonalization
            eigenvalues, eigenvectors = np.linalg.eigh(self.hamiltonian)
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
            elapsed_time = time.time() - start_time
            print(f"  ✓ CPU diagonalization completed in {elapsed_time:.2f} seconds")
        
        print(f"  ✓ Found {len(self.eigenvalues)} eigenvalues")
        print(f"  ✓ Energy range: [{np.min(self.eigenvalues):.6f}, {np.max(self.eigenvalues):.6f}]")
        print(f"  ✓ Energy bandwidth: {np.max(self.eigenvalues) - np.min(self.eigenvalues):.6f}")
    
    def get_statistics(self) -> Dict[str, float]:
        """Hitung statistik dari eigenvalues"""
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
        """Analisis wavefunction untuk state tertentu"""
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
        size_threshold = 1e-16
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
    
    def get_gpu_info(self) -> None:
        """Print GPU information jika tersedia"""
        if not CUPY_AVAILABLE:
            print("\n[GPU INFO] CuPy not available")
            return
        
        print("\n[GPU INFO]")
        device = cp.cuda.Device()
        print(f"  Device ID: {device.id}")
        
        # Try to get GPU name
        try:
            print(f"  Device name: {device.compute_capability}")
        except:
            # Memory info
            mempool = cp.get_default_memory_pool()
            print(f"  Used memory: {mempool.used_bytes() / (1024**2):.2f} MB")
            print(f"  Total memory: {mempool.total_bytes() / (1024**2):.2f} MB")
        
        # CUDA info
        try:
            print(f"  CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        except:
            pass
        
        try:
            print(f"  Compute capability: {device.compute_capability}")
        except:
            pass


def print_separator(char: str = "=", length: int = 80) -> None:
    """Print separator line"""
    print(char * length)


def main() -> None:
    """Main program untuk tight binding dengan GPU acceleration"""
    print_separator()
    print("PENROSE CENTER MODEL - TIGHT BINDING (GPU ACCELERATED)")
    print("Nearest-Neighbor Hopping with ε₀=0, t=1")
    print_separator()
    
    # Check GPU availability
    if CUPY_AVAILABLE:
        print("\n✓ GPU acceleration available")
        try:
            device = cp.cuda.Device()
            print(f"  Device ID: {device.id}")
            mempool = cp.get_default_memory_pool()
            print(f"  Free memory: {mempool.free_bytes() / (1024**2):.2f} MB")
        except:
            print("  Could not access GPU info")
    else:
        print("\n✗ GPU acceleration not available (using CPU)")
        print("  To enable GPU: pip install cupy-cuda12x")
    
    # Inisialisasi model
    tb_model = CenterPenroseTightBindingGPU(epsilon_0=0.0, t=1.0, use_gpu=True)
    
    # Load data
    try:
        tb_model.load_from_pickle('center_model/data/center_model_penrose_lattice.pkl')
    except FileNotFoundError:
        print("\n✗ center_model/data/center_model_penrose_lattice.pkl not found")
        print("  Run center_penrose_tiling_fast.py first to generate center model")
        return
    
    # GPU info
    if tb_model.use_gpu:
        tb_model.get_gpu_info()
    
    # Build Hamiltonian
    tb_model.build_hamiltonian()
    
    # Diagonalize
    tb_model.diagonalize()
    
    # Statistics
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
    energy_diff = np.abs(tb_model.eigenvalues - target_energy)
    sorted_indices = np.argsort(energy_diff)
    
    # Ambil 1 state terdekat ke E=2
    closest_idx = sorted_indices[0]
    closest_energy = tb_model.eigenvalues[closest_idx]
    
    print(f"\nFound closest state to E = {target_energy}:")
    print(f"  State {closest_idx}: E = {closest_energy:.6f}")
    
    # Plot wavefunction untuk state di E≈2
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    tb_model.plot_wavefunction(closest_idx, ax)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    wavefunction_filename = f'center_model/imgs/center_wavefunctions_E2_gpu.png'
    plt.savefig(wavefunction_filename, dpi=500, bbox_inches='tight')
    print(f"\n  ✓ Saved wavefunction plots: {wavefunction_filename}")
    plt.close()
    
    # GPU memory cleanup
    if tb_model.use_gpu:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.cuda.PinnedMemoryPool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        print("\n  ✓ GPU memory released")
    
    # Final summary
    print("\n")
    print_separator()
    print("✅ Center model tight binding analysis completed!")
    print(f"📊 N={tb_model.N} vertices processed")
    print(f"🚀 Device: {'GPU (CuPy)' if tb_model.use_gpu else 'CPU (NumPy)'}")
    print_separator()
    print()


if __name__ == "__main__":
    main()
