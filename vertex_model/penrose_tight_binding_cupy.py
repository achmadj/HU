"""
Tight Binding Model untuk Penrose Lattice - GPU Accelerated Version
Menggunakan CuPy untuk diagonalisasi cepat pada sistem besar (N > 5k)

Requirements:
    pip install cupy-cuda12x  # Sesuaikan dengan CUDA version
"""

import numpy as np
import pickle
import time
from typing import Dict, Tuple
from numpy.typing import NDArray

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✓ CuPy detected - GPU acceleration enabled")
except ImportError:
    CUPY_AVAILABLE = False
    print("✗ CuPy not available - falling back to NumPy")
    print("  Install with: pip install cupy-cuda12x")


class PenroseTightBindingGPU:
    """
    Tight Binding Model untuk Penrose Lattice dengan GPU acceleration
    
    Attributes:
        vertices (Dict[int, NDArray]): Dictionary vertex_id → koordinat
        edges (Dict[Tuple[int, int], int]): Dictionary (i, j) → arrow_type
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
    
    def load_from_pickle(self, filename: str = 'vertex_model/data/penrose_lattice_data.pkl') -> None:
        """Load data dari pickle file"""
        print(f"\n[LOADING] Reading data from {filename}...")
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.vertices = data['vertices']
        self.edges = data['edges']
        self.N = data['N']
        self.E = data['E']
        self.phi = data.get('phi', (1 + np.sqrt(5)) / 2)
        self.iteration = data.get('iteration', 0)
        
        print(f"  ✓ Loaded N={self.N} vertices")
        print(f"  ✓ Loaded E={self.E} edges")
        print(f"  ✓ Iteration: {self.iteration}")
        print(f"  ✓ Golden ratio φ: {self.phi:.6f}")
    
    def load_from_numpy(self, filename: str = 'penrose_lattice_data.npz') -> None:
        """Load data dari numpy file"""
        print(f"\n[LOADING] Reading data from {filename}...")
        
        data = np.load(filename)
        
        # Reconstruct vertices dict
        vertex_ids = data['vertex_ids']
        vertex_coords = data['vertex_coords']
        self.vertices = {int(vid): coord for vid, coord in zip(vertex_ids, vertex_coords)}
        
        # Reconstruct edges dict
        edge_list = data['edge_list']
        edge_types = data['edge_types']
        self.edges = {(int(e[0]), int(e[1])): int(t) for e, t in zip(edge_list, edge_types)}
        
        self.N = int(data['N'])
        self.E = int(data['E'])
        self.phi = float(data['phi'])
        self.iteration = int(data['iteration'])
        
        print(f"  ✓ Loaded N={self.N} vertices")
        print(f"  ✓ Loaded E={self.E} edges")
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
        for (i, j), arrow_type in self.edges.items():
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
            is_hermitian = cp.allclose(self.hamiltonian, self.hamiltonian.T)
        else:
            is_hermitian = np.allclose(H, H.T)
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
            
            diag_time = time.time() - start_time
            
            # Transfer results back to CPU
            print(f"  ✓ Diagonalization completed in {diag_time:.2f} seconds")
            print(f"  Transferring results to CPU...")
            
            transfer_start = time.time()
            self.eigenvalues = cp.asnumpy(eigenvalues_gpu)
            self.eigenvectors = cp.asnumpy(eigenvectors_gpu)
            transfer_time = time.time() - transfer_start
            
            total_time = time.time() - start_time
            
            print(f"  ✓ Transfer completed in {transfer_time:.2f} seconds")
            print(f"  ✓ Total time: {total_time:.2f} seconds")
            
        else:
            # CPU diagonalization
            self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.hamiltonian)
            diag_time = time.time() - start_time
            print(f"  ✓ Diagonalization completed in {diag_time:.2f} seconds")
        
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
    
    def analyze_wavefunction(self, state_index: int) -> Dict[str, any]:
        """Analisis wavefunction untuk state tertentu"""
        if state_index < 0 or state_index >= self.N:
            raise ValueError(f"State index must be in range [0, {self.N-1}]")
        
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
            gpu_name = device.attributes.get('Name', b'Unknown').decode()
            print(f"  GPU Name: {gpu_name}")
        except:
            try:
                # Alternative method
                props = cp.cuda.runtime.getDeviceProperties(device.id)
                gpu_name = props['name'].decode()
                print(f"  GPU Name: {gpu_name}")
            except:
                print(f"  GPU Name: Unknown")
        
        # Memory info
        mempool = cp.get_default_memory_pool()
        print(f"  Used memory: {mempool.used_bytes() / (1024**2):.2f} MB")
        print(f"  Total memory: {mempool.total_bytes() / (1024**2):.2f} MB")
        
        # CUDA info
        try:
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            print(f"  CUDA Version: {cuda_version}")
        except:
            print(f"  CUDA Version: Unknown")
        
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            print(f"  Device Count: {device_count}")
        except:
            pass


def print_separator(char: str = "=", length: int = 80) -> None:
    """Print separator line"""
    print(char * length)


def main() -> None:
    """Main program untuk tight binding dengan GPU acceleration"""
    print_separator()
    print("PENROSE LATTICE - TIGHT BINDING MODEL (GPU ACCELERATED)")
    print("Nearest-Neighbor Hopping with ε₀=0, t=1")
    print_separator()
    
    # Check GPU availability
    if CUPY_AVAILABLE:
        print("\n✓ GPU acceleration available")
        try:
            device = cp.cuda.Device()
            try:
                gpu_name = device.attributes.get('Name', b'Unknown').decode()
                print(f"  GPU: {gpu_name}")
            except:
                try:
                    props = cp.cuda.runtime.getDeviceProperties(device.id)
                    gpu_name = props['name'].decode()
                    print(f"  GPU: {gpu_name}")
                except:
                    print(f"  GPU: Device {device.id}")
        except:
            print("  GPU info unavailable")
    else:
        print("\n✗ GPU acceleration not available (using CPU)")
        print("  To enable GPU: pip install cupy-cuda12x")
    
    # Inisialisasi model
    tb_model = PenroseTightBindingGPU(epsilon_0=0.0, t=1.0, use_gpu=True)
    
    # Load data
    try:
        tb_model.load_from_pickle('vertex_model/data/penrose_lattice_data.pkl')
    except FileNotFoundError:
        print("\n✗ vertex_model/data/penrose_lattice_data.pkl not found")
        print("  Run penrose_tiling_fast.py first to generate data")
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
    
    # Build adjacency list to count neighbors
    adjacency = {i: set() for i in range(tb_model.N)}
    for (i, j) in tb_model.edges.keys():
        adjacency[i].add(j)
        adjacency[j].add(i)
    
    # Count coordination numbers
    coordination_numbers = {i: len(adjacency[i]) for i in range(tb_model.N)}
    
    # Statistics
    from collections import Counter
    z_distribution = Counter(coordination_numbers.values())
    
    # Find zero-energy states (E ≈ 0)
    energy_threshold = 1e-10
    zero_energy_indices = np.where(np.abs(tb_model.eigenvalues) < energy_threshold)[0]
    
    if len(zero_energy_indices) == 0:
        # Find closest states to E=0
        abs_energies = np.abs(tb_model.eigenvalues)
        sorted_indices = np.argsort(abs_energies)
        zero_energy_indices = sorted_indices[:min(3, len(sorted_indices))]
    
    # Analyze electron localization at zero energy states
    print("\n")
    print_separator()
    print("ZERO MODE LOCALIZATION ANALYSIS")
    print_separator()
    
    if len(zero_energy_indices) > 0:
        print(f"\nAnalyzing {len(zero_energy_indices)} states with E ≈ 0...")
        
        # Convert coordination_numbers dict to array
        neighbor_counts = np.array([coordination_numbers[i] for i in range(tb_model.N)])
        
        # Get zero-energy eigenvectors (already in CPU from diagonalize())
        zero_vectors = tb_model.eigenvectors[:, zero_energy_indices]
        
        # Calculate average probability density across all zero states
        average_density = np.sum(np.abs(zero_vectors)**2, axis=1)
        # Normalize to sum = 1
        average_density /= np.sum(average_density)
        
        # Group by coordination number
        total_probability_distribution = {}
        for z in sorted(z_distribution.keys()):
            mask = (neighbor_counts == z)
            total_probability_distribution[z] = np.sum(average_density[mask])
        
        # Print results
        print("\nElectron localization at E=0 by coordination number (z):")
        print(f"{'z':<4} | {'Atom Population':<20} | {'Electron Localization':<20}")
        print("-" * 50)
        
        for z in sorted(total_probability_distribution.keys()):
            atom_count = z_distribution[z]
            atom_percent = (atom_count / tb_model.N) * 100
            electron_percent = total_probability_distribution[z] * 100
            
            print(f"{z:<4} | {atom_percent:6.2f}% | {electron_percent:6.2f}%")
        
        print("\nINTERPRETATION:")
        val_z3 = total_probability_distribution.get(3, 0) * 100
        if val_z3 > 80:
            print(f"✓ CONFIRMED! Majority of electrons ({val_z3:.1f}%) localized at z=3 sites.")
        elif val_z3 > 50:
            print(f"→ Electrons prefer z=3 sites ({val_z3:.1f}%), but not exclusively.")
        else:
            print(f"⚠ UNCLEAR. Electrons distributed across sites (only {val_z3:.1f}% at z=3).")
    else:
        print("\nNo zero-energy states found for localization analysis.")
    
    # GPU memory cleanup
    if tb_model.use_gpu:
        print("\n")
        print_separator()
        print("GPU MEMORY CLEANUP")
        print_separator()
        mempool = cp.get_default_memory_pool()
        print(f"  Used memory before cleanup: {mempool.used_bytes() / (1024**2):.2f} MB")
        mempool.free_all_blocks()
        print(f"  Used memory after cleanup: {mempool.used_bytes() / (1024**2):.2f} MB")
    
    # Final summary
    print("\n")
    print_separator()
    print("✅ Tight binding analysis completed!")
    print(f"📊 N={tb_model.N} vertices processed")
    print(f"🚀 Device: {'GPU (CuPy)' if tb_model.use_gpu else 'CPU (NumPy)'}")
    print_separator()
    print()


if __name__ == "__main__":
    main()
