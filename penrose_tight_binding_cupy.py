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
    
    def load_from_pickle(self, filename: str = 'data/penrose_lattice_data.pkl') -> None:
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
        tb_model.load_from_pickle('data/penrose_lattice_data.pkl')
    except FileNotFoundError:
        print("\n✗ data/penrose_lattice_data.pkl not found")
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
    
    # Wavefunction analysis
    print("\n")
    print_separator()
    print("WAVEFUNCTION ANALYSIS")
    print_separator()
    
    # Ground state
    print("\n[Ground State]")
    ground_state = tb_model.analyze_wavefunction(0)
    print(f"  Energy: {ground_state['energy']:.6f}")
    print(f"  Participation ratio: {ground_state['participation_ratio']:.2f} / {tb_model.N}")
    print(f"  Max amplitude: {ground_state['max_amplitude']:.6f}")
    print(f"  Normalization: {ground_state['normalization']:.9f}")
    
    # Middle state (E ≈ 0)
    mid_index = tb_model.N // 2
    print(f"\n[Middle State (index={mid_index}, E≈0)]")
    mid_state = tb_model.analyze_wavefunction(mid_index)
    print(f"  Energy: {mid_state['energy']:.6f}")
    print(f"  Participation ratio: {mid_state['participation_ratio']:.2f} / {tb_model.N}")
    print(f"  Max amplitude: {mid_state['max_amplitude']:.6f}")
    print(f"  Normalization: {mid_state['normalization']:.9f}")
    
    # Highest state
    print(f"\n[Highest State (index={tb_model.N-1})]")
    highest_state = tb_model.analyze_wavefunction(tb_model.N - 1)
    print(f"  Energy: {highest_state['energy']:.6f}")
    print(f"  Participation ratio: {highest_state['participation_ratio']:.2f} / {tb_model.N}")
    print(f"  Max amplitude: {highest_state['max_amplitude']:.6f}")
    print(f"  Normalization: {highest_state['normalization']:.9f}")
    
    # Find states near E=0
    print(f"\n[States near E=0]")
    near_zero = np.abs(tb_model.eigenvalues) < 0.1
    zero_count = np.sum(near_zero)
    print(f"  Number of states with |E| < 0.1: {zero_count}")
    
    if zero_count > 0:
        zero_indices = np.where(near_zero)[0]
        print(f"  First 5 states near zero:")
        for i in zero_indices[:5]:
            print(f"    State {i}: E = {tb_model.eigenvalues[i]:.6f}")
    
    # Analyze PR around E=0 (50 states above and below)
    print(f"\n[Participation Ratio Analysis Around E=0]")
    
    # Find index closest to E=0
    zero_idx = np.argmin(np.abs(tb_model.eigenvalues))
    print(f"  State closest to E=0: index={zero_idx}, E={tb_model.eigenvalues[zero_idx]:.8f}")
    
    # Get 50 states below and 50 states above
    idx_start = max(0, zero_idx - 50)
    idx_end = min(tb_model.N, zero_idx + 51)
    
    print(f"\n  Analyzing states from index {idx_start} to {idx_end-1}")
    print(f"  {'Index':<8} {'Energy':<15} {'PR':<12} {'PR%':<10} {'|ψ|²_max':<12}")
    print("  " + "-" * 70)
    
    for idx in range(idx_start, idx_end):
        state_info = tb_model.analyze_wavefunction(idx)
        energy = state_info['energy']
        pr = state_info['participation_ratio']
        pr_percent = pr / tb_model.N * 100
        max_amp = state_info['max_amplitude']
        
        # Mark the zero state
        marker = " ← E≈0" if idx == zero_idx else ""
        print(f"  {idx:<8} {energy:+.8f}   {pr:>10.1f}  {pr_percent:>8.2f}%  {max_amp:.6f}{marker}")
    
    # Summary statistics for regions
    print(f"\n  Summary Statistics:")
    
    # Below E=0 (50 states)
    below_start = max(0, zero_idx - 50)
    below_end = zero_idx
    if below_end > below_start:
        below_prs = []
        for idx in range(below_start, below_end):
            state_info = tb_model.analyze_wavefunction(idx)
            below_prs.append(state_info['participation_ratio'])
        below_prs = np.array(below_prs)
        print(f"\n  States BELOW E=0 (indices {below_start}-{below_end-1}):")
        print(f"    Mean PR: {np.mean(below_prs):.1f} ({np.mean(below_prs)/tb_model.N*100:.2f}%)")
        print(f"    Median PR: {np.median(below_prs):.1f} ({np.median(below_prs)/tb_model.N*100:.2f}%)")
        print(f"    Min PR: {np.min(below_prs):.1f} ({np.min(below_prs)/tb_model.N*100:.2f}%)")
        print(f"    Max PR: {np.max(below_prs):.1f} ({np.max(below_prs)/tb_model.N*100:.2f}%)")
    
    # At E≈0
    state_zero = tb_model.analyze_wavefunction(zero_idx)
    print(f"\n  State AT E≈0 (index {zero_idx}):")
    print(f"    PR: {state_zero['participation_ratio']:.1f} ({state_zero['participation_ratio']/tb_model.N*100:.2f}%)")
    
    # Above E=0 (50 states)
    above_start = zero_idx + 1
    above_end = min(tb_model.N, zero_idx + 51)
    if above_end > above_start:
        above_prs = []
        for idx in range(above_start, above_end):
            state_info = tb_model.analyze_wavefunction(idx)
            above_prs.append(state_info['participation_ratio'])
        above_prs = np.array(above_prs)
        print(f"\n  States ABOVE E=0 (indices {above_start}-{above_end-1}):")
        print(f"    Mean PR: {np.mean(above_prs):.1f} ({np.mean(above_prs)/tb_model.N*100:.2f}%)")
        print(f"    Median PR: {np.median(above_prs):.1f} ({np.median(above_prs)/tb_model.N*100:.2f}%)")
        print(f"    Min PR: {np.min(above_prs):.1f} ({np.min(above_prs)/tb_model.N*100:.2f}%)")
        print(f"    Max PR: {np.max(above_prs):.1f} ({np.max(above_prs)/tb_model.N*100:.2f}%)")
    
    # Density of states info
    print(f"\n[Density of States]")
    hist, bin_edges = np.histogram(tb_model.eigenvalues, bins=50)
    max_dos_idx = np.argmax(hist)
    max_dos_energy = (bin_edges[max_dos_idx] + bin_edges[max_dos_idx + 1]) / 2
    print(f"  DOS peak at energy: {max_dos_energy:.6f}")
    print(f"  DOS peak value: {hist[max_dos_idx]} states")
    
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
