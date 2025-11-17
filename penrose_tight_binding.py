"""
Tight Binding Model untuk Penrose Lattice
Diagonalisasi Hamiltonian dengan nearest-neighbor hopping
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import matplotlib.pyplot as plt


class PenroseTightBinding:
    """
    Tight Binding Model untuk Penrose Lattice
    
    Attributes:
        vertices (Dict[int, NDArray[np.float64]]): Dictionary vertex_id → koordinat
        edges (Dict[Tuple[int, int], int]): Dictionary (i, j) → arrow_type
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
    
    def load_from_pickle(self, filename: str = 'data/penrose_lattice_data.pkl') -> None:
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
        
        print(f"  ✓ Loaded N={self.N} vertices")
        print(f"  ✓ Loaded E={self.E} edges")
        print(f"  ✓ Iteration: {self.iteration}")
        print(f"  ✓ Golden ratio φ: {self.phi:.6f}")
    
    def load_from_numpy(self, filename: str = 'penrose_lattice_data.npz') -> None:
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
        # Karena edges adalah directed, kita perlu symmetrize
        hopping_count: int = 0
        
        for (i, j), arrow_type in self.edges.items():
            # Set hopping i <-> j (symmetrize for undirected hopping)
            H[i, j] = -self.t
            H[j, i] = -self.t
            hopping_count += 1
        
        self.hamiltonian = H
        
        print(f"  ✓ Matrix size: {self.N}×{self.N}")
        print(f"  ✓ On-site energy ε₀: {self.epsilon_0}")
        print(f"  ✓ Hopping parameter t: {self.t}")
        print(f"  ✓ Total hopping terms: {hopping_count} (from {self.E} directed edges)")
        
        # Check if Hamiltonian is Hermitian (should be for real symmetric)
        is_hermitian: bool = np.allclose(H, H.T)
        print(f"  ✓ Hermiticity check: {'PASSED' if is_hermitian else 'FAILED'}")
    
    def diagonalize(self) -> None:
        """
        Diagonalisasi Hamiltonian untuk mendapatkan eigenvalues dan eigenvectors
        
        Output:
            None (modifikasi self.eigenvalues dan self.eigenvectors in-place)
        """
        import time
        
        print(f"\n[DIAGONALIZATION] Solving eigenvalue problem...")
        print(f"  Matrix size: {self.N}×{self.N} ({self.N**2:,} elements)")
        print(f"  Starting diagonalization...")
        
        # Timer start
        start_time = time.time()
        
        # Diagonalisasi menggunakan numpy.linalg.eigh (untuk Hermitian matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(self.hamiltonian)
        
        # Timer end
        elapsed_time = time.time() - start_time
        
        # Tidak diurutkan ulang, gunakan urutan dari eigh() langsung
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        print(f"  ✓ Diagonalization completed in {elapsed_time:.2f} seconds")
        print(f"  ✓ Found {len(self.eigenvalues)} eigenvalues")
        print(f"  ✓ Energy range: [{np.min(self.eigenvalues):.6f}, {np.max(self.eigenvalues):.6f}]")
        print(f"  ✓ Energy bandwidth: {np.max(self.eigenvalues) - np.min(self.eigenvalues):.6f}")
    
    def get_density_of_states(self, bins: int = 100) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Hitung Density of States (DOS)
        
        Input:
            bins (int): Jumlah bins untuk histogram
        
        Output:
            Tuple[NDArray, NDArray]: (energies, dos)
        """
        hist, bin_edges = np.histogram(self.eigenvalues, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist
    
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
    
    def plot_energy_spectrum(self, save_fig: bool = True, filename: str = 'imgs/penrose_energy_spectrum.png') -> None:
        """
        Plot energy spectrum (eigenvalues vs index)
        
        Input:
            save_fig (bool): Simpan figure ke file
            filename (str): Nama file output
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        indices = np.arange(len(self.eigenvalues))
        ax.scatter(indices, self.eigenvalues, s=5, alpha=0.6, color='blue')
        
        ax.set_xlabel('State Index', fontsize=12)
        ax.set_ylabel('Energy (units of t)', fontsize=12)
        ax.set_title(f'Penrose Lattice Energy Spectrum (N={self.N}, Iteration={self.iteration})', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='E=0')
        ax.legend()
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            print(f"  ✓ Saved spectrum plot: {filename}")
        
        plt.close()
    
    def plot_density_of_states(self, bins: int = 100, save_fig: bool = True, 
                                filename: str = 'imgs/penrose_dos.png') -> None:
        """
        Plot Density of States (DOS)
        
        Input:
            bins (int): Jumlah bins untuk histogram
            save_fig (bool): Simpan figure ke file
            filename (str): Nama file output
        """
        energies, dos = self.get_density_of_states(bins=bins)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(energies, dos, linewidth=2, color='darkblue')
        ax.fill_between(energies, dos, alpha=0.3, color='blue')
        
        ax.set_xlabel('Energy (units of t)', fontsize=12)
        ax.set_ylabel('Density of States', fontsize=12)
        ax.set_title(f'Penrose Lattice DOS (N={self.N}, Iteration={self.iteration})', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='E=0')
        ax.legend()
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            print(f"  ✓ Saved DOS plot: {filename}")
        
        plt.close()
    
    def plot_integrated_dos(self, save_fig: bool = True, 
                            filename: str = 'imgs/penrose_idos.png') -> None:
        """
        Plot Integrated Density of States (IDOS): N₀/N vs Energy
        
        N₀(E) = number of states with energy ≤ E
        IDOS = N₀(E) / N (normalized)
        
        Input:
            save_fig (bool): Simpan figure ke file
            filename (str): Nama file output
        """
        # Sort eigenvalues (should already be sorted, but ensure it)
        sorted_energies = np.sort(self.eigenvalues)
        
        # IDOS: for each energy E, count states with energy ≤ E
        # This is simply the cumulative count normalized by N
        idos = np.arange(1, self.N + 1) / self.N
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(sorted_energies, idos, linewidth=2, color='darkgreen')
        ax.fill_between(sorted_energies, idos, alpha=0.3, color='green')
        
        ax.set_xlabel('Energy (units of t)', fontsize=12)
        ax.set_ylabel('N₀/N (Integrated DOS)', fontsize=12)
        ax.set_title(f'Penrose Lattice Integrated DOS (N={self.N}, Iteration={self.iteration})', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='E=0')
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Half-filling')
        ax.legend()
        
        # Set y-axis range
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            print(f"  ✓ Saved IDOS plot: {filename}")
        
        plt.close()
    
    def compute_bipartite_sublattices(self) -> Dict[int, str]:
        """
        Hitung pembagian sublattice A dan B menggunakan BFS (Breadth-First Search).
        Kisi Penrose adalah bipartite, sehingga bisa dibagi menjadi dua sublattice
        di mana tidak ada edge yang menghubungkan situs dalam sublattice yang sama.
        
        Output:
            Dict[int, str]: Peta site_id → 'A' atau 'B'
        """
        from collections import deque
        
        # 1. Buat adjacency list (undirected)
        adjacency = {i: set() for i in range(self.N)}
        for (i, j) in self.edges.keys():
            adjacency[i].add(j)
            adjacency[j].add(i)  # undirected
        
        # 2. Inisialisasi sublattice map
        sublattice_map = {}  # site_id -> 'A' or 'B'
        
        # 3. BFS untuk pewarnaan
        queue = deque()
        
        # Mulai dari situs 0
        start_node = 0
        sublattice_map[start_node] = 'A'
        queue.append(start_node)
        
        # BFS traversal
        while queue:
            u = queue.popleft()
            label_u = sublattice_map[u]
            opposite_label = 'B' if label_u == 'A' else 'A'
            
            # Iterasi semua tetangga u
            for v in adjacency[u]:
                if v not in sublattice_map:
                    # Belum dilihat, beri label lawan
                    sublattice_map[v] = opposite_label
                    queue.append(v)
                else:
                    # Sudah memiliki label, cek konsistensi (untuk validasi)
                    if sublattice_map[v] == label_u:
                        # Konflik! Tidak bipartite (seharusnya tidak terjadi untuk Penrose)
                        print(f"WARNING: Graph is not bipartite! Conflict at edge ({u}, {v})")
        
        # Handle disconnected components (jika ada)
        for i in range(self.N):
            if i not in sublattice_map:
                # Pulau terputus, mulai BFS baru
                sublattice_map[i] = 'A'
                queue.append(i)
                
                while queue:
                    u = queue.popleft()
                    label_u = sublattice_map[u]
                    opposite_label = 'B' if label_u == 'A' else 'A'
                    
                    for v in adjacency[u]:
                        if v not in sublattice_map:
                            sublattice_map[v] = opposite_label
                            queue.append(v)
        
        return sublattice_map
    
    def analyze_wavefunction(self, state_index: int) -> Dict[str, any]:
        """
        Analisis wavefunction untuk state tertentu
        
        Input:
            state_index (int): Index state yang ingin dianalisis
        
        Output:
            Dict: Informasi tentang wavefunction
        """
        if state_index < 0 or state_index >= self.N:
            raise ValueError(f"State index must be in range [0, {self.N-1}]")
        
        psi = self.eigenvectors[:, state_index]
        energy = self.eigenvalues[state_index]
        
        # Probability density
        prob_density = np.abs(psi)**2
        
        # Participation ratio (inverse)
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
        Plot distribusi probabilitas wavefunction (|Ψ|²) untuk state tertentu.
        
        Input:
            state_index (int): Index dari state (0 = ground state, N//2 = mid-state, dll.)
            ax (plt.Axes): Axis matplotlib untuk plotting
            size_scale (float): Faktor untuk skala ukuran titik plot
        """
        if state_index < 0 or state_index >= self.N:
            raise ValueError(f"State index harus dalam rentang [0, {self.N-1}]")
            
        # 1. Dapatkan Wavefunction (Eigenvector)
        psi: NDArray[np.float64] = self.eigenvectors[:, state_index]
        energy: float = self.eigenvalues[state_index]
        
        # 2. Dapatkan Distribusi (Probabilitas)
        prob_density: NDArray[np.float64] = np.abs(psi)**2
        
        # Hitung participation ratio untuk menampilkan info lokalisasi
        participation_ratio: float = 1.0 / np.sum(prob_density**2)
        
        # Ambil koordinat x, y dari dictionary vertices
        # Pastikan urutannya sesuai dengan indeks 0...N-1
        coords = np.array([self.vertices[i] for i in range(self.N)])
        
        # Rotasi 36 derajat searah jarum jam (clockwise = negative angle)
        theta = -18.0  # derajat
        theta_rad = np.deg2rad(theta)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        
        # Matriks rotasi 2D
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                     [sin_theta, cos_theta]])
        
        # Rotasikan semua koordinat vertex untuk plotting
        coords_rotated = coords @ rotation_matrix.T
        x_coords = coords_rotated[:, 0]
        y_coords = coords_rotated[:, 1]
        
        # 3. Plot Distribusi
        # Plot semua edges kisi sebagai latar belakang (abu-abu tipis)
        # Rotasi koordinat edges juga
        for (i, j), _ in self.edges.items():
            v_i = self.vertices[i]
            v_j = self.vertices[j]
            # Rotasi koordinat edges
            v_i_rot = rotation_matrix @ v_i
            v_j_rot = rotation_matrix @ v_j
            ax.plot([v_i_rot[0], v_j_rot[0]], [v_i_rot[1], v_j_rot[1]], 
                   color='gray', linewidth=0.3, alpha=0.3, zorder=1)

        # Hitung pembagian bipartite sublattice (A dan B)
        sublattice_map = self.compute_bipartite_sublattices()
        
        # Normalisasi prob_density agar plot terlihat bagus
        # Gunakan ukuran (s) dan warna (c) untuk merepresentasikan probabilitas
        # Threshold untuk ukuran: wf < 1e-15 sangat kecil, wf > 1e-2 besar
        size_threshold = 1e-16
        color_threshold = 1e-7
        
        # Ukuran: sangat kecil untuk < 1e-15, normal untuk 1e-15 to 1e-2, besar untuk >= 1e-2
        sizes = np.where(prob_density < size_threshold, 0.01,
                        np.where(prob_density >= color_threshold, 5.0, 1.0))
        
        # Plot scatter plot untuk semua titik dengan cmap 'hot'
        sc = ax.scatter(x_coords, y_coords, s=sizes, c=prob_density, 
                       cmap='hot', alpha=0.8, edgecolors='black', 
                       linewidth=0.3, zorder=2, vmin=0, vmax=np.max(prob_density))
        
        # Plot titik dengan prob >= threshold dengan warna berdasarkan sublattice
        # Sublattice A: merah, Sublattice B: biru
        high_prob_mask = prob_density >= color_threshold
        if np.any(high_prob_mask):
            # Pisahkan berdasarkan sublattice
            high_prob_indices = np.where(high_prob_mask)[0]
            
            # Sublattice A (merah)
            sublattice_A_mask = np.array([sublattice_map[i] == 'A' for i in high_prob_indices])
            if np.any(sublattice_A_mask):
                indices_A = high_prob_indices[sublattice_A_mask]
                ax.scatter(x_coords[indices_A], y_coords[indices_A], 
                          s=sizes[indices_A], c='red', 
                          alpha=1.0, edgecolors='black', linewidth=0.5, zorder=3)
            
            # Sublattice B (biru)
            sublattice_B_mask = np.array([sublattice_map[i] == 'B' for i in high_prob_indices])
            if np.any(sublattice_B_mask):
                indices_B = high_prob_indices[sublattice_B_mask]
                ax.scatter(x_coords[indices_B], y_coords[indices_B], 
                          s=sizes[indices_B], c='blue', 
                          alpha=1.0, edgecolors='black', linewidth=0.5, zorder=3)
        
        # Tambahkan info di title
        ax.set_title(f"State {state_index}: E={energy:.4f}\n" + 
                    f"PR={participation_ratio:.1f}/{self.N}", 
                    fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')


def print_separator(char: str = "=", length: int = 80) -> None:
    """Print separator line"""
    print(char * length)


def main() -> None:
    """
    Main program untuk tight binding model Penrose lattice
    """
    print_separator()
    print("PENROSE LATTICE - TIGHT BINDING MODEL")
    print("Nearest-Neighbor Hopping with ε₀=0, t=1")
    print_separator()
    
    # Inisialisasi model
    tb_model = PenroseTightBinding(epsilon_0=0.0, t=1.0)
    
    # Load data (gunakan pickle)
    tb_model.load_from_pickle('data/penrose_lattice_data.pkl')
    
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
    
    # Plot spectrum
    print("\n")
    print_separator()
    print("GENERATING PLOTS")
    print_separator()
    
    tb_model.plot_energy_spectrum(save_fig=True)
    tb_model.plot_density_of_states(bins=100, save_fig=True)
    tb_model.plot_integrated_dos(save_fig=True)
    
    # Find zero-energy states (E ≈ 0)
    print("\n")
    print_separator()
    print("ZERO-ENERGY STATES ANALYSIS")
    print_separator()
    
    # Find states with |E| < threshold
    energy_threshold = 1e-10
    zero_energy_indices = np.where(np.abs(tb_model.eigenvalues) < energy_threshold)[0]
    
    print(f"\nFound {len(zero_energy_indices)} states with |E| < {energy_threshold}")
    
    if len(zero_energy_indices) == 0:
        print("WARNING: No exact zero-energy states found!")
        print("Searching for states closest to E=0...")
        # Find closest states to E=0
        abs_energies = np.abs(tb_model.eigenvalues)
        sorted_indices = np.argsort(abs_energies)
        zero_energy_indices = sorted_indices[:min(3, len(sorted_indices))]
        print(f"Using {len(zero_energy_indices)} states closest to E=0")
    
    # Analyze each zero-energy state
    for i, state_idx in enumerate(zero_energy_indices):
        state_info = tb_model.analyze_wavefunction(state_idx)
        print(f"\n[Zero-Energy State {i+1}, index={state_idx}]")
        print(f"  Energy: {state_info['energy']:.10f}")
        print(f"  Participation ratio: {state_info['participation_ratio']:.2f} / {tb_model.N}")
        print(f"  Max amplitude: {state_info['max_amplitude']:.6f}")
    
    # Plot probability density for zero-energy states
    print("\n")
    print_separator()
    print("GENERATING WAVEFUNCTION PLOTS (E=0 STATES)")
    print_separator()
    
    num_states = len(zero_energy_indices)
    if num_states == 1:
        fig, axes = plt.subplots(1, 1, figsize=(7, 7))
        axes = [axes]
    elif num_states == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    else:
        # For 3 or more states, plot up to 3
        num_states = min(num_states, 3)
        fig, axes = plt.subplots(1, num_states, figsize=(7*num_states, 7))
        if num_states == 1:
            axes = [axes]
    
    fig.suptitle(f'Zero-Energy States (E≈0) - N={tb_model.N}, Iteration={tb_model.iteration}', 
                 fontsize=16, fontweight='bold')
    
    # Plot each zero-energy state
    for i in range(num_states):
        state_idx = zero_energy_indices[i]
        tb_model.plot_wavefunction(state_idx, axes[i])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    wavefunction_filename = 'imgs/penrose_wavefunctions_E0.png'
    plt.savefig(wavefunction_filename, dpi=500, bbox_inches='tight')
    print(f"  ✓ Saved wavefunction plots: {wavefunction_filename}")
    plt.close()
    
    print("\n")
    print_separator()
    print("✅ Tight binding analysis completed!")
    print("📊 Energy spectrum, DOS, IDOS, and wavefunction plots generated")
    print_separator()
    print()


if __name__ == "__main__":
    main()
