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
    
    def load_from_pickle(self, filename: str = 'penrose_lattice_data.pkl') -> None:
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
        print(f"\n[DIAGONALIZATION] Solving eigenvalue problem...")
        
        # Diagonalisasi menggunakan numpy.linalg.eigh (untuk Hermitian matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(self.hamiltonian)
        
        # Sort by eigenvalue (ascending)
        sort_indices = np.argsort(eigenvalues)
        self.eigenvalues = eigenvalues[sort_indices]
        self.eigenvectors = eigenvectors[:, sort_indices]
        
        print(f"  ✓ Found {len(self.eigenvalues)} eigenvalues")
        print(f"  ✓ Energy range: [{self.eigenvalues[0]:.6f}, {self.eigenvalues[-1]:.6f}]")
        print(f"  ✓ Energy bandwidth: {self.eigenvalues[-1] - self.eigenvalues[0]:.6f}")
    
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
            'zero_energy_count': int(np.sum(np.abs(self.eigenvalues) < 1e-10))
        }
    
    def plot_energy_spectrum(self, save_fig: bool = True, filename: str = 'penrose_energy_spectrum.png') -> None:
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
                                filename: str = 'penrose_dos.png') -> None:
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
                            filename: str = 'penrose_idos.png') -> None:
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
    tb_model.load_from_pickle('penrose_lattice_data.pkl')
    
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
    
    # Analisis beberapa states menarik
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
    print(f"  Localization: {'Localized' if ground_state['participation_ratio'] < tb_model.N/10 else 'Extended'}")
    
    # Middle state
    mid_index = tb_model.N // 2
    print(f"\n[Middle State (index={mid_index})]")
    mid_state = tb_model.analyze_wavefunction(mid_index)
    print(f"  Energy: {mid_state['energy']:.6f}")
    print(f"  Participation ratio: {mid_state['participation_ratio']:.2f} / {tb_model.N}")
    print(f"  Max amplitude: {mid_state['max_amplitude']:.6f}")
    print(f"  Localization: {'Localized' if mid_state['participation_ratio'] < tb_model.N/10 else 'Extended'}")
    
    # Highest state
    print(f"\n[Highest State (index={tb_model.N-1})]")
    highest_state = tb_model.analyze_wavefunction(tb_model.N - 1)
    print(f"  Energy: {highest_state['energy']:.6f}")
    print(f"  Participation ratio: {highest_state['participation_ratio']:.2f} / {tb_model.N}")
    print(f"  Max amplitude: {highest_state['max_amplitude']:.6f}")
    print(f"  Localization: {'Localized' if highest_state['participation_ratio'] < tb_model.N/10 else 'Extended'}")
    
    print("\n")
    print_separator()
    print("✅ Tight binding analysis completed!")
    print("📊 Energy spectrum, DOS, and IDOS plots generated")
    print_separator()
    print()


if __name__ == "__main__":
    main()
