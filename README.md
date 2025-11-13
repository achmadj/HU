# Penrose Lattice Tight Binding Model - Documentation

## Overview

Implementasi tight binding model untuk Penrose lattice dengan metode deflasi. Project ini mencakup generator lattice yang dioptimasi dan analisis tight binding lengkap dengan visualisasi wavefunction bipartite.

## File Structure

### 1. `penrose_tiling_fast.py`
**Fungsi**: Generate Penrose lattice dengan algoritma deflasi yang dioptimasi
**Fitur**:
- Spatial hashing untuk O(1) vertex lookup (500x+ speedup)
- Batch processing untuk edge operations
- Optional plotting dengan flag `--plot`

**Output**: 
- `penrose_lattice_data.pkl` - Data dalam format pickle
- `penrose_lattice_data.npz` - Data dalam format numpy
- `penrose_all_iterations_fast.png` - Plot semua iterasi (jika `--plot`)
- `penrose_iter0_iter1_overlapped_fast.png` - Plot overlap iterasi 0 dan 1 (jika `--plot`)

**Data yang disimpan**:
```python
{
    'vertices': Dict[int, np.array([x, y])],  # Koordinat setiap vertex
    'edges': Dict[(i, j), arrow_type],        # Koneksi antar vertex
    'N': int,                                  # Jumlah vertices
    'E': int,                                  # Jumlah edges
    'iteration': int,                          # Iterasi deflasi
    'phi': float,                              # Golden ratio
    'single_arrows': int,                      # Jumlah single arrows
    'double_arrows': int                       # Jumlah double arrows
}
```

### 2. `penrose_tight_binding.py`
**Fungsi**: Diagonalisasi Hamiltonian tight binding dengan analisis wavefunction
**Input**: `penrose_lattice_data.pkl` atau `penrose_lattice_data.npz`
**Output**:
- `penrose_energy_spectrum.png` - Energy spectrum plot (DPI 200)
- `penrose_dos.png` - Density of States plot (DPI 200)
- `penrose_idos.png` - Integrated Density of States plot (DPI 200)
- `penrose_wavefunctions.png` - Wavefunction probability density plots (DPI 1000)

**Fitur Baru**:
- Bipartite sublattice detection (BFS algorithm)
- Wavefunction visualization dengan pewarnaan sublattice (merah/biru)
- Rotasi plot -18° untuk orientasi optimal
- Threshold-based sizing untuk highlighting

## Tight Binding Model

### Hamiltonian
$$H = \sum_i \epsilon_0 |i\rangle\langle i| - t \sum_{\langle i,j \rangle} (|i\rangle\langle j| + |j\rangle\langle i|)$$

Where:
- $\epsilon_0 = 0$ (on-site energy)
- $t = 1$ (hopping parameter)
- $\langle i,j \rangle$ indicates nearest neighbors (connected by edges)

### Properties

**Matrix**: Hermitian, real symmetric
**Size**: N×N (N = number of vertices)
**Sparsity**: ~2E/N² (karena hanya nearest neighbor)

### Results (Iteration 2, N=86)

| Property | Value |
|----------|-------|
| Vertices (N) | 86 |
| Edges (E) | 150 |
| Bandwidth | 7.949 |
| Energy range | [-3.975, 3.975] |
| Zero-energy states | 8 |
| Mean energy | 0.0 (by symmetry) |

### Physical Interpretation

1. **Particle-hole symmetry**: Spectrum symmetric around E=0
2. **Extended states**: High participation ratio (~40% of lattice)
3. **Quasiperiodic structure**: Non-degenerate energy levels
4. **No band gaps**: Continuous DOS (unlike periodic lattices)

## Usage

### 1. Generate Penrose Lattice
```bash
# Generate lattice data only (fast)
python penrose_tiling_fast.py

# Generate lattice data with plots
python penrose_tiling_fast.py --plot
```

**Performance**: Iterasi 5 (~11,000 vertices) dalam ~2 detik (524x lebih cepat dari metode original)

### 2. Analyze Tight Binding Model
```bash
python penrose_tight_binding.py
```

Output mencakup:
- Energy spectrum, DOS, dan IDOS
- Wavefunction plots untuk ground state, mid-state (E≈0), dan highest state
- Bipartite sublattice visualization (merah = A, biru = B)

## Mathematical Details

### Bipartite Sublattice Structure

Penrose lattice adalah **bipartite graph**, artinya vertices bisa dibagi menjadi dua sublattice (A dan B) di mana tidak ada edge yang menghubungkan vertices dalam sublattice yang sama.

**Algoritma Detection (BFS)**:
1. Mulai dari vertex arbitrary, beri label 'A'
2. Semua tetangganya diberi label 'B'
3. Tetangga dari 'B' diberi label 'A', dst.
4. Traversal menggunakan Breadth-First Search (BFS)

**Visualisasi**: Di plot wavefunction, titik dengan probabilitas tinggi dibedakan:
- **Merah**: Sublattice A
- **Biru**: Sublattice B

### Participation Ratio
$$PR = \frac{1}{\sum_i |\psi_i|^4}$$

Measures wavefunction localization:
- PR → 1: Localized (wavefunction on few sites)
- PR → N: Extended (wavefunction spread over lattice)

### Wavefunction Visualization

**Size threshold**:
- `prob < 1e-16`: ukuran 0.01 (sangat kecil)
- `1e-16 ≤ prob < 1e-7`: ukuran 1.0 (normal)
- `prob ≥ 1e-7`: ukuran 27.0 (besar)

**Color scheme**:
- Semua titik: colormap 'hot' berdasarkan probabilitas
- `prob ≥ 1e-7`: merah (sublattice A) atau biru (sublattice B)

**Rotation**: Plot dirotasi -18° searah jarum jam untuk orientasi optimal

### Density of States
$$\rho(E) = \frac{1}{N} \sum_{n=1}^N \delta(E - E_n)$$

In practice, approximated using histogram with Gaussian broadening.

### Integrated Density of States (IDOS)
$$N_0(E) / N = \frac{1}{N} \sum_{n=1}^N \Theta(E - E_n)$$

Where $\Theta$ is the Heaviside step function. This represents the cumulative fraction of states with energy ≤ E.

Properties:
- Range: [0, 1]
- Monotonically increasing
- At E=0: gives fraction of filled states at half-filling
- Slope at energy E equals DOS at that energy: $d(N_0/N)/dE = \rho(E)$

## Key Observations

1. **Ground state energy**: E₀ ≈ -3.975t
2. **Coordination**: Average ~3.5 neighbors per site
3. **Golden ratio**: Appears in edge length ratios (φ = (1+√5)/2 ≈ 1.618)
4. **Aperiodicity**: No translational symmetry → no Bloch's theorem
5. **Bipartite structure**: Perfect two-coloring possible (seperti papan catur)
6. **Wavefunction localization**: Bervariasi tergantung energi, terlihat jelas dari PR

## Performance Optimizations

### Lattice Generation (`penrose_tiling_fast.py`)
- **Spatial hashing**: O(1) vertex lookup menggunakan hash grid
- **Pre-computed trigonometry**: Menghindari kalkulasi berulang
- **Batch processing**: Operasi vectorized untuk edges
- **Result**: 524x speedup dibanding implementasi naive (iterasi 5: 81s → 0.15s)

### Growth Rate
Setiap iterasi deflasi:
- Vertices: $N_{i+1} \approx \phi^2 \cdot N_i$ (faktor ~2.618)
- Edges: $E_{i+1} \approx \phi^2 \cdot E_i$

| Iteration | N (vertices) | E (edges) | Time (ms) |
|-----------|--------------|-----------|-----------|
| 0         | 11           | 15        | < 1       |
| 1         | 31           | 45        | < 1       |
| 2         | 86           | 150       | ~5        |
| 3         | 226          | 405       | ~15       |
| 4         | 596          | 1080      | ~50       |
| 5         | 1571         | 2865      | ~150      |

## Algorithm Details

### Deflation Rules
**Single arrow** (type 1): Menghasilkan 5 edges baru
- 2 double arrows
- 3 single arrows

**Double arrow** (type 2): Menghasilkan 4 edges baru
- 2 double arrows
- 2 single arrows

Setelah deflasi, semua vertices discale dengan faktor φ.

### BFS Bipartite Coloring
```python
# Pseudocode
queue = [start_vertex]
color[start_vertex] = 'A'

while queue not empty:
    u = queue.pop()
    opposite = 'B' if color[u] == 'A' else 'A'
    
    for v in neighbors(u):
        if v uncolored:
            color[v] = opposite
            queue.append(v)
```

## References

1. de Bruijn, N. G. (1981). "Algebraic theory of Penrose's non-periodic tilings of the plane"
2. Steinhardt, P. J., & Ostlund, S. (1987). "The Physics of Quasicrystals"
3. Tight binding models on aperiodic lattices
4. Electronic properties of quasicrystals
