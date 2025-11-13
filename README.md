# Penrose Lattice Tight Binding Model - Documentation

## Overview

Implementasi tight binding model untuk Penrose lattice dengan metode deflasi.

## File Structure

### 1. `penrose_lattice_stats.py`
**Fungsi**: Generate vertices dan edges dari Penrose lattice
**Output**: 
- `penrose_lattice_data.pkl` - Data dalam format pickle
- `penrose_lattice_data.npz` - Data dalam format numpy

**Data yang disimpan**:
```python
{
    'vertices': Dict[int, np.array([x, y])],  # Koordinat setiap vertex
    'edges': Dict[(i, j), arrow_type],        # Koneksi antar vertex
    'N': int,                                  # Jumlah vertices
    'E': int,                                  # Jumlah edges
    'iteration': int,                          # Iterasi deflasi
    'phi': float                               # Golden ratio
}
```

### 2. `penrose_tight_binding.py`
**Fungsi**: Diagonalisasi Hamiltonian tight binding
**Input**: `penrose_lattice_data.pkl` atau `penrose_lattice_data.npz`
**Output**:
- `penrose_energy_spectrum.png` - Energy spectrum plot
- `penrose_dos.png` - Density of States plot
- `penrose_idos.png` - Integrated Density of States plot

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

### Generate Lattice Data
```bash
python penrose_lattice_stats.py
```

### Analyze Tight Binding
```bash
python penrose_tight_binding.py
```

## Mathematical Details

### Participation Ratio
$$PR = \frac{1}{\sum_i |\psi_i|^4}$$

Measures wavefunction localization:
- PR → 1: Localized (wavefunction on few sites)
- PR → N: Extended (wavefunction spread over lattice)

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
3. **Golden ratio**: Appears in edge length ratios
4. **Aperiodicity**: No translational symmetry → no Bloch's theorem

## Future Extensions

- [ ] Add disorder (Anderson localization)
- [ ] Calculate conductivity (Kubo formula)
- [ ] Study wavefunction spatial distribution
- [ ] Analyze fractal dimension of spectrum
- [ ] Compare with periodic lattices

## References

1. Penrose tilings and quasicrystals
2. Tight binding models on aperiodic lattices
3. Electronic properties of quasicrystals
