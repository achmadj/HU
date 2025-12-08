# Comprehensive Zeeman Effect Analysis on Penrose Lattice

## Overview

This directory contains comprehensive analysis tools for studying the Zeeman effect on Penrose lattice tight-binding models. The analysis addresses key physical questions about magnetization, spin density distribution, and edge state localization in quasiperiodic systems.

## Physical Model

### Hamiltonian

The system is described by a tight-binding Hamiltonian with Zeeman splitting:

$$H = H_{\text{hop}} + H_Z$$

Where:
- **Hopping term**: $H_{\text{hop}} = -t \sum_{\langle i,j \rangle} (c_i^\dagger c_j + \text{h.c.})$
- **Zeeman term**: $H_Z = -h^z \sum_i S_z^i c_i^\dagger c_i$

With $S_z = +1/2$ for spin-up, $S_z = -1/2$ for spin-down.

### Basis Convention

The Hamiltonian is constructed in a $(2N) \times (2N)$ basis:
$$[\text{up}_0, \text{up}_1, \ldots, \text{up}_{N-1}, \text{down}_0, \text{down}_1, \ldots, \text{down}_{N-1}]$$

This gives:
- **Spin-up energies**: $E_{\text{up}} = E_0 - h^z/2$
- **Spin-down energies**: $E_{\text{down}} = E_0 + h^z/2$

## Key Physical Quantities

### 1. Total Magnetization

$$M_{\text{tot}} = \langle S_z^{\text{tot}} \rangle = \sum_i \langle S_z^i \rangle$$

For ground state (T=0) at half-filling:
$$M_{\text{tot}} = \sum_{n=1}^{N} \left[ \frac{1}{2}(|\psi_n^{\uparrow}|^2 - |\psi_n^{\downarrow}|^2) \right]$$

**Expected behavior**:
- Linear increase at small $h^z$
- Saturation at large $h^z$ (fully polarized)

### 2. Local Spin Density

$$\langle S_z^i \rangle = \sum_{n=1}^{N_{\text{filled}}} \frac{1}{2}(|\psi_n^i(\uparrow)|^2 - |\psi_n^i(\downarrow)|^2)$$

This shows spatial distribution of magnetization across the Penrose lattice.

**Key observations**:
- Non-uniform distribution (breaks translational symmetry)
- Follows Penrose tiling structure
- Enhanced at certain coordination sites

### 3. Edge State Localization

Edge states are identified by:
1. Finding states near $E = 0$ (or other degenerate energies)
2. Calculating probability density: $\rho_i = |\psi_i|^2$
3. Measuring edge localization percentage

**Edge localization metric**:
$$\eta_{\text{edge}} = \frac{\sum_{i \in \text{edge}} |\psi_i|^2}{\sum_{i} |\psi_i|^2} \times 100\%$$

## Analysis Scripts

### `penrose_zeeman_analysis.py`

Comprehensive analysis script that computes:

1. **Magnetization vs Zeeman field**: $M_{\text{tot}}(h^z)$ curve
2. **Local spin density maps**: Spatial distribution of $\langle S_z^i \rangle$
3. **DOS analysis**: Identifies degenerate energy peaks
4. **Probability density maps**: $|\psi_i|^2$ for states near degeneracies
5. **Edge state analysis**: Quantifies edge localization

#### Usage

```bash
python penrose_zeeman_analysis.py --iteration 4 --zeeman-values 0.0 0.5 1.0 3.0
```

**Arguments**:
- `--iteration`: Penrose lattice iteration number (default: 4)
- `--zeeman-values`: List of Zeeman field values (default: 0.0 0.5 1.0 3.0)
- `--data-dir`: Directory with lattice data (default: ../vertex_model/data)
- `--output-dir`: Output directory (default: results_analysis)
- `--edge-percentile`: Percentile for edge identification (default: 85)

#### Output Files

For each Zeeman value $h^z$:
- `dos_peaks_Z{h^z}_iter{N}.png` - DOS with marked peaks
- `local_spin_density_Z{h^z}_iter{N}.png` - Spin density map
- `prob_density_E{E}_state{n}_Z{h^z}_iter{N}.png` - Wave function plots

Summary:
- `magnetization_vs_zeeman_iter{N}.png` - $M_{\text{tot}}$ vs $h^z$ curve

### `run_analysis.sh`

Batch script for complete analysis:

```bash
./run_analysis.sh
```

This runs the full analysis pipeline for Zeeman values: 0.0, 0.5, 1.0, 3.0.

## Physical Interpretation

### Degeneracy and Zeeman Splitting

Without magnetic field ($h^z = 0$):
- Spin-up and spin-down states are degenerate
- DOS shows peaks at degenerate energies
- No net magnetization

With magnetic field ($h^z > 0$):
- Degeneracy is lifted: $\Delta E = h^z$
- DOS peaks split into two peaks separated by $h^z$
- Net magnetization favors spin-up states

### Magnetization Curve

The $M_{\text{tot}}(h^z)$ curve shows:

1. **Linear regime** (small $h^z$): 
   - $M \propto h^z$
   - Susceptibility: $\chi = dM/dh^z$

2. **Saturation regime** (large $h^z$):
   - $M \to M_{\text{max}} = N/2$
   - All electrons polarized spin-up

### Spatial Patterns

The local spin density map reveals:
- **Quasiperiodic modulation**: Following golden ratio structure
- **Site-dependent magnetization**: Higher coordination → larger $\langle S_z \rangle$
- **No translational symmetry**: Unlike periodic lattices

### Edge States

Edge states in Penrose lattices:
- Localized at boundaries (high radial distance)
- Contribute significantly to total magnetization
- Sensitive to boundary conditions
- Different from topological edge states (no band gap here)

## Answering Key Physics Questions

### Question 1: Linear to Saturation Behavior

**Plot**: `magnetization_vs_zeeman_iter{N}.png`

Shows $M_{\text{tot}}$ increases linearly then saturates, proving paramagnetic response with finite magnetic moment.

### Question 2: Non-uniform Magnetization

**Plot**: `local_spin_density_Z{h^z}_iter{N}.png`

Demonstrates spatially varying $\langle S_z^i \rangle$ following Penrose structure, proving lack of translational symmetry.

### Question 3: Edge State Contribution

**Plot**: `prob_density_E0.00_state{n}_Z{h^z}_iter{N}.png`

Shows states near $E=0$ are localized at edges with quantified edge localization percentage.

### Question 4: Degeneracy Splitting

**Plot**: `dos_peaks_Z{h^z}_iter{N}.png`

Shows DOS peak splitting increases with $h^z$, confirming Zeeman effect lifts degeneracy.

## Expected Results

### Iteration 4 (N ≈ 200-500)

| $h^z$ | Expected $M_{\text{tot}}$ | DOS Peak Splitting |
|-------|--------------------------|-------------------|
| 0.0   | ≈ 0                      | None (degenerate) |
| 0.5   | 5-20% of $M_{\text{max}}$ | 0.5              |
| 1.0   | 20-40% of $M_{\text{max}}$| 1.0              |
| 3.0   | 60-90% of $M_{\text{max}}$| 3.0 (saturating) |

### Edge Localization

- Edge states near $E=0$: 40-70% localization at boundary
- Bulk states: <30% at boundary
- Depends on iteration (larger → more edge states)

## Computational Notes

### Performance

For iteration 4 (N ~ 500):
- Hamiltonian size: 1000 × 1000
- Diagonalization: ~1-5 seconds (dense method)
- Memory: ~8 MB per Hamiltonian

For larger iterations:
- Use sparse eigensolvers (`scipy.sparse.linalg.eigsh`)
- Focus on states near Fermi energy
- Consider iterative methods (Lanczos)

### Numerical Stability

- Energy window for peak finding: ±0.05
- DOS binning: 500 bins (adjustable)
- Edge percentile: 85th percentile (adjustable)
- Degeneracy threshold: 90th percentile of DOS

## Verification Checklist

✅ **Magnetization curve**: Linear → saturating behavior  
✅ **Spin density map**: Quasiperiodic pattern visible  
✅ **DOS splitting**: Peaks shift with $h^z$  
✅ **Edge states**: High localization near boundaries  
✅ **Sum rules**: $\sum_i \langle S_z^i \rangle = M_{\text{tot}}$

## References

1. Penrose tilings and quasicrystals
2. Tight-binding models on aperiodic lattices
3. Zeeman effect in quantum systems
4. Edge states in finite systems
5. Magnetization in tight-binding models

## Citation

If you use this code, please cite:
```
Penrose Lattice Zeeman Effect Analysis
GitHub: [repository URL]
```

## Contact

For questions or issues, please open an issue on GitHub.
