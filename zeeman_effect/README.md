# Zeeman Effect in Penrose Lattice

This directory contains code for calculating the electronic properties of a Penrose lattice with an applied magnetic field (Zeeman effect).

## Overview

The tight-binding Hamiltonian includes:

1. **Hopping term**: H_hop = -t * Σ_{<i,j>} (c_i† c_j + h.c.)
2. **Zeeman splitting**: H_Z = -g μ_B B Σ_i s_z c_i† c_i

where:
- t is the hopping parameter (default: -1.0)
- s_z = +1/2 for spin-up, -1/2 for spin-down
- The Zeeman parameter represents the energy splitting: ΔE = g μ_B B

## Usage

### Basic usage:

```bash
python penrose_zeeman.py --iteration 4 --zeeman 0.2
```

### Command-line options:

- `--iteration, -i`: Penrose lattice iteration (default: 4)
- `--zeeman, -z`: Zeeman splitting parameter (default: 0.0)
- `--hopping, -t`: Hopping parameter (default: -1.0)
- `--data-dir`: Directory containing Penrose lattice data (default: ../vertex_model/data)
- `--output-dir`: Output directory for plots (default: results)

### Examples:

1. **No magnetic field** (Zeeman = 0):
   ```bash
   python penrose_zeeman.py --iteration 4 --zeeman 0.0
   ```

2. **Weak magnetic field** (Zeeman = 0.2):
   ```bash
   python penrose_zeeman.py --iteration 4 --zeeman 0.2
   ```

3. **Strong magnetic field** (Zeeman = 1.0):
   ```bash
   python penrose_zeeman.py --iteration 4 --zeeman 1.0
   ```

4. **Different iteration**:
   ```bash
   python penrose_zeeman.py --iteration 5 --zeeman 0.5
   ```

## Output

The script generates:

1. **Energy Spectrum**: Plot showing all eigenvalues
2. **Density of States (DOS)**: Energy distribution of states
3. **Integrated Density of States (IDOS)**: Cumulative state count

Results are saved to `results/zeeman_Z<value>_iter<n>.png`

## Physics

### Zeeman Effect

When a magnetic field B is applied perpendicular to the 2D lattice:
- Spin-up states shift down by -Z/2
- Spin-down states shift up by +Z/2
- Total splitting between spin states: ΔE = Z

### Spin Sectors

The Hamiltonian is block-diagonal in spin space (no spin-flip terms):
- Spin-up sector: H↑ = H_0 - Z/2 I
- Spin-down sector: H↓ = H_0 + Z/2 I

where H_0 is the tight-binding Hamiltonian without Zeeman effect.

## Data Requirements

Before running the Zeeman calculations, generate Penrose lattice data:

```bash
cd ../vertex_model
python penrose_tiling_fast.py --iteration 4 --save-all
```

This creates `data/penrose_lattice_iter4.npz` containing vertex coordinates and edge lists.

## Comparison Tool

To compare multiple Zeeman values at once:

```bash
python compare_zeeman.py --iteration 4 --zeeman-values 0.0 0.2 0.5 1.0
```

This generates:
- Individual DOS plots for each Zeeman value (2×2 grid)
- Overlay plot showing all DOS curves together
- Summary statistics for each case

### Batch Processing

For systematic parameter sweep, use the batch script:

```bash
./run_batch.sh
```

This automatically runs calculations for Zeeman values: 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0

## File Structure

```
zeeman_effect/
├── penrose_zeeman.py          # Main calculation script
├── compare_zeeman.py           # Comparison tool
├── run_batch.sh                # Batch processing script
├── README.md                   # This file
└── results/                    # Output directory
    ├── zeeman_Z0.000_iter4.png
    ├── zeeman_Z0.200_iter4.png
    ├── zeeman_comparison_iter4.png
    └── zeeman_overlay_iter4.png
```

## Dependencies

- numpy
- scipy
- matplotlib

## Notes

- For large systems (N > 2500), the dense diagonalization may be slow
- The code automatically adjusts for system size
- Spin-up and spin-down sectors are computed together (dimension: 2N × 2N)
- Zero energy states disappear when Zeeman splitting is applied (breaks degeneracy)
- The energy spectrum shifts symmetrically: E_up → E - Z/2, E_down → E + Z/2
