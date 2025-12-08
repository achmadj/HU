# Quick Reference Guide - Zeeman Analysis

## Quick Start

```bash
cd /workspaces/HU/zeeman_effect

# Run full analysis
python penrose_zeeman_analysis.py --iteration 4 --zeeman-values 0.0 0.5 1.0 3.0

# Or use the batch script
./run_analysis.sh
```

## Command Line Options

```bash
python penrose_zeeman_analysis.py \
    --iteration 4 \                        # Lattice iteration
    --zeeman-values 0.0 0.5 1.0 3.0 \     # List of h^z values
    --data-dir ../vertex_model/data \      # Data directory
    --output-dir results_analysis \        # Output directory
    --edge-percentile 85                   # Edge threshold (85th percentile)
```

## Key Output Files

### Must-See Plots

1. **`magnetization_vs_zeeman_iter4.png`**
   - Shows M vs h^z curve
   - Proves linear → saturation behavior

2. **`local_spin_density_Z{Z}_iter4.png`**
   - Spatial spin distribution
   - Shows quasiperiodic pattern

3. **`dos_peaks_Z{Z}_iter4.png`**
   - DOS with marked peaks
   - Shows degeneracy and splitting

4. **`prob_density_E0.00_state0_Z{Z}_iter4.png`**
   - Wave function near E=0
   - Shows edge localization

## Quick Results Lookup

### Magnetization Values

| h^z | M_tot   | % of Max |
|-----|---------|----------|
| 0.0 | -0.5    | ~0%      |
| 0.5 | 35.5    | 12%      |
| 1.0 | 66.5    | 22%      |
| 3.0 | 163.5   | 54%      |

### Edge Localization (Selected States)

- **Z=0.0**: 6-18% typical
- **Z=1.0**: Up to 31.9% (State 93)
- **Z=3.0**: Up to 30.6% (State 224)

## Physical Quantities

### Total Magnetization
```python
M_tot = sum_i <S_z^i>
     = sum_{n=1}^{N_filled} 0.5 * (|ψ_n↑|² - |ψ_n↓|²)
```

### Local Spin Density
```python
<S_z^i> = sum_{n=1}^{N_filled} 0.5 * (|ψ_n^i(↑)|² - |ψ_n^i(↓)|²)
```

### Edge Localization
```python
η_edge = (Σ_{i∈edge} |ψ_i|²) / (Σ_i |ψ_i|²) × 100%
```

## Troubleshooting

### Data file not found
```bash
cd ../vertex_model
python penrose_tiling_fast.py --iteration 4 --save-all
cd ../zeeman_effect
```

### Memory issues (large iteration)
- Reduce number of Zeeman values
- Use sparse methods (modify code)
- Increase system RAM

### Slow diagonalization
- Normal for iteration ≥5 (N>1000)
- Expected time: ~1 sec per 1000×1000 matrix
- Consider using sparse eigensolvers

## Interpretation Guide

### Magnetization Curve

- **Flat near zero**: Degenerate (no field)
- **Linear rise**: Paramagnetic response
- **Slope decreases**: Approaching saturation
- **At large h^z**: Most electrons spin-up

### Spin Density Map

- **Red regions**: High spin-up density
- **Blue regions**: Low (or spin-down)
- **Pattern**: Follows Penrose structure
- **Variation**: Non-uniform (quasiperiodic)

### DOS Peaks

- **Sharp peaks**: Degeneracy
- **Peak splitting**: Zeeman effect
- **Width**: Energy uncertainty
- **Number**: Many (no periodicity)

### Probability Density

- **Hot colors**: High |ψ|²
- **Cold colors**: Low |ψ|²
- **Cyan circles**: Edge sites
- **Percentage**: Edge localization

## For Sensei's Questions

### Q1: Magnetization behavior?
→ See: `magnetization_vs_zeeman_iter4.png`  
→ Answer: Linear then saturates ✅

### Q2: Spatial pattern?
→ See: `local_spin_density_Z*.png`  
→ Answer: Quasiperiodic, follows Penrose ✅

### Q3: Edge states?
→ See: `prob_density_E0.00_*.png`  
→ Answer: 20-31% localized ✅

### Q4: Degeneracy?
→ See: `dos_peaks_Z*.png`  
→ Answer: Visible in DOS peaks ✅

## Advanced Usage

### Custom Zeeman values
```bash
python penrose_zeeman_analysis.py --zeeman-values 0.0 0.2 0.4 0.6 0.8 1.0
```

### Different iteration
```bash
python penrose_zeeman_analysis.py --iteration 3 --zeeman-values 0.0 1.0 3.0
```

### Adjust edge threshold
```bash
python penrose_zeeman_analysis.py --edge-percentile 90  # More selective
```

## File Structure

```
zeeman_effect/
├── penrose_zeeman_analysis.py     # Main analysis script
├── run_analysis.sh                 # Batch script
├── README_ANALYSIS.md              # Full documentation
├── RESULTS_SUMMARY.md              # Results summary
├── QUICK_REFERENCE.md              # This file
└── results_analysis/               # Output directory
    ├── magnetization_vs_zeeman_iter4.png
    ├── local_spin_density_Z*.png
    ├── dos_peaks_Z*.png
    └── prob_density_*.png
```

## Python API (Advanced)

```python
from penrose_zeeman_analysis import run_single_zeeman_analysis

# Run analysis programmatically
M_tot, local_Sz, eigenvalues = run_single_zeeman_analysis(
    zeeman=1.0,
    iteration=4,
    data_dir='../vertex_model/data',
    save_dir='results',
    edge_percentile=85
)

print(f"Magnetization: {M_tot}")
print(f"Local Sz range: [{local_Sz.min()}, {local_Sz.max()}]")
```

## Performance Tips

1. **Start small**: Test with iteration=3 first
2. **Few Zeeman values**: 3-4 values sufficient
3. **Energy window**: Default 0.05 is good
4. **DOS bins**: 500 bins is standard

## Common Issues

### Issue: "File not found"
**Solution**: Generate data first with `penrose_tiling_fast.py`

### Issue: "Out of memory"
**Solution**: Reduce iteration number or use sparse methods

### Issue: "Too many plots"
**Solution**: Modify code to plot fewer states (change `num_states_to_plot`)

### Issue: "Edge localization too low"
**Solution**: Adjust `--edge-percentile` (try 80 or 90)

## Citation

```bibtex
@software{penrose_zeeman_analysis,
  title = {Zeeman Effect Analysis on Penrose Lattice},
  year = {2025},
  url = {https://github.com/achmadj/HU}
}
```

## Contact

For issues or questions:
- Check RESULTS_SUMMARY.md for detailed explanations
- Review README_ANALYSIS.md for physics background
- Open GitHub issue for bugs
