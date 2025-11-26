# Quick Usage Guide - Reorganized Directory Structure

## Directory Structure

The project is now organized into two main folders:

- **`center_model/`** - Center model (dual graph) calculations and results
- **`vertex_model/`** - Vertex model calculations and results

Each folder contains:
- `data/` - Input/output data files (.npz, .pkl)
- `imgs/` - Generated plots and visualizations
- Python scripts for that model type

## Running Scripts

**⚠️ Important**: Always run scripts from the project root directory `/home/achmadjae/HU/penrose/HU/`

### Vertex Model Scripts

```bash
# Generate Penrose lattice (vertex model)
python vertex_model/penrose_tiling_fast.py --iteration 7 --plot --save-all

# Tight binding analysis
python vertex_model/penrose_tight_binding.py

# Finite size effect analysis
python vertex_model/finite_size_analysis_lanczos.py --iterations 2 3 4

# Plot LDOS
python vertex_model/finite_size_analysis_lanczos.py --plot-ldos --ldos-iteration 4
```

### Center Model Scripts

```bash
# Generate center model from vertex model
python center_model/center_penrose_tiling_fast.py --iteration 7

# Tight binding for center model
python center_model/center_penrose_tight_binding.py

# GPU-accelerated version
python center_model/center_penrose_tight_binding_cupy.py

# Lanczos method for center model
python center_model/lanczos_center_tight_binding.py
```

## Output Locations

### Vertex Model Outputs
- Data: `vertex_model/data/penrose_lattice_*.npz`
- Images: `vertex_model/imgs/*.png`
- Animations: `vertex_model/imgs/gif/*.gif`

### Center Model Outputs
- Data: `center_model/data/center_model_penrose_lattice.*`
- Images: `center_model/imgs/*.png`

## Migration Notes

All scripts have been updated to use the new paths:
- Data loading: `vertex_model/data/...` or `center_model/data/...`
- Image saving: `vertex_model/imgs/...` or `center_model/imgs/...`
- Cross-references: Center model scripts read vertex model data from `vertex_model/data/`

The old `data/` and `imgs/` directories at the root have been removed.
