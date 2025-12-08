#!/bin/bash

# Comprehensive Zeeman Effect Analysis Script
# Runs analysis for multiple Zeeman field values

echo "=========================================="
echo "Penrose Lattice Zeeman Effect Analysis"
echo "=========================================="
echo ""

# Set parameters
ITERATION=4
DATA_DIR="../vertex_model/data"
OUTPUT_DIR="results_analysis"

# Check if data file exists
DATA_FILE="${DATA_DIR}/penrose_lattice_iter${ITERATION}.npz"

if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    echo ""
    echo "Please generate the data first:"
    echo "  cd ../vertex_model"
    echo "  python penrose_tiling_fast.py --iteration $ITERATION --save-all"
    exit 1
fi

echo "Data file found: $DATA_FILE"
echo ""

# Run comprehensive analysis
echo "Running comprehensive analysis..."
echo "  Zeeman values: 0.0, 0.5, 1.0, 3.0"
echo "  Output directory: $OUTPUT_DIR"
echo ""

python penrose_zeeman_analysis.py \
    --iteration $ITERATION \
    --zeeman-values 0.0 0.5 1.0 3.0 \
    --data-dir $DATA_DIR \
    --output-dir $OUTPUT_DIR \
    --edge-percentile 85

echo ""
echo "=========================================="
echo "Analysis completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Generated plots:"
echo "  1. Magnetization vs Zeeman field (M vs h^z)"
echo "  2. Local spin density maps for each Zeeman value"
echo "  3. DOS with peak markers"
echo "  4. Probability density maps near degenerate energies"
echo ""
