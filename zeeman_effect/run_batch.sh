#!/bin/bash

# Batch script to run Zeeman effect calculations for different parameters

echo "========================================================================"
echo "BATCH ZEEMAN EFFECT CALCULATIONS"
echo "========================================================================"

# Iteration to use
ITERATION=4

# Array of Zeeman values
ZEEMAN_VALUES=(0.0 0.1 0.2 0.3 0.5 0.7 1.0 1.5 2.0)

echo ""
echo "Running calculations for iteration ${ITERATION}"
echo "Zeeman values: ${ZEEMAN_VALUES[@]}"
echo ""

# Run for each Zeeman value
for Z in "${ZEEMAN_VALUES[@]}"; do
    echo "----------------------------------------"
    echo "Running Z = ${Z}"
    echo "----------------------------------------"
    python penrose_zeeman.py --iteration ${ITERATION} --zeeman ${Z}
    echo ""
done

echo ""
echo "========================================================================"
echo "Running comparison plots..."
echo "========================================================================"

# Create comparison plot
python compare_zeeman.py --iteration ${ITERATION} --zeeman-values ${ZEEMAN_VALUES[@]}

echo ""
echo "========================================================================"
echo "✅ ALL CALCULATIONS COMPLETED"
echo "========================================================================"
echo ""
echo "Results saved in: results/"
echo ""
