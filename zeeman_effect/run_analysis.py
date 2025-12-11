#!/usr/bin/env python3
"""Quick script to run Zeeman analysis"""
import subprocess
import sys

cmd = [sys.executable, 'penrose_zeeman_analysis.py', '-z', '0.5', '-i', '6']
print(f"Running: {' '.join(cmd)}")
subprocess.run(cmd)
