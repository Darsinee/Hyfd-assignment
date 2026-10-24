# HyFD 

This repository contains my implementation and evaluation of the HyFD algorithm for discovering functional dependencies.

## Files

- `data/reproduce_hyfd.py` — Runs HyFD and the baselines and cross-checks the result with brute force
- `Implementation_Hyfd.ipynb` — notebook that runs the implementation and compares its results with the paper.
- `results/` — CSV files produced by the experiments.

## Environment

The code has been written and run in Google Colab rather than VS Code because the Desbordante package used in this project does not provide pre-built Windows installers. The script requires Python, pandas, NumPy, and desbordante version 2.5.0, and runs on Linux, macOS, or Google Colab.

## Running the notebook

Open `Implementation_Hyfd.ipynb` in Google Colab and run the cells in order. Upload the reproduce_hyfd.py file when prompted. The notebook produces CSV result files in `results/`.
