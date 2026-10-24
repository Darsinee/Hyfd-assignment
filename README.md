# HyFD 

This repository contains my implementation and evaluation of the HyFD algorithm for discovering functional dependencies.

## Files

- `data/reproduce_hyfd.py` — Python script that reproduces the algorithm.
- `Implementation_Hyfd.ipynb` — notebook that runs the implementation and compares its results with the paper.
- `results/` — CSV files produced by the experiments.

## Environment

I wrote and ran the code in Google Colab rather than VS Code because the Desbordante package used in this project does not provide pre-built Windows installers. Colab provides a Linux environment that supports the available package.

## Running the notebook

Open `Implementation_Hyfd.ipynb` in Google Colab and run the cells in order. Upload the reproduce_hyfd.py file when prompted. The notebook produces CSV result files in `results/`.
