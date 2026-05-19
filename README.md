# SMD Anomaly Detection with Graph Neural Networks

This project compares **LSTM** and **GNN+Correlation Graph** for anomaly detection on the SMD (Server Machine Dataset).  
It is the course project for *Data Science and Engineering*.

## Dataset
- **SMD (Server Machine Dataset)**: 28 machines, 38 metrics per time step, point-level anomaly labels.
- Download from: [https://github.com/NetManAIOps/SMD](https://github.com/NetManAIOps/SMD)
- After download, place the `ServerMachineDataset` folder under `smd/omni_anomaly/ServerMachineDataset/` (adjust path in scripts if needed).

## Requirements
- Python 3.8 – 3.10 (3.11/3.12 may work but not tested)
- PyTorch (CPU or CUDA)
- torch-geometric
- scikit-learn, pandas, numpy, matplotlib, seaborn

Install all dependencies with:
```bash
pip install -r requirements.txt
```
Note: torch-scatter and torch-sparse may require manual installation depending on your PyTorch version. If you encounter errors, please download the appropriate wheel from https://data.pyg.org/whl/.

## Repository Structure
.
├── code/
│   ├── LSTM_all_windows.py      # LSTM baseline on all window sizes
│   ├── GNN-corr.py              # GNN + Pearson correlation graph
│   ├── window_ablation.py       # GNN window ablation (5,10,15,30,45,60)
│   └── plot_corrected.py        # Generate all figures (final version)
├── results/
│   ├── lstm_all_windows_results.csv
│   └── *.png                    # All figures used in the paper
├── .gitignore
├── README.md
└── requirements.txt

## How to Run
- Prepare data as described above.

- Install dependencies:
```bash
pip install -r requirements.txt
```
- Run experiments (optional – figures can be generated directly):
```bash
python code/LSTM_all_windows.py
python code/GNN-corr.py
python code/window_ablation.py
```
- Generate all figures:
```bash
- python code/plot_corrected.py
```
All output figures and CSV results will be saved to the results/ folder.

## Results Summary
Model	Best F1 (window)	Best AUC (window)
LSTM	0.2762 (60)	0.8853 (30)
GNN+Corr Graph	0.2795 (5)	0.9088 (30)
GNN improves AUC by ~2.6% at window=30, demonstrating better discrimination ability.

F1 improvement is moderate, possibly due to noise in the static correlation graph or the 95% threshold.
## License
MIT (code). The dataset follows its original license.

## Acknowledgments
- SMD dataset: Su et al., Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network, KDD 2019.

- ChronoGraph (inspiration): Lutu et al., ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset, NeurIPS 2025.