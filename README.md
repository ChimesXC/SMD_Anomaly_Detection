# SMD Anomaly Detection with Graph Neural Networks

This project compares LSTM and GNN+Correlation Graph for anomaly detection on the SMD (Server Machine Dataset).  
Course project for [数据科学与工程].

## Dataset
- **SMD (Server Machine Dataset)**: 28 machines, 38 metrics, point-level anomaly labels.
- Download from: [https://github.com/NetManAIOps/SMD](https://github.com/NetManAIOps/SMD)
- After download, place the `ServerMachineDataset` folder under `smd/omni_anomaly/ServerMachineDataset/` (adjust path in scripts if needed).

## Requirements
- Python 3.8–3.10
- PyTorch (CPU or CUDA)
- torch-geometric
- scikit-learn, pandas, numpy, matplotlib, seaborn

Install all dependencies with:
```bash
pip install -r requirements.txt
