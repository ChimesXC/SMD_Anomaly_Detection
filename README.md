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
Repository Structure
.
├── code/
│   ├── LSTM_all_windows.py
│   ├── GNN_corr.py
│   ├── window_ablation.py
│   └── plot_corrected.py
├── results/
│   ├── lstm_all_windows_results.csv
│   └── *.png (all figures)
├── .gitignore
├── README.md
└── requirements.txt
How to Run
Prepare data as described above.

Install dependencies: pip install -r requirements.txt

Run experiments:

bash
python code/LSTM_all_windows.py
python code/GNN_corr.py
python code/window_ablation.py
Generate all figures:

bash
python code/plot_corrected.py
Results Summary
LSTM best F1: 0.2762 (window=60), best AUC: 0.8853 (window=30)

GNN+Corr best F1: 0.2795 (window=5), best AUC: 0.9088 (window=30)

GNN improves AUC by ~2.6% at window=30.

License
MIT (code). Dataset follows its original license.

Acknowledgments
SMD dataset: Su et al., KDD 2019

ChronoGraph (inspiration): Lutu et al., NeurIPS 2025

text

保存文件。

---

## 📦 2. 创建 `requirements.txt`

在相同目录下新建 `requirements.txt`，内容如下（每一行是一个依赖包及版本）：

```txt
torch>=2.0.0
torch-geometric>=2.3.0
torch-scatter
torch-sparse
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
