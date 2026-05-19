import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ======================== 正确数据 ========================
windows = [5, 10, 15, 30, 45, 60]
lstm_f1 = [0.2725, 0.2681, 0.2678, 0.2735, 0.2718, 0.2762]
lstm_auc = [0.8832, 0.8734, 0.8820, 0.8853, 0.8790, 0.8801]
gnn_f1 = [0.2795, 0.2692, 0.2609, 0.2607, 0.2629, 0.2648]
gnn_auc = [0.8624, 0.8850, 0.8742, 0.9088, 0.8913, 0.8855]

# ========== 图1：F1 对比折线图 ==========
plt.figure(figsize=(8,5))
plt.plot(windows, lstm_f1, 'b-o', label='LSTM', linewidth=2, markersize=8)
plt.plot(windows, gnn_f1, 'r-s', label='GNN+Corr', linewidth=2, markersize=8)
plt.xlabel('Window Length (time steps)')
plt.ylabel('F1 Score')
plt.title('LSTM vs GNN: F1 Score across Different Windows')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(windows)
plt.savefig('fig1_f1_comparison_corrected.png', dpi=300)
plt.show()

# ========== 图2：AUC 对比折线图 ==========
plt.figure(figsize=(8,5))
plt.plot(windows, lstm_auc, 'b-o', label='LSTM', linewidth=2, markersize=8)
plt.plot(windows, gnn_auc, 'r-s', label='GNN+Corr', linewidth=2, markersize=8)
plt.xlabel('Window Length (time steps)')
plt.ylabel('AUC')
plt.title('LSTM vs GNN: AUC across Different Windows')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(windows)
plt.savefig('fig2_auc_comparison_corrected.png', dpi=300)
plt.show()

# ========== 图3：GNN 双轴图 ==========
fig, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(windows, gnn_f1, 'b-o', label='F1', linewidth=2, markersize=8)
ax1.set_xlabel('Window Length (time steps)')
ax1.set_ylabel('F1 Score', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax2 = ax1.twinx()
ax2.plot(windows, gnn_auc, 'r-s', label='AUC', linewidth=2, markersize=8)
ax2.set_ylabel('AUC', color='r')
ax2.tick_params(axis='y', labelcolor='r')
plt.title('GNN+Corr Performance vs Window Length')
fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.savefig('fig3_gnn_dual_axis_corrected.png', dpi=300)
plt.show()

# ========== 图4：最佳窗口柱状图（窗口5和30） ==========
labels = ['W5 F1', 'W5 AUC', 'W30 F1', 'W30 AUC']
lstm_vals = [lstm_f1[0], lstm_auc[0], lstm_f1[3], lstm_auc[3]]
gnn_vals  = [gnn_f1[0], gnn_auc[0], gnn_f1[3], gnn_auc[3]]
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, lstm_vals, width, label='LSTM', color='steelblue')
plt.bar(x + width/2, gnn_vals, width, label='GNN+Corr', color='coral')
plt.xticks(x, labels, fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('LSTM vs GNN at Window 5 and Window 30', fontsize=14)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.ylim(0, 1)
plt.savefig('fig4_best_windows_comparison_corrected.png', dpi=300)
plt.show()

# ========== 图5：热力图（可选，需要加载数据） ==========
try:
    data_dir = "D:/Project/DataScience/smd/ServerMachineDataset/"
    train_df = pd.read_csv(f"{data_dir}train/machine-1-1.txt", header=None)
    corr = train_df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True, cbar_kws={"shrink":0.8})
    plt.title('Pearson Correlation Matrix of 38 Features')
    plt.savefig('fig5_correlation_heatmap_corrected.png', dpi=300)
    plt.show()
except Exception as e:
    print(f"热力图生成失败: {e}")

# ========== 图6：损失曲线（沿用之前数据） ==========
epochs = [10,20,30,40,50]
lstm_loss = [0.160858, 0.156242, 0.153821, 0.150917, 0.148900]
gnn_loss  = [0.163303, 0.157416, 0.154121, 0.150501, 0.147744]
plt.figure(figsize=(8,5))
plt.plot(epochs, lstm_loss, 'b-o', label='LSTM (window=5)')
plt.plot(epochs, gnn_loss, 'r-s', label='GNN+Corr (window=5)')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Curves (Window=5)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('fig6_loss_curves_corrected.png', dpi=300)
plt.show()

print("所有修正图表已生成完毕！")