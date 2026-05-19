import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
import time
import matplotlib.pyplot as plt

# -------------------- 配置 --------------------
data_dir = "D:/Project/DataScience/smd/ServerMachineDataset/"
machine = "machine-1-1"
hidden_dim = 64
batch_size = 64
epochs = 50          # 可酌情减少到20以快速验证趋势
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 要测试的窗口列表
windows = [5, 10, 15, 30, 45, 60]

# 存储结果
results = []

# -------------------- 加载原始数据（只加载一次） --------------------
train_df = pd.read_csv(f"{data_dir}train/{machine}.txt", header=None)
test_df = pd.read_csv(f"{data_dir}test/{machine}.txt", header=None)
label_df = pd.read_csv(f"{data_dir}test_label/{machine}.txt", header=None)

scaler = StandardScaler()
train_data = scaler.fit_transform(train_df.values)
test_data = scaler.transform(test_df.values)
labels = label_df.values.flatten()

# -------------------- 定义滑动窗口函数 --------------------
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

# -------------------- 定义 LSTM 模型 --------------------
class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# -------------------- 循环训练每个窗口 --------------------
for window in windows:
    print(f"\n===== Training LSTM with window = {window} =====")
    start_time = time.time()

    # 构建数据集
    X_train, y_train = create_sequences(train_data, window)
    X_test, y_test = create_sequences(test_data, window)
    y_test_labels = labels[window:]  # 对齐标签

    # 转换为张量
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = LSTMAnomalyDetector(input_dim=train_data.shape[1], hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")

    # 计算训练集残差（用于确定阈值）
    model.eval()
    train_residuals = []
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x).cpu()
            mse = torch.mean((pred - batch_y) ** 2, dim=1).numpy()
            train_residuals.extend(mse)
    threshold = np.percentile(train_residuals, 95)  # 95%分位数
    print(f"Threshold: {threshold:.6f}")

    # 测试集评估
    test_residuals = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x).cpu()
            mse = torch.mean((pred - batch_y) ** 2, dim=1).numpy()
            test_residuals.extend(mse)
    y_pred = (np.array(test_residuals) > threshold).astype(int)

    # 对齐长度
    min_len = min(len(y_test_labels), len(y_pred))
    f1 = f1_score(y_test_labels[:min_len], y_pred[:min_len])
    auc = roc_auc_score(y_test_labels[:min_len], test_residuals[:min_len])

    elapsed = time.time() - start_time
    print(f"Window {window} -> F1: {f1:.4f}, AUC: {auc:.4f}, Time: {elapsed:.1f}s")
    results.append({'window': window, 'f1': f1, 'auc': auc})

# -------------------- 输出汇总 --------------------
print("\n" + "="*40)
print("LSTM Results Summary")
print("="*40)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# 保存到CSV
df_results.to_csv('lstm_all_windows_results.csv', index=False)

# 可选：绘制LSTM窗口消融图
plt.figure()
plt.plot(df_results['window'], df_results['f1'], 'b-o', label='F1')
plt.plot(df_results['window'], df_results['auc'], 'r-s', label='AUC')
plt.xlabel('Window Length')
plt.ylabel('Score')
plt.title('LSTM Performance vs Window Length')
plt.legend()
plt.grid(True)
plt.savefig('lstm_window_ablation.png')
plt.show()

print("\nAll experiments finished. Results saved to 'lstm_all_windows_results.csv'.")