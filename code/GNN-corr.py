import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
import scipy.sparse as sp

# -------------------- 1. 配置 --------------------
data_dir = "D:/Project/DataScience/smd/ServerMachineDataset/"
machine = "machine-1-1"
window = 10
hidden_dim = 64
batch_size = 64
epochs = 50
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- 2. 加载数据 --------------------
train_df = pd.read_csv(f"{data_dir}train/{machine}.txt", header=None)
test_df = pd.read_csv(f"{data_dir}test/{machine}.txt", header=None)
label_df = pd.read_csv(f"{data_dir}test_label/{machine}.txt", header=None)

scaler = StandardScaler()
train_data = scaler.fit_transform(train_df.values)
test_data = scaler.transform(test_df.values)
labels = label_df.values.flatten()

# -------------------- 3. 构建滑动窗口 --------------------
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, window)
X_test, y_test = create_sequences(test_data, window)
y_test_labels = labels[window:]

# 转为张量
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -------------------- 4. 构建皮尔逊相关系数图 --------------------
# 计算训练数据上特征之间的相关系数矩阵
corr = np.corrcoef(train_data.T)  # (38, 38)
threshold = 0.6
adj = (corr >= threshold).astype(float)
np.fill_diagonal(adj, 0)

# 转换为 PyG 的 edge_index 格式
edge_index = torch.tensor(np.where(adj == 1), dtype=torch.long)
print(f"Number of edges in correlation graph: {edge_index.shape[1]}")

# -------------------- 5. 定义 GNN+LSTM 模型 --------------------
class GNNLSTMAnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_index):
        super().__init__()
        self.edge_index = edge_index
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (batch, window, features)
        batch, window, f = x.shape
        # 合并 batch 和 window 维度，以便 GCN 处理
        x = x.reshape(batch * window, f)
        h = torch.relu(self.gcn(x, self.edge_index))
        h = h.reshape(batch, window, -1)
        out, _ = self.lstm(h)
        return self.fc(out[:, -1, :])

model = GNNLSTMAnomalyDetector(input_dim=train_data.shape[1], hidden_dim=hidden_dim, edge_index=edge_index).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -------------------- 6. 训练 --------------------
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

# -------------------- 7. 计算训练集残差阈值 --------------------
model.eval()
train_residuals = []
with torch.no_grad():
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        pred = model(batch_x).cpu()
        mse = torch.mean((pred - batch_y) ** 2, dim=1).numpy()
        train_residuals.extend(mse)
train_residuals = np.array(train_residuals)
threshold = np.percentile(train_residuals, 95)
print(f"Threshold: {threshold:.6f}")

# -------------------- 8. 测试集评估 --------------------
test_residuals = []
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        pred = model(batch_x).cpu()
        mse = torch.mean((pred - batch_y) ** 2, dim=1).numpy()
        test_residuals.extend(mse)
test_residuals = np.array(test_residuals)
y_pred = (test_residuals > threshold).astype(int)

min_len = min(len(y_test_labels), len(y_pred))
f1 = f1_score(y_test_labels[:min_len], y_pred[:min_len])
auc = roc_auc_score(y_test_labels[:min_len], test_residuals[:min_len])
print(f"GNN+LSTM (corr graph) -> F1: {f1:.4f}, AUC: {auc:.4f}")