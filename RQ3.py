import matplotlib.pyplot as plt
import pandas as pd

# 你的结果
data = {'window': [5,10,15,30,45,60],
        'f1': [0.2795,0.2692,0.2609,0.2607,0.2629,0.2648],
        'auc': [0.8624,0.8850,0.8742,0.9088,0.8913,0.8855]}
df = pd.DataFrame(data)

fig, ax1 = plt.subplots()
ax1.plot(df['window'], df['f1'], 'b-o', label='F1')
ax1.set_xlabel('Window Length (time steps)')
ax1.set_ylabel('F1 Score', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(df['window'], df['auc'], 'r-s', label='AUC')
ax2.set_ylabel('AUC', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Effect of Window Length on Anomaly Detection Performance (GNN+Corr)')
fig.tight_layout()
plt.savefig('window_ablation_final.png', dpi=300)
plt.show()