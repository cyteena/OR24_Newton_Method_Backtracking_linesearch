from data_a9a_load import load_data
import numpy as np
import nbm_gpu as nbm
import torch
from utils import save_results_to_json, plot_training_metrics, test_backtracking_params, sigmoid

# 读取 a9a 数据
file_name = 'a9a.txt'
is_sparse = 1
feat = 123
batch_size = 32

filepath = './' + file_name

Xtrain, Ylabel = load_data(filepath, feat, is_sparse, batch_size, change_label = True)

# 若是稀疏矩阵，需要转化为稠密矩阵
if is_sparse:
    Xtrain = Xtrain.toarray()

Xtrain = np.array(Xtrain, dtype=np.float32)
Ylabel = np.array(Ylabel, dtype=np.float32)

# move on to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Xtrain = torch.tensor(Xtrain, device=device)
Ylabel = torch.tensor(Ylabel, device=device)


# 划分训练集和验证集（90% vs 10%）
indices = np.arange(Xtrain.shape[0])
np.random.shuffle(indices)
num_train = int(0.9 * len(indices))
train_idx = indices[:num_train]
val_idx = indices[num_train:]

X_train_split = Xtrain[train_idx]
Y_train_split = Ylabel[train_idx]
X_val_split = Xtrain[val_idx]
Y_val_split = Ylabel[val_idx]
print(f"Train: {X_train_split.shape}, {Y_train_split.shape}")
print(f"Validation: {X_val_split.shape}, {Y_val_split.shape}")
print(f"Sparsity of X_train: {torch.mean((X_train_split == 0).float()) * 100:.2f}%")
print(f"Sparsity of X_val: {torch.mean((X_val_split == 0).float()) * 100:.2f}%")
print(f'Y_train[:10]: {Y_train_split[:10]}')
print("Data Loaded.")


train_losses = []  # 假设这是训练过程中记录的损失值列表
grad_norms = []    # 假设这是训练过程中记录的梯度范数列表
alpha = 0.5            # 假设这是超参数 alpha 的值
beta = 0.9         # 假设这是超参数 beta 的值
max_iter = 1000  # 假设这是最大迭代次数
c = 2             # 假设这是Armjio常数 initial c = 2, newton method 一步结束
lam_regular = 1e-5  # 假设这是正则化系数


# 训练
print("Start Training (Backtracking Logistic Regression)...")
w, end_iter = nbm.logistic_regression_newton_backtracking(X_train_split, Y_train_split, max_iter=max_iter, tol=1e-6, lam_regular=lam_regular, c = c, adaptive_c = False, train_losses = train_losses, grad_norms=grad_norms)
print(f'end_iter = {end_iter}')
print(f'train_losses is None? {train_losses is None}')
print(f'grad_norms is None? {grad_norms is None}')
print("Training Completed.")
print("Learned Parameters:", w)

# 验证集上的loss
val_loss, _ = nbm.logistic_loss_and_grad(w, X_val_split, Y_val_split)
print(f"Validation Loss = {val_loss:.4f}")

# 计算准确率
preds = (X_val_split @ w) > 0.5
acc = torch.mean((preds == Y_val_split).float())
print(f"Validation Accuracy = {acc * 100:.2f}%")

save_results_to_json(w = w, X_train=X_train_split, Y_train=Y_train_split, X_val=X_val_split, Y_val=Y_val_split, alpha = alpha, beta = beta, lam_regular = lam_regular, c=c, iterations = end_iter, filename = 'backtracking_gd_results.json')

print("train_losses:前五项", train_losses[:5],"grad_norms:前五项", grad_norms[:5])

plot_training_metrics(train_losses = train_losses, grad_norms = grad_norms, alpha = alpha, beta = beta, c = c, max_iter = max_iter)


max_iter = 200
alphas = [0.01, 0.5, 0.9]
betas = [0.3, 0.5, 0.9]
cs = [0.5, 1, 2]
test_backtracking_params(X_train_split, Y_train_split, X_val_split, Y_val_split, alphas, betas, cs, max_iter = max_iter, gradient_end=False)

