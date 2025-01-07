import torch
import numpy as np
from utils import save_results_to_json, plot_training_metrics, sigmoid, logistic_loss_and_grad, load_data, logistic_regression_backtracking
from utils import test_backtracking_params

# 设置设备为 GPU，如果可用的话
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##############################################################################
# 第一步：读取 a9a 数据
##############################################################################
file_name = 'a9a.txt'
is_sparse = 1
feat = 123
batch_size = 32

filepath = './' + file_name

Xtrain, Ylabel = load_data(filepath, feat, is_sparse, batch_size, change_label=True)

# 若是稀疏矩阵，需要转化为稠密矩阵
if is_sparse:
    Xtrain = torch.tensor(Xtrain.toarray(), dtype=torch.float32)
else:
    Xtrain = torch.tensor(Xtrain, dtype=torch.float32)

Ylabel = torch.tensor(Ylabel, dtype=torch.float32)

# 将数据移动到 GPU
Xtrain = Xtrain.to(device)
Ylabel = Ylabel.to(device)

##############################################################################
# 第二步：定义逻辑回归与回溯搜索
##############################################################################


##############################################################################
# 第三步：训练并测试
##############################################################################
# 划分训练集和验证集（90% vs 10%）
indices = torch.randperm(Xtrain.shape[0])
num_train = int(0.9 * len(indices))
train_idx = indices[:num_train]
val_idx = indices[num_train:]

X_train_split = Xtrain[train_idx]
Y_train_split = Ylabel[train_idx]
X_val_split = Xtrain[val_idx]
Y_val_split = Ylabel[val_idx]

train_losses = []  # 假设这是训练过程中记录的损失值列表
grad_norms = []    # 假设这是训练过程中记录的梯度范数列表
alpha = 0.4            # 假设这是超参数 alpha 的值
beta = 0.8         # 假设这是超参数 beta 的值
max_iter = 1000  # 假设这是最大迭代次数

# 训练
print("Start Training (Backtracking Logistic Regression)...")
w, end_iter = logistic_regression_backtracking(X_train_split, Y_train_split, max_iter=max_iter, alpha = alpha, beta = beta , tol=1e-6, gradient_end=True, train_losses=train_losses, grad_norms=grad_norms)
print("Training Completed.")
print("Learned Parameters:", w)

# 验证集上的loss
val_loss, _ = logistic_loss_and_grad(w, X_val_split, Y_val_split)
print(f"Validation Loss = {val_loss:.4f}")

# 计算准确率
preds = sigmoid(X_val_split.matmul(w)) > 0.5
acc = torch.mean((preds == Y_val_split).float())
print(f"Validation Accuracy = {acc * 100:.2f}%")

save_results_to_json(w = w, X_train=X_train_split, Y_train=Y_train_split, X_val=X_val_split, Y_val=Y_val_split, iterations = end_iter, filename = 'backtracking_gd_results.json')

plot_training_metrics(train_losses, grad_norms, alpha, beta, max_iter)

alphas = [0.01, 0.1, 0.5]
betas = [0.1, 0.5, 0.9]
test_backtracking_params(X_train_split, Y_train_split, X_val_split, Y_val_split, alphas, betas)