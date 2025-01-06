from data_a9a_load import load_data
import numpy as np
import Newton_Backtracking_Method as nbm
from utils import sigmoid

# 读取 a9a 数据
file_name = 'a9a.txt'
is_sparse = 1
feat = 123
batch_size = 32

filepath = './' + file_name

Xtrain, Ylabel = load_data(filepath, feat, is_sparse, batch_size, change_label = False)

# 若是稀疏矩阵，需要转化为稠密矩阵
if is_sparse:
    Xtrain = Xtrain.toarray()

Xtrain = np.array(Xtrain, dtype=np.float32)
Ylabel = np.array(Ylabel, dtype=np.float32)


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
print(f"Sparsity of X_train: {np.mean(X_train_split == 0) * 100:.2f}%")
print(f"Sparsity of X_val: {np.mean(X_val_split == 0) * 100:.2f}%")
print(f'Y_train[:10]: {Y_train_split[:10]}')
print("Data Loaded.")

# for lam_regular in [0.00005, 0.0005, 0.005, 0.05, 0.5]:
#     print(f"Regularization Parameter: {lam_regular}")
#     w = nbm.logistic_regression_newton_backtracking(X_train_split, Y_train_split, max_iter=1000, tol=1e-6, lam_regular=lam_regular)
#     val_loss, _ = nbm.logistic_loss_and_grad(w, X_val_split, Y_val_split)
#     print(f"Validation Loss = {val_loss:.4f}")
#     preds = X_val_split.dot(w) > 0.5
#     acc = np.mean(preds == Y_val_split)
#     print(f"Validation Accuracy = {acc * 100:.2f}%")

# for c in np.linspace(0.3, 2.5, 10):
#     print(f"Constant for Armijo Condition: {c}")
#     w = nbm.logistic_regression_newton_backtracking(X_train_split, Y_train_split, max_iter=1000, tol=1e-6, lam_regular=0.00005, c=c)
#     val_loss, _ = nbm.logistic_loss_and_grad(w, X_val_split, Y_val_split)
#     print(f"Validation Loss = {val_loss:.4f}")
#     preds = X_val_split.dot(w) > 0.5
#     acc = np.mean(preds == Y_val_split)
#     print(f"Validation Accuracy = {acc * 100:.2f}%")


# 训练
print("Start Training (Backtracking Logistic Regression)...")
w = nbm.logistic_regression_newton_backtracking(X_train_split, Y_train_split, max_iter=1000, tol=1e-6, lam_regular=0.00005,c = 2, apdative_c=False)
print("Training Completed.")
print("Learned Parameters:", w)

# 验证集上的loss
val_loss, _ = nbm.logistic_loss_and_grad(w, X_val_split, Y_val_split)
print(f"Validation Loss = {val_loss:.4f}")

# 计算准确率
preds = X_val_split @ w > 0
acc = np.mean(preds == Y_val_split)
print(f"Validation Accuracy = {acc * 100:.2f}%")

