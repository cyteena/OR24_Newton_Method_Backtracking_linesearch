from data_a9a_load import load_data
import numpy as np
from Newton_Backtracking_Method import logistic_loss_and_grad, logistic_regression_newton_backtracking, sigmoid

file_name = 'a9a.txt'
is_sparse = 1
feat = 123
batch_size = 32

filepath = './' + file_name

Xtrain, Ylabel = load_data(filepath, feat, is_sparse, batch_size, change_label=True)

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

# 训练
print("Start Training (Backtracking Logistic Regression)...")
w = logistic_regression_newton_backtracking(X_train_split, Y_train_split, max_iter=1000, tol=1e-6, lam_regular=1e-6)
print("Training Completed.")
print("Learned Parameters:", w)

# 验证集上的loss
val_loss, _ = logistic_loss_and_grad(w, X_val_split, Y_val_split, lam_regular=1e-6)
print(f"Validation Loss = {val_loss:.4f}")

# 计算准确率
preds = sigmoid(X_val_split @ w) > 0.5
acc = np.mean(preds == Y_val_split)
print(f"Validation Accuracy = {acc * 100:.2f}%")
