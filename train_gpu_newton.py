from data_a9a_load import load_data
import numpy as np
import nbm_gpu as nbm
import torch

# 读取 a9a 数据
file_name = 'a9a.txt'
is_sparse = 1
feat = 123
batch_size = 32

filepath = './' + file_name

Xtrain, Ylabel = load_data(filepath, feat, is_sparse, batch_size,change_label = True)

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

for lam_regular in np.linspace(1e-5, 0.0001, 20):
    print(f"Regularization Parameter: {lam_regular}")
    w = nbm.logistic_regression_newton_backtracking(X_train_split, Y_train_split, max_iter=1000, tol=1e-6, lam_regular=lam_regular, c = 2, adaptive_c = False)
    val_loss, _ = nbm.logistic_loss_and_grad(w, X_val_split, Y_val_split)
    print(f"Validation Loss = {val_loss:.4f}")
    preds = (X_val_split @ w) > 0.5
    acc = torch.mean((preds == Y_val_split).float())
    print(f"Validation Accuracy = {acc * 100:.2f}%")

for c in np.linspace(1, 3, 20):
    print(f"Constant for Armijo Condition: {c}")
    w = nbm.logistic_regression_newton_backtracking(X_train_split, Y_train_split, max_iter=1000, tol=1e-6, lam_regular=0.00005, c = 2, adaptive_c = False)
    val_loss, _ = nbm.logistic_loss_and_grad(w, X_val_split, Y_val_split)
    print(f"Validation Loss = {val_loss:.4f}")
    preds = (X_val_split @ w) > 0.5
    acc = torch.mean((preds == Y_val_split).float())
    print(f"Validation Accuracy = {acc * 100:.2f}%")

def find_best_params(X_train_split, Y_train_split, X_val_split, Y_val_split):
    best_params = {'lam_regular': None, 'c': None}
    best_val_loss = float('inf')
    best_acc = 0

    for lam_regular in np.linspace(1e-5, 0.0001, 20):
        for c in np.linspace(1, 3, 20):
            print(f"Testing lam_regular: {lam_regular}, c: {c}")
            w = nbm.logistic_regression_newton_backtracking(X_train_split, Y_train_split, max_iter=1000, tol=1e-6, lam_regular=lam_regular, c=c, adaptive_c=False)
            val_loss, _ = nbm.logistic_loss_and_grad(w, X_val_split, Y_val_split)
            preds = (X_val_split @ w) > 0.5
            acc = torch.mean((preds == Y_val_split).float())
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_acc = acc
                best_params['lam_regular'] = lam_regular
                best_params['c'] = c

            print(f"Validation Loss = {val_loss:.4f}, Validation Accuracy = {acc * 100:.2f}%")

    print(f"Best Parameters: lam_regular = {best_params['lam_regular']}, c = {best_params['c']}")
    print(f"Best Validation Loss = {best_val_loss:.4f}, Best Validation Accuracy = {best_acc * 100:.2f}%")
    return best_params

# 调用函数以找到最佳参数
best_params = find_best_params(X_train_split, Y_train_split, X_val_split, Y_val_split)

# 训练
print("Start Training (Backtracking Logistic Regression)...")
w = nbm.logistic_regression_newton_backtracking(X_train_split, Y_train_split, max_iter=1000, tol=1e-6, lam_regular=0.00005, c = 2, adaptive_c = False)
print("Training Completed.")
print("Learned Parameters:", w)

# 验证集上的loss
val_loss, _ = nbm.logistic_loss_and_grad(w, X_val_split, Y_val_split)
print(f"Validation Loss = {val_loss:.4f}")

# 计算准确率
preds = (X_val_split @ w) > 0.5
acc = torch.mean((preds == Y_val_split).float())
print(f"Validation Accuracy = {acc * 100:.2f}%")

