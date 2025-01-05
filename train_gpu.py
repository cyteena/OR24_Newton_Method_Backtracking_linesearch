import numpy as np
from scipy.sparse import lil_matrix, vstack
from tqdm import tqdm
import torch

# 读取 a9a 数据
file_name = 'a9a.txt'
is_sparse = 1
feat = 123
batch_size = 32

filepath = './' + file_name

def parse_line(line):
    parts = line.strip().split()
    label = int(parts[0])
    # 将标签从 -1 转换为 0
    label = 0 if label == -1 else 1
    indices = []
    values = []
    for item in parts[1:]:
        index, value = item.split(':')
        indices.append(int(index))
        values.append(float(value))
    return label, indices, values

def load_data(filepath, feat, is_sparse=True, batch_size=1000):
    if is_sparse:
        Xtrain = lil_matrix((0, feat))
    else:
        Xtrain = np.zeros((0, feat))
    
    Ylabel = []
    batch_X = []
    batch_Y = []

    with open(filepath, 'r') as fid:
        for line in tqdm(fid, desc="Reading data"):
            label, indices, values = parse_line(line)
            batch_Y.append(label)
            
            if is_sparse:
                new_row = lil_matrix((1, feat))
                for idx, val in zip(indices, values):
                    if 0 <= idx < feat:
                        new_row[0, idx] = val
                batch_X.append(new_row)
            else:
                new_row = np.zeros((1, feat))
                for idx, val in zip(indices, values):
                    if 0 <= idx < feat:
                        new_row[0, idx] = val
                batch_X.append(new_row)
            
            if len(batch_X) >= batch_size:
                if is_sparse:
                    Xtrain = vstack([Xtrain] + batch_X)
                else:
                    Xtrain = np.vstack([Xtrain] + batch_X)
                Ylabel.extend(batch_Y)
                batch_X = []
                batch_Y = []

    if batch_X:
        if is_sparse:
            Xtrain = vstack([Xtrain] + batch_X)
        else:
            Xtrain = np.vstack([Xtrain] + batch_X)
        Ylabel.extend(batch_Y)

    return Xtrain, Ylabel

Xtrain, Ylabel = load_data(filepath, feat, is_sparse, batch_size)

# 若是稀疏矩阵，需要转化为稠密矩阵
if is_sparse:
    Xtrain = Xtrain.toarray()

Xtrain = np.array(Xtrain, dtype=np.float32)
Ylabel = np.array(Ylabel, dtype=np.float32)

# 将数据移动到 GPU 上
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
print(f"Sparsity of X_train: {np.mean(X_train_split.cpu().numpy() == 0) * 100:.2f}%")
print(f"Sparsity of X_val: {np.mean(X_val_split.cpu().numpy() == 0) * 100:.2f}%")
print(f'Y_train[:10]: {Y_train_split[:10]}')
print("Data Loaded.")

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

def logistic_loss_and_grad(w, X, y, lam_regular=0.0):
    num_samples = X.shape[0]
    z = X @ w
    preds = sigmoid(z)
    loss = -torch.mean(y * torch.log(preds + 1e-9) + (1 - y) * torch.log(1 - preds + 1e-9)) + lam_regular * torch.sum(w ** 2)
    grad = (1 / num_samples) * X.T @ (preds - y) + 2 * lam_regular * w
    return loss, grad

def backtracking_line_search(w, grad, X, y, alpha=0.4, beta=0.8):
    t = 1.0
    loss, _ = logistic_loss_and_grad(w, X, y)
    while True:
        new_loss, _ = logistic_loss_and_grad(w - t * grad, X, y)
        if new_loss <= loss - alpha * t * torch.dot(grad, grad):
            break
        t *= beta
    return t

def logistic_regression_backtracking(X, y, max_iter=1000, tol=1e-6, lam_regular=0.0):
    w = torch.zeros(X.shape[1], device=device, dtype=torch.float32)
    for i in range(max_iter):
        loss, grad = logistic_loss_and_grad(w, X, y, lam_regular)
        step_size = backtracking_line_search(w, grad, X, y)
        w_new = w - step_size * grad
        
        if torch.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
        
        if i % 50 == 0:
            print(f"Iter {i}, loss = {loss.item():.4f}")
    return w

# 训练
print("Start Training (Backtracking Logistic Regression)...")
w = logistic_regression_backtracking(X_train_split, Y_train_split, max_iter=1000, tol=1e-6, lam_regular=0.00005)
print("Training Completed.")
print("Learned Parameters:", w)

# 验证集上的loss
val_loss, _ = logistic_loss_and_grad(w, X_val_split, Y_val_split)
print(f"Validation Loss = {val_loss.item():.4f}")

# 计算准确率
preds = sigmoid(X_val_split @ w) > 0.5
acc = torch.mean((preds == Y_val_split).float())
print(f"Validation Accuracy = {acc.item() * 100:.2f}%")