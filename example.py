import numpy as np
from scipy.sparse import lil_matrix, vstack
from tqdm import tqdm

##############################################################################
# 第一步：读取 a9a 数据
##############################################################################
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

##############################################################################
# 第二步：定义逻辑回归与回溯搜索
##############################################################################
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_loss_and_grad(w, X, y):
    """
    w: 参数向量 (dim,)
    X: 数据矩阵 (num_samples, dim)
    y: 标签 (num_samples,)
    """
    m = X.shape[0]
    z = X.dot(w)
    preds = sigmoid(z)
    # 损失函数: -1/m * sum[ y*log(preds) + (1-y)*log(1-preds) ]
    # 为了避免log(0)，加一个很小的数1e-9
    loss = -np.mean(y * np.log(preds + 1e-9) + (1 - y) * np.log(1 - preds + 1e-9))
    # 梯度: 1/m * X^T (preds - y)
    grad = (1 / m) * X.T.dot(preds - y)
    return loss, grad

def backtracking_line_search(w, grad, X, y, alpha=0.4, beta=0.8):
    """
    使用回溯搜索确定合适的学习率
    """
    t = 1.0
    loss, _ = logistic_loss_and_grad(w, X, y)
    # Armijo 条件
    while True:
        new_loss, _ = logistic_loss_and_grad(w - t * grad, X, y)
        if new_loss <= loss - alpha * t * np.dot(grad, grad):  
            break
        t *= beta
    return t

def logistic_regression_backtracking(X, y, max_iter=1000, tol=1e-6):
    """
    使用带回溯搜索的梯度下降训练逻辑回归
    """
    w = np.zeros(X.shape[1])
    for i in range(max_iter):
        loss, grad = logistic_loss_and_grad(w, X, y)
        step_size = backtracking_line_search(w, grad, X, y)
        w_new = w - step_size * grad
        
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
        
        if i % 50 == 0:
            print(f"Iter {i}, loss = {loss:.4f}")
    return w

##############################################################################
# 第三步：训练并测试
##############################################################################
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
w = logistic_regression_backtracking(X_train_split, Y_train_split, max_iter=1000, tol=1e-6)
print("Training Completed.")
print("Learned Parameters:", w)

# 验证集上的loss
val_loss, _ = logistic_loss_and_grad(w, X_val_split, Y_val_split)
print(f"Validation Loss = {val_loss:.4f}")

# 计算准确率
preds = X_val_split.dot(w) > 0.0
acc = np.mean(preds == Y_val_split)
print(f"Validation Accuracy = {acc * 100:.2f}%")