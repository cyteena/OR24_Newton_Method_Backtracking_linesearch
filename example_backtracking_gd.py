import torch
from scipy.sparse import lil_matrix, vstack
from tqdm import tqdm
from utils import save_results_to_json, plot_training_metrics

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

def parse_line(line, change_label = False):
    parts = line.strip().split()
    label = int(parts[0])
    if change_label:
        # 将标签从 -1 转换为 0
        label = 0 if label == -1 else 1
    indices = []
    values = []
    for item in parts[1:]:
        index, value = item.split(':')
        indices.append(int(index))
        values.append(float(value))
    return label, indices, values

def load_data(filepath, feat, is_sparse=True, batch_size=1000, change_label = False):
    if is_sparse:
        Xtrain = lil_matrix((0, feat))
    else:
        Xtrain = torch.zeros((0, feat))
    
    Ylabel = []
    batch_X = []
    batch_Y = []

    with open(filepath, 'r') as fid:
        for line in tqdm(fid, desc="Reading data"):
            label, indices, values = parse_line(line, change_label)
            batch_Y.append(label)
            
            if is_sparse:
                new_row = lil_matrix((1, feat))
                for idx, val in zip(indices, values):
                    if 0 <= idx < feat:
                        new_row[0, idx] = val
                batch_X.append(new_row)
            else:
                new_row = torch.zeros((1, feat))
                for idx, val in zip(indices, values):
                    if 0 <= idx < feat:
                        new_row[0, idx] = val
                batch_X.append(new_row)
            
            if len(batch_X) >= batch_size:
                if is_sparse:
                    Xtrain = vstack([Xtrain] + batch_X)
                else:
                    Xtrain = torch.cat([Xtrain] + batch_X, dim=0)
                Ylabel.extend(batch_Y)
                batch_X = []
                batch_Y = []

    if batch_X:
        if is_sparse:
            Xtrain = vstack([Xtrain] + batch_X)
        else:
            Xtrain = torch.cat([Xtrain] + batch_X, dim=0)
        Ylabel.extend(batch_Y)

    return Xtrain, Ylabel

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
def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

def logistic_loss_and_grad(w, X, y):
    """
    w: 参数向量 (dim,)
    X: 数据矩阵 (num_samples, dim)
    y: 标签 (num_samples,)
    """
    m = X.shape[0]
    z = X.matmul(w)
    preds = sigmoid(z)
    # 损失函数: -1/m * sum[ y*log(preds) + (1-y)*log(1-preds) ]
    # 为了避免log(0)，加一个很小的数1e-9
    loss = -torch.mean(y * torch.log(preds + 1e-9) + (1 - y) * torch.log(1 - preds + 1e-9))
    # 梯度: 1/m * X^T (preds - y)
    grad = (1 / m) * X.t().matmul(preds - y)
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
        if new_loss <= loss - alpha * t * torch.dot(grad, grad):  
            break
        t *= beta
    return t

def logistic_regression_backtracking(X, y, max_iter=1000, tol=1e-6, alpha = 0.4, beta = 0.8, gradient_end = False, train_losses = [], grad_norms = []):
    """
    使用带回溯搜索的梯度下降训练逻辑回归
    """
    w = torch.zeros(X.shape[1], dtype=torch.float32, device=device)
    for i in range(max_iter):
        loss, grad = logistic_loss_and_grad(w, X, y)
        train_losses.append(loss.item())
        grad_norms.append(torch.norm(grad).item())
        step_size = backtracking_line_search(w, grad, X, y, alpha=alpha, beta=beta)
        w_new = w - step_size * grad
        if not gradient_end:
            if torch.norm(w_new - w) < tol:
                w = w_new
                end_iter = i
                break
        else: 
            if torch.norm(grad) < tol:
                w = w_new
                end_iter = i
                break
        w = w_new
        
        if i % 50 == 0:
            print(f"Iter {i}, loss = {loss:.4f}")
    return w, end_iter

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
w, end_iter = logistic_regression_backtracking(X_train_split, Y_train_split, max_iter=max_iter, alpha = alpha, beta = beta , tol=1e-6, gradient_end=True)
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