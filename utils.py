import json
import torch
from scipy.sparse import lil_matrix, vstack
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

################################################################################

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
    end_iter = None
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
        if not end_iter:
            end_iter = max_iter
    return w, end_iter

################################################################################

def save_results_to_json(w, X_train, Y_train, X_val, Y_val, iterations, filename="result.json"):
    # 训练集上的loss
    train_loss, _ = logistic_loss_and_grad(w, X_train, Y_train)
    
    # 验证集上的loss
    val_loss, _ = logistic_loss_and_grad(w, X_val, Y_val)
    
    # 计算验证集上的准确率
    preds = sigmoid(X_val.matmul(w)) > 0.5
    acc = torch.mean((preds == Y_val).float())
    
    # 准备结果字典
    results = {
        "validation_accuracy": acc.item() * 100,
        "iterations": iterations,
        "train_loss": train_loss.item(),
        "val_loss": val_loss.item()
    }
    
    # 将结果追加到 JSON 文件中
    try:
        with open(filename, "r") as f:
            existing_results = json.load(f)
    except FileNotFoundError:
        existing_results = []

    existing_results.append(results)

    with open(filename, "w") as f:
        json.dump(existing_results, f, indent=4)
    
    print(f"Results appended to {filename}")

# # 示例调用
# iterations = 100  # 假设这是最终迭代次数
# save_results_to_json(w, X, y, X_val_split, Y_val_split, iterations)

import matplotlib.pyplot as plt
import datetime

def plot_training_metrics(train_losses, grad_norms, alpha, beta, max_iter):
    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.legend()
    
    # 绘制梯度范数
    plt.subplot(1, 2, 2)
    plt.plot(grad_norms, label='Gradient Norm')
    plt.xlabel('Iteration')
    plt.ylabel('Norm')
    plt.title('Gradient Norm Over Iterations')
    plt.legend()
    
    # 保存图表
    filename = f"figure/training_metrics_{current_time}_alpha{alpha}_beta{beta}_maxiter{max_iter}.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"Training metrics plot saved to {filename}")


import matplotlib.pyplot as plt
import datetime
import torch

def test_backtracking_params(X_train, Y_train, X_val, Y_val, alphas, betas, max_iter=1000, tol=1e-6, gradient_end = True):
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)
    print(f'X_train.shape={X_train.shape}, Y_train.shape={Y_train.shape}')

    end_iter = None

    for alpha in alphas:
        for beta in betas:
            train_losses = []
            grad_norms = []
            w = torch.zeros(X_train.shape[1], device=device)  # 初始化权重
            for iter in range(max_iter):
                loss, grad = logistic_loss_and_grad(w, X_train, Y_train)
                print(f'grad.shape={grad.shape}')
                train_losses.append(loss.item())
                grad_norms.append(torch.norm(grad).item())

                step_size = backtracking_line_search(w, grad, X_train, Y_train, alpha=alpha, beta=beta)
                w_new = w - step_size * grad
                if not gradient_end:
                    if torch.norm(w_new - w) < tol:
                        w = w_new
                        end_iter = iter
                        break
                else: 
                    if torch.norm(grad) < tol:
                        w = w_new
                        end_iter = iter
                        break
                w = w_new

                if iter % 100 == 0:
                    print(f'alpha={alpha}, beta={beta}, iter={iter}, loss={loss:.4f}')
            if not end_iter:
                end_iter = max_iter

            results.append((alpha, beta, end_iter, train_losses, grad_norms))

    # 绘制图像
    plt.figure(figsize=(12, 6))
    for alpha, beta, iter, train_losses, grad_norms in results:
        plt.plot(train_losses, label=f'alpha={alpha}, beta={beta}, iter={iter}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Train Loss')
    plt.title('Train Loss Over Iterations for Different Backtracking Parameters')
    plt.legend()
    
    # 保存图表
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"figure/backtracking_params_{current_time}.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"Backtracking parameters plot saved to {filename}")


# # 示例调用
# alphas = [0.01, 0.1, 0.5]
# betas = [0.1, 0.5, 0.9]
# test_backtracking_params(X_train_split, Y_train_split, X_val_split, Y_val_split, alphas, betas)

# # 示例调用
# train_losses = []  # 假设这是训练过程中记录的损失值列表
# grad_norms = []    # 假设这是训练过程中记录的梯度范数列表
# alpha = 0.01       # 假设这是超参数 alpha 的值
# beta = 0.5         # 假设这是超参数 beta 的值
# max_iter = 100000  # 假设这是最大迭代次数

# plot_training_metrics(train_losses, grad_norms, alpha, beta, max_iter)