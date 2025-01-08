import numpy as np
import torch
import json
from nbm_gpu import logistic_loss_and_grad, backtracking_line_search, Newton_method_find_direction, logistic_regression_newton_backtracking

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


def save_results_to_json(w, X_train, Y_train, X_val, Y_val, alpha, beta, lam_regular, c, iterations, filename="result.json"):
    # 训练集上的loss
    train_loss, _ = logistic_loss_and_grad(w, X_train, Y_train)
    
    # 验证集上的loss
    val_loss, _ = logistic_loss_and_grad(w, X_val, Y_val)
    
    # 计算验证集上的准确率
    preds = X_val.matmul(w) > 0.5
    acc = torch.mean((preds == Y_val).float())
    
    # 准备结果字典
    results = {
        "validation_accuracy": acc.item() * 100,
        "iterations": iterations,
        "train_loss": train_loss.item(),
        "val_loss": val_loss.item(),
        "alpha": alpha,
        "beta": beta,
        "lam_regular": lam_regular,
        "c": c
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

def plot_training_metrics(train_losses, grad_norms, alpha, beta, c, max_iter):
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
    filename = f"figure/training_metrics_{current_time}_alpha{alpha}_beta{beta}_c{c}_maxiter{max_iter}.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"Training metrics plot saved to {filename}")


import matplotlib.pyplot as plt
import datetime
import torch

def test_backtracking_params(X_train, Y_train, X_val, Y_val, alphas, betas, cs, max_iter=1000, tol=1e-6, gradient_end = False):
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)
    print(f'X_train.shape={X_train.shape}, Y_train.shape={Y_train.shape}')

    end_iter = None

    for c in cs:
        for alpha in alphas:
            for beta in betas:
                train_losses = []
                grad_norms = []
                w = torch.zeros(X_train.shape[1], device=device)  # 初始化权重
                for iter in range(max_iter):
                    loss, grad = logistic_loss_and_grad(w, X_train, Y_train)
                    train_losses.append(loss.item())
                    grad_norms.append(torch.norm(grad).item())

                    direction = Newton_method_find_direction(w=w, X=X_train, y=Y_train, grad=grad, lam_regular=1e-5)
                    step_size = backtracking_line_search(w, direction, X_train, Y_train, alpha_step=alpha, beta=beta, c=c, adaptive_c=False)
                    w_new = w + step_size * direction
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
                        print(f'alpha={alpha}, beta={beta}, c = {c}, iter={iter}, loss={loss:.4f}')
                if end_iter is None:
                    end_iter = max_iter

            results.append((alpha, beta, c, end_iter, train_losses, grad_norms))

    # 绘制图像
    plt.figure(figsize=(12, 6))
    for alpha, beta, c, end_iter, train_losses, grad_norms in results:
        plt.plot(train_losses, label=f'alpha={alpha}, beta={beta}, c = {c}, end_iter={end_iter}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Train Loss')
    plt.title('Train Loss Over Iterations for Different Backtracking Parameters')
    plt.legend()
    
    # 保存图表
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"figure/backtracking_params_{current_time}_gradient_end{gradient_end}.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"Backtracking parameters plot saved to {filename}")


def find_best_params(X_train_split, Y_train_split, X_val_split, Y_val_split):
    best_params = {'lam_regular': None, 'c': None}
    best_val_loss = float('inf')
    best_acc = 0

    for lam_regular in np.linspace(1e-5, 0.0001, 20):
        for c in np.linspace(1, 3, 20):
            print(f"Testing lam_regular: {lam_regular}, c: {c}")
            w = logistic_regression_newton_backtracking(X_train_split, Y_train_split, max_iter=1000, tol=1e-6, lam_regular=lam_regular, c=c, adaptive_c=False)
            val_loss, _ = logistic_loss_and_grad(w, X_val_split, Y_val_split)
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