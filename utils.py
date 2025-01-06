import json
import torch
from example_backtracking_gd import logistic_loss_and_grad, sigmoid

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

# # 示例调用
# train_losses = []  # 假设这是训练过程中记录的损失值列表
# grad_norms = []    # 假设这是训练过程中记录的梯度范数列表
# alpha = 0.01       # 假设这是超参数 alpha 的值
# beta = 0.5         # 假设这是超参数 beta 的值
# max_iter = 100000  # 假设这是最大迭代次数

# plot_training_metrics(train_losses, grad_norms, alpha, beta, max_iter)