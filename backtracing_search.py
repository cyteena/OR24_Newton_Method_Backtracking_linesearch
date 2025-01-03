import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_loss_and_grad(w, X, y):
    """
    w: 参数向量 (dim,)
    X: 数据矩阵 (num_samples, dim)
    y: 标签 (num_samples,)
    返回: (损失, 梯度)
    """
    m = X.shape[0]
    z = X.dot(w)
    preds = sigmoid(z)
    # 损失函数: -1/m * sum[ y*log(preds) + (1-y)*log(1-preds) ]
    loss = -np.mean(y * np.log(preds + 1e-9) + (1 - y) * np.log(1 - preds + 1e-9))
    # 梯度: 1/m * X^T (preds - y)
    grad = (1 / m) * X.T.dot(preds - y)
    return loss, grad

def backtracking_line_search(w, grad, X, y, alpha=0.4, beta=0.8):
    """
    使用回溯搜索确定合适的学习率
    alpha, beta 为常见的回溯搜索参数
    """
    t = 1.0
    loss, _ = logistic_loss_and_grad(w, X, y)
    while True:
        new_loss, _ = logistic_loss_and_grad(w - t * grad, X, y)
        # 判断是否满足 Armijo 条件
        if new_loss <= loss - alpha * t * np.dot(grad, grad):
            break
        t *= beta
    return t

def logistic_regression_backtracking(X, y, max_iter=1000, tol=1e-6):
    """
    使用带回溯搜索的梯度下降训练逻辑回归
    """
    # 初始化参数向量
    w = np.zeros(X.shape[1])
    for i in range(max_iter):
        loss, grad = logistic_loss_and_grad(w, X, y)
        # 回溯搜索
        step_size = backtracking_line_search(w, grad, X, y)
        w_new = w - step_size * grad
        
        # 如果收敛就停止
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
        if i % 50 == 0:
            print(f"Iter {i}, loss = {loss:.4f}")
    return w

# ------------------ 使用示例 ------------------
# 假设你已经使用前面的代码读取了 a9a 数据, 得到了 Xtrain, Ylabel (numpy 数组格式)
# 确保 Ylabel 中的标签是 0 或 1

# 例如:
# Xtrain: (num_samples, num_features)
# Ylabel: (num_samples,)

# 为了与上面函数匹配，此处 Xtrain 应为 [num_samples, dim]
# 如果你的数据是转置的，请先转成这样的形状

# w = logistic_regression_backtracking(Xtrain, Ylabel)
# print("Finished training.")
# print("Learned parameters:", w)