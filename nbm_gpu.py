import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def exp_y_mulwise_Xw(w, X, y):
    """
    w: parameter we need to figure out (dim,)
    X: data matrix (num_samples, dim)
    y: label (num_samples,)

    return exp(y * (X @ w))
    """
    return torch.exp(y * (X @ w))

def y_mulwise_Xw(w, X, y):
    """
    w: parameter we need to figure out (dim,)
    X: data matrix (num_samples, dim)
    y: label (num_samples,)

    return y * (X @ w)
    """
    return y * (X @ w)

def logistic_loss_and_grad(w, X, y, lam_regular = 0.05, short_cut = None):
    """
    w: parameter we need to figure out (dim,)
    X: data matrix (num_samples, dim)
    y: label (num_samples,)
    lam_regular: regularization parameter

    loss = np.mean(ln(1 + exp(-y (X @ w)))) + lam_regular * np.sum(w ** 2)

    grad = 1/num_samples * X^T @ (y / (1 + z) - y) + 2 * lam_regular * w

    """
    if short_cut is None:
        short_cut = y_mulwise_Xw(w, X, y)
    num_samples = X.shape[0]
    z = short_cut
    preds = torch.log((1 + torch.exp(-z))) # (num_samples,)
    loss = torch.mean(preds) + lam_regular * torch.sum(w ** 2)

    grad = 1 / num_samples * X.T @ (torch.divide(y, 1 + torch.exp(-z)) - y) + 2 * lam_regular * w # (dim,)

    return loss, grad

def backtracking_line_search(w , direction, X, y, alpha_step = 0.5, beta= 0.6, c = 0.5, short_cut = None, adaptive_c = False):
    """
    alpha_step: initial step size
    gamma: step size shrinkage factor (use this to find the smallest number t such that the Armijo condition is satisfied)
    direction: the direction on which we are going to search for the step size
    direction we used here is going to minimize the function
    c: constant for Armijo condition

    line search to find the step size (given w (x_k) and direction (p_k) to find the step_size t(alpha_k))
    Then next point is x_{k+1} = x_k + t * p_k

    update: We find the result is extremely sensetive to the value of c,
            so we want to adjust the value of c adaptively
            when the decrease of the loss is not enough, we want to increase the value of c
            but when we find the decrease of the loss is enough, we want to decrease the value of c
    """
    t = alpha_step
    loss, grad = logistic_loss_and_grad(w, X, y, short_cut=short_cut)
    if adaptive_c is True:
        while True:
            new_loss, _ = logistic_loss_and_grad(w + t * direction, X, y)
            if loss - new_loss < 0.005:
                c *= 1.1
            if new_loss <= loss + c * t * torch.dot(direction, grad) or loss - new_loss > 0.0005:
                break
            t *= beta
    
    else:
        while True:
            new_loss, _ = logistic_loss_and_grad(w + t * direction, X, y)
            if new_loss <= loss + c * t * torch.dot(direction, grad):
                break
            t *= beta

    return t

def Newton_method_find_direction(w, X, y, grad, lam_regular = 0.05, exp_short_cut = None):
    """
    w: parameter we need to figure out (dim.)
    X: date matrix  (num_samples, dim)
    y: label (num_samples,)
    lam_regular: regularization parameter

    Hessian = 1/num_samples * X^T @ D @ X + 2 * lam_regular * I

    ( (y ** 2) * short_cut / ((1 + short_cut) ** 2) * X^T @ X ) / num_samples + 2 * lam_regular * I

    Newton method: Given w(x_k) and X, y, lam_regular, short_cut, grad
    then return the direction p_k we used to update w(x_k -> x_{k+1})
    """
    num_samples = X.shape[0]
    if exp_short_cut is None:
        exp_short_cut = exp_y_mulwise_Xw(w, X, y)

    D = torch.diag((y ** 2) * exp_short_cut / ((1 + exp_short_cut) ** 2))
    Hessian = 1 / num_samples * (X.T @ D @ X) + 2 * lam_regular * torch.eye(X.shape[1], device = device) # Great idea
    return -torch.linalg.solve(Hessian, grad)


def logistic_regression_newton_backtracking(X, y, max_iter=1000, tol=1e-6, alpha = 0.4, beta = 0.8, lam_regular = 0.05, c = 0.5, adaptive_c = False, train_losses = [], grad_norms = []):
    """
    X: data matrix (num_samples, dim)
    y: label (num_samples,)
    max_iter: maximum number of iterations
    tol: tolerance for stopping criteria
    lam_regular: regularization parameter

    return w: the parameter we need to figure out (dim,)

    1. We need the initial point w(x_0) to start our iteration, then calculate the gradient at w(x_0)
    2. direction can be solved by Newton_method_find_direction
    3. Now we have w(x_0) and direction, we can use backtracking_line_search to find the step size
    4. update w(x_0) -> w(x_1) = w(x_0) + t * direction
    5. repeat 1-4 until the stopping criteria is satisfied
    """
    w = torch.zeros(X.shape[1], device = device, dtype = torch.float32)
    step_size = 0.5
    end_iter = None

    for i in range(max_iter):
        short_cut = y_mulwise_Xw(w, X, y)
        loss, grad = logistic_loss_and_grad(w, X, y, lam_regular, short_cut=short_cut)
        train_losses.append(loss.item())
        grad_norms.append(torch.norm(grad).item())

        direction = Newton_method_find_direction(w, X, y, grad, lam_regular, exp_short_cut=torch.exp(short_cut))
        step_size = backtracking_line_search(w, direction, X, y, short_cut=short_cut, alpha_step = alpha, beta = beta,c = c, adaptive_c=adaptive_c) # step_size find here garantee the Armijo condition (strictly decreasing)
        w_new = w + step_size * direction

        if torch.norm(w_new - w) < tol:
            w = w_new
            end_iter = i + 1
            break

        w = w_new
        print(f"Iter {i}, loss = {loss:.4f}")
    
    if end_iter is None:
        end_iter = max_iter - 1
    
    print(f'train_losses[:10]: {train_losses[:10]}')

    return w, end_iter
