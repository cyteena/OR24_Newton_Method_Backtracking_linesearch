import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def exp_y_mulwise_Xw(w, X, y):
    """
    w: parameter we need to figure out (dim,)
    X: data matrix (num_samples, dim)
    y: label (num_samples,)

    return exp(y * (X @ w))
    """
    return np.exp(np.multiply(y, X @ w))

def logistic_loss_and_grad(w, X, y, lam_regular = 0.0):
    """
    w: parameter we need to figure out (dim,)
    X: data matrix (num_samples, dim)
    y: label (num_samples,)
    lam_regular: regularization parameter

    loss = np.mean(ln(1 + exp(-y (X @ w)))) + lam_regular * np.sum(w ** 2)

    grad = 1/num_samples * X^T @ (y / (1 + z) - y) + 2 * lam_regular * w

    """
    num_samples = X.shape[0]
    z = np.multiply(y, X @ w) # (num_samples,)
    preds = np.log((1 + np.exp(-z))) # (num_samples,)
    loss = np.mean(preds) + lam_regular * np.sum(w ** 2)

    grad = 1 / num_samples * X.T @ (np.divide(y, 1 + np.exp(-z)) - y) + 2 * lam_regular * w # (dim,)

    return loss, grad

def backtracking_line_search(w , direction, X, y, alpha_step = 0.4, gamma = 0.8, c = 0.001):
    """
    alpha_step: initial step size
    gamma: step size shrinkage factor (use this to find the smallest number t such that the Armijo condition is satisfied)
    direction: the direction on which we are going to search for the step size
    direction we used here is going to minimize the function
    c: constant for Armijo condition

    line search to find the step size (given w (x_k) and direction (p_k) to find the step_size t(alpha_k))
    Then next point is x_{k+1} = x_k + t * p_k
    """
    t = alpha_step
    loss, grad = logistic_loss_and_grad(w, X, y)
    while True:
        new_loss, _ = logistic_loss_and_grad(w + t * direction, X, y)
        if new_loss <= loss + c * t * np.dot(direction, grad):
            break
        t *= gamma

    return t

def Newton_method_find_direction(w, X, y, grad, lam_regular = 0.0, short_cut = None):
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
    if short_cut is None:
        short_cut = exp_y_mulwise_Xw(w, X, y)

    D = np.diag((y ** 2) * short_cut / ((1 + short_cut) ** 2))
    Hessian = 1 / num_samples * X.T @ D @ X + 2 * lam_regular * np.eye(X.shape[1]) # Great idea
    return -np.linalg.inv(Hessian) @ grad


def logistic_regression_newton_backtracking(X, y, max_iter=1000, tol=1e-6, lam_regular = 0.0):
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
    w = np.zeros(X.shape[1])

    for i in range(max_iter):
        loss, grad = logistic_loss_and_grad(w, X, y, lam_regular)
        direction = Newton_method_find_direction(w, X, y, grad, lam_regular)
        step_size = backtracking_line_search(w, direction, X, y) # step_size find here garantee the Armijo condition (strictly decreasing)
        w_new = w + step_size * direction

        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break

        w = w_new
        if i % 50 == 0:
            print(f"Iter {i}, loss = {loss:.4f}")

    return w




