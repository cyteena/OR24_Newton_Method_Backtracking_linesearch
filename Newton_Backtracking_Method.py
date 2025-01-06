import numpy as np

def exp_y_mulwise_Xw(w, X, y):
    """
    w: parameter we need to figure out (dim,)
    X: data matrix (num_samples, dim)
    y: label (num_samples,)

    return exp(y * (X @ w))
    """
    return np.exp(np.multiply(y, X @ w))

def y_mulwise_Xw(w, X, y):
    """
    w: parameter we need to figure out (dim,)
    X: data matrix (num_samples, dim)
    y: label (num_samples,)

    return y * (X @ w)
    """
    return np.multiply(y, X @ w)

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
    preds = np.log((1 + np.exp(-z))) # (num_samples,)
    loss = np.mean(preds) + lam_regular * np.sum(w ** 2)

    grad = 1 / num_samples * X.T @ (np.divide(y, 1 + np.exp(-z)) - y) + 2 * lam_regular * w # (dim,)

    return loss, grad

def backtracking_line_search(w , direction, X, y, alpha_step = 0.5, gamma = 0.6, c = 0.5, short_cut = None, adaptive_c = True):
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
            if new_loss <= loss + c * t * np.dot(direction, grad) or loss - new_loss > 0.0005:
                break
            t *= gamma
    
    else:
        while True:
            new_loss, _ = logistic_loss_and_grad(w + t * direction, X, y)
            if new_loss <= loss + c * t * np.dot(direction, grad):
                break
            t *= gamma

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

    D = np.diag((y ** 2) * exp_short_cut / ((1 + exp_short_cut) ** 2))
    Hessian = 1 / num_samples * X.T @ D @ X + 2 * lam_regular * np.eye(X.shape[1]) # Great idea
    return -np.linalg.solve(Hessian, grad)


def logistic_regression_newton_backtracking(X, y, max_iter=1000, tol=1e-6, lam_regular = 0.000005, c = 2, apdative_c = False):
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
    step_size = 0.5

    for i in range(max_iter):
        short_cut = y_mulwise_Xw(w, X, y)
        loss, grad = logistic_loss_and_grad(w, X, y, lam_regular, short_cut=short_cut)
        direction = Newton_method_find_direction(w, X, y, grad, lam_regular, exp_short_cut=np.exp(short_cut))
        step_size = backtracking_line_search(w, direction, X, y, short_cut=short_cut, alpha_step = step_size, c = c, adaptive_c = apdative_c) # step_size find here garantee the Armijo condition (strictly decreasing)
        w_new = w + step_size * direction

        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break

        w = w_new
        print(f"Iter {i}, loss = {loss:.4f}")

    return w
