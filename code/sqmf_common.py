import numpy as np


def symmetrize(A):
    return 0.5 * (A + A.swapaxes(1, 2))


def quadratic_features(A, T):
    return np.einsum("aij,in,jn->an", A, T, T)


def compute_residual(X, c, U, V, T, A):
    Q = quadratic_features(A, T)
    return X - c[:, None] - U @ T - V @ Q, Q


def objective_value(R, p):
    return np.sum(np.abs(R) ** p) / p


def residual_gradient(R, p):
    return np.sign(R) * np.abs(R) ** (p - 1)


def stiefel_retraction(W):
    Q, R = np.linalg.qr(W, mode="reduced")
    return Q * np.where(np.diag(R) < 0, -1.0, 1.0)


def basis_direction(G, U, V, T, Q):
    W = np.column_stack((U, V))
    H = np.column_stack((-G @ T.T, -G @ Q.T))
    H_tan = H - W @ (0.5 * (W.T @ H + H.T @ W))
    return -H_tan


def latent_direction(G, U, V, T, A):
    direction = np.empty_like(T)
    for i in range(T.shape[1]):
        tau, g = T[:, i], G[:, i]
        J = np.einsum("aij,j->ai", A, tau) + np.einsum("aji,j->ai", A, tau)
        gradient = -U.T @ g - J.T @ (V.T @ g)
        direction[:, i] = -gradient
    return direction


def quadratic_direction(G, V, T):
    return symmetrize(np.einsum("an,in,jn->aij", V.T @ G, T, T))


def backtracking(objective, F, direction, alpha, beta, armijo, min_step,
                 max_backtracks=64):
    norm_sq = np.sum(direction ** 2)
    if norm_sq <= 1e-30:
        return 0.0
    for _ in range(max_backtracks):
        if alpha < min_step:
            break
        F_new = objective(alpha)
        if np.isfinite(F_new) and F_new <= F - armijo * alpha * norm_sq:
            return alpha
        alpha *= beta
    return 0.0


def result_dict(X, c, U, V, T, A, p, history, converged):
    R, Q = compute_residual(X, c, U, V, T, A)
    return dict(c=c, U=U, V=V, T=T, A=A, R=R, Q=Q,
                objective=objective_value(R, p), history=np.asarray(history),
                n_iter=len(history) - 1, converged=converged)
