import numpy as np

from sqmf_common import (
    backtracking, basis_direction, compute_residual, latent_direction,
    objective_value, quadratic_direction, quadratic_features,
    residual_gradient, result_dict, stiefel_retraction, symmetrize,
)


def irgu_sqmf(
    X, c, U, V, T, A, p=2.0,
    alpha_c=1.0, alpha_W=1.0, alpha_T=1.0, alpha_A=1.0,
    beta=0.5, armijo=1e-4, tol=1e-6, patience=5,
    min_step=1e-12, max_iter=10000,
):
    """Algorithm 3: refresh residual gradients after each block update."""
    c, U, V, T, A = (np.array(a, dtype=float, copy=True) for a in (c, U, V, T, A))
    A = symmetrize(A)
    d = T.shape[0]
    R, Q = compute_residual(X, c, U, V, T, A)
    F = objective_value(R, p)
    history = [F]
    steps = [alpha_c, alpha_W, alpha_T, alpha_A]
    stable = 0

    for _ in range(max_iter):
        F_start = F
        G = residual_gradient(R, p)
        D_c = G.sum(axis=1)
        a = backtracking(
            lambda a: objective_value(R - a * D_c[:, None], p),
            F, D_c, steps[0], beta, armijo, min_step,
        )
        if a:
            c_new = c + a * D_c
            R -= (c_new - c)[:, None]
            c = c_new
            F = objective_value(R, p)
            steps[0] = min(a / beta, 1e6)

        G = residual_gradient(R, p)
        W = np.column_stack((U, V))
        D_W = basis_direction(G, U, V, T, Q)

        def objective_W(a):
            W_new = stiefel_retraction(W + a * D_W)
            return objective_value(X - c[:, None] - W_new[:, :d] @ T
                                   - W_new[:, d:] @ Q, p)

        a = backtracking(objective_W, F, D_W, steps[1], beta, armijo, min_step)
        if a:
            W_new = stiefel_retraction(W + a * D_W)
            U_new, V_new = W_new[:, :d], W_new[:, d:]
            R -= (U_new - U) @ T
            R -= (V_new - V) @ Q
            U, V = U_new, V_new
            F = objective_value(R, p)
            steps[1] = min(a / beta, 1e6)

        G = residual_gradient(R, p)
        D_T = latent_direction(G, U, V, T, A)

        def objective_T(a):
            T_new = T + a * D_T
            return objective_value(X - c[:, None] - U @ T_new
                                   - V @ quadratic_features(A, T_new), p)

        a = backtracking(objective_T, F, D_T, steps[2], beta, armijo, min_step)
        if a:
            T_new = T + a * D_T
            Q_new = quadratic_features(A, T_new)
            R -= U @ (T_new - T)
            R -= V @ (Q_new - Q)
            T, Q = T_new, Q_new
            F = objective_value(R, p)
            steps[2] = min(a / beta, 1e6)

        G = residual_gradient(R, p)
        D_A = quadratic_direction(G, V, T)

        def objective_A(a):
            A_new = symmetrize(A + a * D_A)
            return objective_value(X - c[:, None] - U @ T
                                   - V @ quadratic_features(A_new, T), p)

        a = backtracking(objective_A, F, D_A, steps[3], beta, armijo, min_step)
        if a:
            A_new = symmetrize(A + a * D_A)
            Q_new = quadratic_features(A_new, T)
            R -= V @ (Q_new - Q)
            A, Q = A_new, Q_new
            steps[3] = min(a / beta, 1e6)

        R, Q = compute_residual(X, c, U, V, T, A)
        F = objective_value(R, p)
        history.append(F)
        change = abs(F_start - F) / max(1.0, abs(F_start))
        stable = stable + 1 if change <= tol else 0
        if stable >= patience:
            break

    return result_dict(X, c, U, V, T, A, p, history, stable >= patience)
