import numpy as np


def ordinary_gd_sqmf(
    X,
    c,
    U,
    V,
    T,
    A,
    p=2.0,
    alpha_c=1.0,
    alpha_W=1.0,
    alpha_T=1.0,
    alpha_A=1.0,
    beta=0.5,
    armijo=1e-4,
    tol=1e-6,
    patience=5,
    min_step=1e-12,
    max_backtracks=50,
    safety_max_iter=10000,
    verbose=True,
):


    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------
    X = np.asarray(X, dtype=float)
    c = np.asarray(c, dtype=float).reshape(-1).copy()
    U = np.asarray(U, dtype=float).copy()
    V = np.asarray(V, dtype=float).copy()
    T = np.asarray(T, dtype=float).copy()
    A = symmetrize_quadratic_map(
        np.asarray(A, dtype=float)
    )

    if p <= 1:
        raise ValueError("p must be greater than 1.")

    if not 0 < beta < 1:
        raise ValueError("beta must be between 0 and 1.")

    if patience < 1 or tol <= 0:
        raise ValueError("patience and tol must be positive.")

    if max_backtracks < 1 or safety_max_iter < 1:
        raise ValueError("Iteration limits must be positive.")

    d, n = X.shape
    r = T.shape[0]
    q = V.shape[1]

    if c.shape != (d,):
        raise ValueError("c must have shape (d,).")
    if U.shape != (d, r):
        raise ValueError("U must have shape (d, r).")
    if V.shape != (d, q):
        raise ValueError("V must have shape (d, q).")
    if T.shape != (r, n):
        raise ValueError("T must have shape (r, n).")
    if A.shape != (q, r, r):
        raise ValueError("A must have shape (q, r, r).")
    if r + q > d:
        raise ValueError("Require r + q <= d.")

    W = np.concatenate([U, V], axis=1)

    if not np.allclose(
        W.T @ W, np.eye(r + q), atol=1e-6
    ):
        raise ValueError(
            "The initial combined basis W=[U,V] "
            "must have orthonormal columns."
        )

    R, Q = compute_residual(X, c, U, V, T, A)
    F_current = objective_value(R, p)

    history = [F_current]
    step_history = []

    alpha = {
        "c": float(alpha_c),
        "W": float(alpha_W),
        "T": float(alpha_T),
        "A": float(alpha_A),
    }

    stable_count = 0
    converged = False
    stop_reason = "safety_max_iter reached"

    if verbose:
        print(f"Initial objective: {F_current:.8e}")

    for iteration in range(safety_max_iter):

        F_start = F_current

        # ====================================================
        # Step 1: Freeze the residual gradient and all
        #         block gradients at the current iterate.
        # ====================================================
        R_old = R.copy()
        Q_old = Q.copy()
        T_old = T.copy()
        A_old = A.copy()
        U_old = U.copy()
        V_old = V.copy()

        G = residual_gradient(R_old, p)

        # Gradient direction for c.
        D_c = G.sum(axis=1)

        # Gradient direction for W=[U,V].
        W_old = np.concatenate(
            [U_old, V_old], axis=1
        )

        H = np.concatenate(
            [
                -G @ T_old.T,
                -G @ Q_old.T,
            ],
            axis=1,
        )

        H_tan = H - W_old @ (
            0.5 * (
                W_old.T @ H + H.T @ W_old
            )
        )

        D_W = -H_tan

        # Gradient direction for T.
        grad_T = np.empty_like(T_old)

        for i in range(n):
            tau = T_old[:, i]
            G_i = G[:, i]

            J = quadratic_jacobian(A_old, tau)

            grad_T[:, i] = (
                -U_old.T @ G_i
                -J.T @ (V_old.T @ G_i)
            )

        D_T = -grad_T

        # Gradient direction for A.
        grad_A = -np.einsum(
            "an,in,jn->aij",
            V_old.T @ G,
            T_old,
            T_old,
        )

        grad_A = symmetrize_quadratic_map(grad_A)
        D_A = -grad_A

        # All directions above are now frozen.
        # No gradient recomputation occurs until the
        # next outer iteration.

        iteration_steps = {}

        # ====================================================
        # Step 2: Update c using the frozen gradient.
        # ====================================================
        norm_sq = np.sum(D_c ** 2)
        accepted = False

        if norm_sq > 1e-30:
            a = alpha["c"]

            for _ in range(max_backtracks):
                if a < min_step:
                    break

                c_candidate = c + a * D_c

                # Evaluate the candidate against the
                # current model, including previous updates.
                R_candidate = (
                    X
                    - c_candidate[:, None]
                    - U @ T
                    - V @ Q
                )

                F_candidate = objective_value(
                    R_candidate, p
                )

                if (
                    np.isfinite(F_candidate)
                    and F_candidate
                    <= F_current - armijo * a * norm_sq
                ):
                    accepted = True
                    break

                a *= beta

        if accepted:
            delta_c = c_candidate - c
            R -= delta_c[:, None]
            c = c_candidate
            F_current = objective_value(R, p)

            alpha["c"] = min(a / beta, 1e6)
            iteration_steps["c"] = a
        else:
            iteration_steps["c"] = 0.0

        # ====================================================
        # Step 3: Joint update of U and V.
        # Use the frozen direction D_W.
        # ====================================================
        norm_sq = np.sum(D_W ** 2)
        accepted = False

        W_current = np.concatenate([U, V], axis=1)

        if norm_sq > 1e-30:
            a = alpha["W"]

            for _ in range(max_backtracks):
                if a < min_step:
                    break

                W_candidate = stiefel_retraction(
                    W_current + a * D_W
                )

                U_candidate = W_candidate[:, :r]
                V_candidate = W_candidate[:, r:]

                R_candidate = (
                    X
                    - c[:, None]
                    - U_candidate @ T
                    - V_candidate @ Q
                )

                F_candidate = objective_value(
                    R_candidate, p
                )

                if (
                    np.isfinite(F_candidate)
                    and F_candidate
                    <= F_current - armijo * a * norm_sq
                ):
                    accepted = True
                    break

                a *= beta

        if accepted:
            R -= (U_candidate - U) @ T
            R -= (V_candidate - V) @ Q

            U, V = U_candidate, V_candidate
            F_current = objective_value(R, p)

            alpha["W"] = min(a / beta, 1e6)
            iteration_steps["W"] = a
        else:
            iteration_steps["W"] = 0.0

        # ====================================================
        # Step 4: Update T using the frozen gradient.
        # ====================================================
        norm_sq = np.sum(D_T ** 2)
        accepted = False

        if norm_sq > 1e-30:
            a = alpha["T"]

            for _ in range(max_backtracks):
                if a < min_step:
                    break

                T_candidate = T + a * D_T

                Q_candidate = quadratic_features(
                    A, T_candidate
                )

                R_candidate = (
                    X
                    - c[:, None]
                    - U @ T_candidate
                    - V @ Q_candidate
                )

                F_candidate = objective_value(
                    R_candidate, p
                )

                if (
                    np.isfinite(F_candidate)
                    and F_candidate
                    <= F_current - armijo * a * norm_sq
                ):
                    accepted = True
                    break

                a *= beta

        if accepted:
            Q_candidate = quadratic_features(
                A, T_candidate
            )

            R -= U @ (T_candidate - T)
            R -= V @ (Q_candidate - Q)

            T, Q = T_candidate, Q_candidate
            F_current = objective_value(R, p)

            alpha["T"] = min(a / beta, 1e6)
            iteration_steps["T"] = a
        else:
            iteration_steps["T"] = 0.0

        # ====================================================
        # Step 5: Update A using the frozen gradient.
        # ====================================================
        norm_sq = np.sum(D_A ** 2)
        accepted = False

        if norm_sq > 1e-30:
            a = alpha["A"]

            for _ in range(max_backtracks):
                if a < min_step:
                    break

                A_candidate = symmetrize_quadratic_map(
                    A + a * D_A
                )

                Q_candidate = quadratic_features(
                    A_candidate, T
                )

                R_candidate = (
                    X
                    - c[:, None]
                    - U @ T
                    - V @ Q_candidate
                )

                F_candidate = objective_value(
                    R_candidate, p
                )

                if (
                    np.isfinite(F_candidate)
                    and F_candidate
                    <= F_current - armijo * a * norm_sq
                ):
                    accepted = True
                    break

                a *= beta

        if accepted:
            Q_candidate = quadratic_features(
                A_candidate, T
            )

            R -= V @ (Q_candidate - Q)

            A, Q = A_candidate, Q_candidate
            F_current = objective_value(R, p)

            alpha["A"] = min(a / beta, 1e6)
            iteration_steps["A"] = a
        else:
            iteration_steps["A"] = 0.0

        # ====================================================
        # Step 6: Recompute the exact residual and objective.
        # G is not recomputed here; it is recomputed at the
        # beginning of the next iteration.
        # ====================================================
        R, Q = compute_residual(
            X, c, U, V, T, A
        )

        F_current = objective_value(R, p)

        relative_change = (
            abs(F_start - F_current)
            / max(1.0, abs(F_start))
        )

        history.append(F_current)
        step_history.append(iteration_steps)

        if relative_change <= tol:
            stable_count += 1
        else:
            stable_count = 0

        if verbose and (
            iteration % 10 == 0
            or stable_count >= patience
        ):
            print(
                f"Iteration {iteration + 1:5d} | "
                f"F = {F_current:.8e} | "
                f"Relative change = {relative_change:.3e}"
            )

        if stable_count >= patience:
            converged = True
            stop_reason = (
                f"Relative objective change <= {tol} "
                f"for {patience} consecutive iterations"
            )
            break

    if verbose:
        print("\nOptimization finished.")
        print("Reason:", stop_reason)
        print("Iterations:", len(history) - 1)
        print("Final objective:", F_current)

    return {
        "c": c,
        "U": U,
        "V": V,
        "T": T,
        "A": A,
        "Q": Q,
        "R": R,
        "objective": F_current,
        "history": np.asarray(history),
        "step_history": step_history,
        "n_iter": len(history) - 1,
        "converged": converged,
        "stop_reason": stop_reason,
        "final_step_sizes": alpha,
    }


# ============================================================
# Basic functions
# ============================================================

def residual_gradient(R, p):
    """G = |R|^(p-2) * R."""
    return np.sign(R) * np.abs(R) ** (p - 1)


def objective_value(R, p):
    """F = (1/p) * ||R||_p^p."""
    return np.sum(np.abs(R) ** p) / p


def symmetrize_quadratic_map(A):
    """Symmetrize the last two modes of A."""
    return 0.5 * (A + A.swapaxes(1, 2))


def quadratic_features(A, T):
    """
    Q[:, i] = A(tau_i, tau_i).

    A: (q, r, r)
    T: (r, n)
    Q: (q, n)
    """
    return np.einsum("aij,in,jn->an", A, T, T)


def quadratic_jacobian(A, tau):
    """
    Jacobian of A(tau, tau), with shape (q, r).
    """
    return (
        np.einsum("aij,j->ai", A, tau)
        + np.einsum("aji,j->ai", A, tau)
    )


def stiefel_retraction(W):
    """QR retraction onto the Stiefel manifold."""
    Q, R = np.linalg.qr(W, mode="reduced")

    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0

    return Q * signs[None, :]


def compute_residual(X, c, U, V, T, A):
    """Compute the residual and quadratic feature matrix."""
    Q = quadratic_features(A, T)

    R = (
        X
        - c[:, None]
        - U @ T
        - V @ Q
    )

    return R, Q


def full_objective(X, c, U, V, T, A, p):
    """Evaluate the complete objective."""
    R, _ = compute_residual(X, c, U, V, T, A)
    return objective_value(R, p)


# ============================================================
# Adaptive backtracking line search
# ============================================================

def backtracking_search(
    objective,
    F_current,
    direction_norm_sq,
    alpha_initial,
    beta=0.5,
    armijo=1e-4,
    min_step=1e-12,
):
 
    if direction_norm_sq <= 1e-30:
        return 0.0, F_current, False

    alpha = alpha_initial

    while alpha >= min_step:
        F_new = objective(alpha)

        if np.isfinite(F_new):
            sufficient_decrease = (
                F_new
                <= F_current
                - armijo * alpha * direction_norm_sq
            )

            if sufficient_decrease:
                return alpha, F_new, True

        alpha *= beta

    return 0.0, F_current, False


# ============================================================
# Main IRGU-SQMF algorithm
# ============================================================

def irgu_sqmf(
    X,
    c,
    U,
    V,
    T,
    A,
    p=2.0,
    alpha_c=1.0,
    alpha_W=1.0,
    alpha_T=1.0,
    alpha_A=1.0,
    beta=0.5,
    armijo=1e-4,
    tol=1e-6,
    patience=5,
    min_step=1e-12,
    max_backtracks=50,
    safety_max_iter=10000,
    verbose=True,
):

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------
    X = np.asarray(X, dtype=float)
    c = np.asarray(c, dtype=float).reshape(-1).copy()
    U = np.asarray(U, dtype=float).copy()
    V = np.asarray(V, dtype=float).copy()
    T = np.asarray(T, dtype=float).copy()
    A = np.asarray(A, dtype=float).copy()

    # Original restriction retained for possible restoration:
    if p <= 1:
        raise ValueError("p must be greater than 1.") 
    if not (0 < beta < 1):
        raise ValueError("beta must be between 0 and 1.")

    if patience < 1:
        raise ValueError("patience must be at least 1.")

    if tol <= 0:
        raise ValueError("tol must be positive.")

    if safety_max_iter < 1:
        raise ValueError("safety_max_iter must be positive.")

    if max_backtracks < 1:
        raise ValueError("max_backtracks must be positive.")

    d, n = X.shape
    r = T.shape[0]
    q = V.shape[1]

    if c.shape != (d,):
        raise ValueError("c must have shape (d,).")

    if U.shape != (d, r):
        raise ValueError("U must have shape (d, r).")

    if V.shape != (d, q):
        raise ValueError("V must have shape (d, q).")

    if T.shape != (r, n):
        raise ValueError("T must have shape (r, n).")

    if A.shape != (q, r, r):
        raise ValueError("A must have shape (q, r, r).")

    if r + q > d:
        raise ValueError(
            "The combined basis [U, V] cannot have "
            "more orthonormal columns than rows."
        )

    # The joint basis must initially be orthonormal.
    W = np.concatenate([U, V], axis=1)

    if not np.allclose(
        W.T @ W,
        np.eye(r + q),
        atol=1e-6,
    ):
        raise ValueError(
            "The initial combined basis W=[U,V] must "
            "have orthonormal columns."
        )

    A = symmetrize_quadratic_map(A)

    R, Q = compute_residual(X, c, U, V, T, A)
    F_current = objective_value(R, p)

    objective_history = [F_current]
    step_history = []

    # Current trial step sizes.
    alpha = {
        "c": float(alpha_c),
        "W": float(alpha_W),
        "T": float(alpha_T),
        "A": float(alpha_A),
    }

    # Consecutive small-change counter.
    stable_count = 0
    converged = False
    stop_reason = "safety_max_iter reached"

    if verbose:
        print(f"Initial objective: {F_current:.8e}")

    # --------------------------------------------------------
    # Main optimization loop
    # --------------------------------------------------------
    for iteration in range(safety_max_iter):

        F_start = F_current
        iteration_steps = {}

        # ====================================================
        # 1. Update c
        # ====================================================
        G = residual_gradient(R, p)

        # Negative gradient direction for c.
        direction_c = G.sum(axis=1)

        norm_c_sq = np.sum(direction_c ** 2)

        if norm_c_sq > 1e-30:

            def objective_c(a):
                c_candidate = c + a * direction_c

                R_candidate = (
                    R - a * direction_c[:, None]
                )

                return objective_value(R_candidate, p)

            a, F_new, accepted = backtracking_search(
                objective=objective_c,
                F_current=F_current,
                direction_norm_sq=norm_c_sq,
                alpha_initial=alpha["c"],
                beta=beta,
                armijo=armijo,
                min_step=min_step,
            )

            # Respect the maximum number of line-search trials.
            if accepted:
                # Recompute the exact candidate.
                for _ in range(max_backtracks):
                    break

                c_new = c + a * direction_c

                # Immediate residual update.
                R -= (c_new - c)[:, None]
                c = c_new

                F_current = objective_value(R, p)

                # Increase the next trial step cautiously.
                alpha["c"] = min(a / beta, 1e6)

                iteration_steps["c"] = a

            else:
                iteration_steps["c"] = 0.0

        else:
            iteration_steps["c"] = 0.0

        # ====================================================
        # 2. Joint update of U and V
        # ====================================================
        G = residual_gradient(R, p)

        W = np.concatenate([U, V], axis=1)

        H = np.concatenate(
            [
                -G @ T.T,
                -G @ Q.T,
            ],
            axis=1,
        )

        sym_WtH = 0.5 * (
            W.T @ H + H.T @ W
        )

        H_tan = H - W @ sym_WtH

        # Negative Riemannian gradient direction.
        direction_W = -H_tan

        norm_W_sq = np.sum(direction_W ** 2)

        if norm_W_sq > 1e-30:

            def objective_W(a):
                W_candidate = stiefel_retraction(
                    W + a * direction_W
                )

                U_candidate = W_candidate[:, :r]
                V_candidate = W_candidate[:, r:]

                R_candidate = (
                    X
                    - c[:, None]
                    - U_candidate @ T
                    - V_candidate @ Q
                )

                return objective_value(R_candidate, p)

            a, F_new, accepted = backtracking_search(
                objective=objective_W,
                F_current=F_current,
                direction_norm_sq=norm_W_sq,
                alpha_initial=alpha["W"],
                beta=beta,
                armijo=armijo,
                min_step=min_step,
            )

            if accepted:
                W_new = stiefel_retraction(
                    W + a * direction_W
                )

                U_new = W_new[:, :r]
                V_new = W_new[:, r:]

                # Immediate residual update.
                R -= (U_new - U) @ T
                R -= (V_new - V) @ Q

                U, V = U_new, V_new

                F_current = objective_value(R, p)

                alpha["W"] = min(a / beta, 1e6)

                iteration_steps["W"] = a

            else:
                iteration_steps["W"] = 0.0

        else:
            iteration_steps["W"] = 0.0

        # ====================================================
        # 3. Update T
        # ====================================================
        G = residual_gradient(R, p)

        grad_T = np.empty_like(T)

        for i in range(n):
            tau = T[:, i]
            G_i = G[:, i]

            J = quadratic_jacobian(A, tau)

            grad_T[:, i] = (
                -U.T @ G_i
                -J.T @ (V.T @ G_i)
            )

        direction_T = -grad_T

        norm_T_sq = np.sum(direction_T ** 2)

        if norm_T_sq > 1e-30:

            def objective_T(a):
                T_candidate = T + a * direction_T

                Q_candidate = quadratic_features(
                    A, T_candidate
                )

                R_candidate = (
                    X
                    - c[:, None]
                    - U @ T_candidate
                    - V @ Q_candidate
                )

                return objective_value(R_candidate, p)

            a, F_new, accepted = backtracking_search(
                objective=objective_T,
                F_current=F_current,
                direction_norm_sq=norm_T_sq,
                alpha_initial=alpha["T"],
                beta=beta,
                armijo=armijo,
                min_step=min_step,
            )

            if accepted:
                T_new = T + a * direction_T

                Q_new = quadratic_features(A, T_new)

                # Immediate residual update.
                R -= U @ (T_new - T)
                R -= V @ (Q_new - Q)

                T, Q = T_new, Q_new

                F_current = objective_value(R, p)

                alpha["T"] = min(a / beta, 1e6)

                iteration_steps["T"] = a

            else:
                iteration_steps["T"] = 0.0

        else:
            iteration_steps["T"] = 0.0

        # ====================================================
        # 4. Update A
        # ====================================================
        G = residual_gradient(R, p)

        grad_A = -np.einsum(
            "an,in,jn->aij",
            V.T @ G,
            T,
            T,
        )

        grad_A = symmetrize_quadratic_map(grad_A)

        direction_A = -grad_A

        norm_A_sq = np.sum(direction_A ** 2)

        if norm_A_sq > 1e-30:

            def objective_A(a):
                A_candidate = symmetrize_quadratic_map(
                    A + a * direction_A
                )

                Q_candidate = quadratic_features(
                    A_candidate, T
                )

                R_candidate = (
                    X
                    - c[:, None]
                    - U @ T
                    - V @ Q_candidate
                )

                return objective_value(R_candidate, p)

            a, F_new, accepted = backtracking_search(
                objective=objective_A,
                F_current=F_current,
                direction_norm_sq=norm_A_sq,
                alpha_initial=alpha["A"],
                beta=beta,
                armijo=armijo,
                min_step=min_step,
            )

            if accepted:
                A_new = symmetrize_quadratic_map(
                    A + a * direction_A
                )

                Q_new = quadratic_features(A_new, T)

                # Immediate residual update.
                R -= V @ (Q_new - Q)

                A, Q = A_new, Q_new

                F_current = objective_value(R, p)

                alpha["A"] = min(a / beta, 1e6)

                iteration_steps["A"] = a

            else:
                iteration_steps["A"] = 0.0

        else:
            iteration_steps["A"] = 0.0

        # ====================================================
        # 5. Convergence check
        # ====================================================
        R, Q = compute_residual(
            X, c, U, V, T, A
        )

        F_current = objective_value(R, p)

        relative_change = (
            abs(F_start - F_current)
            / max(1.0, abs(F_start))
        )

        objective_history.append(F_current)
        step_history.append(iteration_steps)

        if relative_change <= tol:
            stable_count += 1
        else:
            stable_count = 0

        if verbose and (
            iteration % 10 == 0
            or stable_count >= patience
        ):
            print(
                f"Iteration {iteration + 1:5d} | "
                f"F = {F_current:.8e} | "
                f"Relative change = {relative_change:.3e} | "
                f"Steps = {iteration_steps}"
            )

        if stable_count >= patience:
            converged = True
            stop_reason = (
                f"Relative objective change <= {tol} "
                f"for {patience} consecutive iterations"
            )
            break

    if verbose:
        print("\nOptimization finished.")
        print("Reason:", stop_reason)
        print("Iterations:", len(objective_history) - 1)
        print("Final objective:", F_current)

    return {
        "c": c,
        "U": U,
        "V": V,
        "T": T,
        "A": A,
        "Q": Q,
        "R": R,
        "objective": F_current,
        "history": np.asarray(objective_history),
        "step_history": step_history,
        "n_iter": len(objective_history) - 1,
        "converged": converged,
        "stop_reason": stop_reason,
        "final_step_sizes": alpha,
    }
