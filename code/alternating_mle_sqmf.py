import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp

from irgu_sqmf import irgu_sqmf
from sqmf_common import compute_residual


def update_eta(R, p):
    return float(max(p * np.mean(np.abs(R) ** p), np.finfo(float).tiny))


def profile_nll_per_entry(p, R):
    values = np.abs(R).ravel()
    log_moment = logsumexp(p * np.log(values[values > 0])) - np.log(values.size)
    log_eta = max(np.log(p) + log_moment, np.log(np.finfo(float).tiny))
    return float(np.log(2.0) + gammaln(1.0 + 1.0 / p) + (log_eta + 1.0) / p)


def update_p(R, p_bounds, xatol=1e-6):
    result = minimize_scalar(
        lambda p: profile_nll_per_entry(p, R),
        bounds=p_bounds,
        method="bounded",
        options={"xatol": xatol},
    )
    candidates = (p_bounds[0], float(result.x), p_bounds[1])
    return float(min(candidates, key=lambda p: profile_nll_per_entry(p, R)))


def negative_log_likelihood(R, p, eta):
    return float(
        R.size * (np.log(2.0) + gammaln(1.0 / p) - np.log(p))
        + R.size * np.log(eta) / p
        + np.sum(np.abs(R) ** p) / eta
    )


def alternating_mle_sqmf(
    X,
    c0,
    U0,
    V0,
    T0,
    A0,
    theta_solver=None,
    p_bounds=(0.5, 3.5),
    p0=1.4,
    max_outer_iter=50,
    theta_max_iter=100,
    tol=1e-8,
):
    """Algorithm 1: alternating quadratic fitting and profile-likelihood updates."""
    if theta_solver is None:
        theta_solver = irgu_sqmf
    X = np.asarray(X, dtype=float)
    parameters = dict(zip(("c", "U", "V", "T", "A"),
                          (np.array(v, dtype=float, copy=True)
                           for v in (c0, U0, V0, T0, A0))))
    p = float(p0)
    R, Q = compute_residual(X, **parameters)
    eta = update_eta(R, p)
    nll = negative_log_likelihood(R, p, eta)
    history = [{"iteration": 0, "p": p, "eta": eta,
                "alpha": eta ** (1.0 / p), "nll": nll}]
    converged = False

    for iteration in range(1, max_outer_iter + 1):
        result = theta_solver(X, **parameters, p=p, max_iter=theta_max_iter)
        parameters = {name: result[name] for name in parameters}
        R, Q = compute_residual(X, **parameters)
        p = update_p(R, p_bounds)
        eta = update_eta(R, p)
        previous_nll = nll
        nll = negative_log_likelihood(R, p, eta)
        history.append({"iteration": iteration, "p": p, "eta": eta,
                        "alpha": eta ** (1.0 / p), "nll": nll})
        if abs(nll - previous_nll) / max(1.0, abs(previous_nll)) <= tol:
            converged = True
            break

    return {
        **parameters, "R": R, "Q": Q, "p": p, "eta": eta,
        "alpha": eta ** (1.0 / p), "nll": nll, "history": history,
        "n_iter": len(history) - 1, "converged": converged,
    }
