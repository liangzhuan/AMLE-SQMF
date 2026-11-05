#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alternating maximum-likelihood estimation for SQMF with generalized-Gaussian
residuals.

Assumed scalar residual density:
    f(e | p, eta) = p / (2 Gamma(1/p) eta**(1/p))
                    * exp(-abs(e)**p / eta)

Thus eta is the powered scale in exp(-|e|^p / eta), not the scale b in
exp(-(|e|/b)**p). In particular, E|e|^p = eta/p.

Expected theta optimizer interface:
    result = theta_solver(X, c, U, V, T, A, p=p, max_iter=theta_max_iter)

The theta optimizer must return a dictionary containing c, U, V, T, A, Q,
and R. By default, the script imports irgu_sqmf from sqmf_optimization.py.
"""

from __future__ import annotations

import argparse
import importlib
import time
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln


def quadratic_features(A: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Q[:, i] = A(T[:, i], T[:, i])."""
    return np.einsum("aij,in,jn->an", A, T, T)


def compute_residual(
    X: np.ndarray,
    c: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    T: np.ndarray,
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Q = quadratic_features(A, T)
    R = X - c[:, None] - U @ T - V @ Q
    return R, Q


def update_eta(R: np.ndarray, p: float, eps: float = 1e-12) -> float:
    """
    MLE for eta when the residual entries are independent and identically
    distributed under the density stated in this file:
        eta_hat = p * mean(|R|**p).

    The lower bound eps also applies in profile_nll_per_entry.
    """
    if p <= 0:
        raise ValueError("p must be positive.")
    eta = float(p * np.mean(np.abs(R) ** p))
    return max(eta, eps)


def profile_nll_per_entry(p: float, R: np.ndarray, eps: float = 1e-12) -> float:
    """
    Profile negative log-likelihood per residual entry, with eta optimized
    analytically for each candidate p.

    At eta_hat(p) = p * mean(|R|**p), the objective is:
        log(2 Gamma(1/p) / p) + log(eta_hat(p))/p + 1/p.

    If the lower bound eta >= eps is active, evaluate the full likelihood
    at eta = eps instead; the final term is then mean(|R|**p)/eps.
    """
    if not np.isfinite(p) or p <= 0:
        return np.inf

    abs_R = np.abs(R)
    # Stable log(mean(abs(R)**p)); logsumexp-style computation avoids overflow.
    flat = abs_R.reshape(-1)
    if flat.size == 0 or not np.all(np.isfinite(flat)):
        return np.inf

    log_flat = np.full_like(flat, -np.inf, dtype=float)
    positive = flat > 0
    log_flat[positive] = np.log(flat[positive])
    z = p * log_flat
    zmax = np.max(z)

    if np.isneginf(zmax):
        log_moment = -np.inf
    else:
        log_moment = zmax + np.log(np.mean(np.exp(z - zmax)))

    # Apply the same scale floor as update_eta, rather than flooring the
    # moment. Retain the actual residual term when that floor is active.
    log_eta = max(float(np.log(p) + log_moment), np.log(eps))

    value = (
        np.log(2.0)
        + gammaln(1.0 / p)
        - np.log(p)
        + log_eta / p
        + np.exp(log_moment - log_eta)
    )
    return float(value) if np.isfinite(value) else np.inf


def update_p(
    R: np.ndarray,
    p_bounds: tuple[float, float],
    xatol: float = 1e-5,
) -> float:
    """Bounded one-dimensional minimization of the profile negative log-likelihood."""
    p_min, p_max = map(float, p_bounds)
    if not (0.0 < p_min < p_max):
        raise ValueError("Require 0 < p_min < p_max.")

    result = minimize_scalar(
        lambda p: profile_nll_per_entry(float(p), R),
        bounds=(p_min, p_max),
        method="bounded",
        options={"xatol": xatol},
    )

    # Also check the endpoints: bounded minimization does not always return one.
    candidates = [
        (p_min, profile_nll_per_entry(p_min, R)),
        (float(result.x), float(result.fun)),
        (p_max, profile_nll_per_entry(p_max, R)),
    ]
    best_p, _ = min(candidates, key=lambda item: item[1])
    return float(best_p)


def negative_log_likelihood(R: np.ndarray, p: float, eta: float) -> float:
    """Full negative log-likelihood, up to no omitted parameter-dependent terms."""
    if eta <= 0 or p <= 0:
        return np.inf
    m = R.size
    return float(
        m * (np.log(2.0) + gammaln(1.0 / p) - np.log(p))
        + (m / p) * np.log(eta)
        + np.sum(np.abs(R) ** p) / eta
    )


def alternating_mle_sqmf(
    X: np.ndarray,
    c0: np.ndarray,
    U0: np.ndarray,
    V0: np.ndarray,
    T0: np.ndarray,
    A0: np.ndarray,
    theta_solver: Callable[..., dict[str, Any]],
    p_bounds: tuple[float, float] = (0.1, 4.0),
    p0: float = 2.0,
    max_outer_iter: int = 50,
    theta_max_iter: int = 100,
    tol: float = 1e-5,
    verbose: bool = True,
) -> dict[str, Any]:
    
    X = np.asarray(X, dtype=float)
    c, U, V, T, A = (
        np.asarray(c0, dtype=float).copy(),
        np.asarray(U0, dtype=float).copy(),
        np.asarray(V0, dtype=float).copy(),
        np.asarray(T0, dtype=float).copy(),
        np.asarray(A0, dtype=float).copy(),
    )

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional (D, n) array.")
    if not (0.0 < p_bounds[0] < p_bounds[1]):
        raise ValueError("Require 0 < p_min < p_max.")
    if not (p_bounds[0] <= p0 <= p_bounds[1]):
        raise ValueError("p0 must lie in p_bounds.")
    if max_outer_iter < 1:
        raise ValueError("max_outer_iter must be at least 1.")

    p = float(p0)
    R, Q = compute_residual(X, c, U, V, T, A)
    eta = update_eta(R, p)
    nll = negative_log_likelihood(R, p, eta)

    history: list[dict[str, float]] = [{
        "iteration": 0,
        "p": p,
        "eta": eta,
        "nll": nll,
        "residual_p_mean": float(np.mean(np.abs(R) ** p)),
    }]

    converged = False
    stop_reason = "maximum outer iterations reached"
    start_time = time.perf_counter()

    if verbose:
        print("Alternating MLE for SQMF")
        print(f"Initial: NLL={nll:.6e}, p={p:.5f}, eta={eta:.6e}")

    for k in range(1, max_outer_iter + 1):
        # Step 1: optimize Theta at fixed p.
        theta_result = theta_solver(
            X, c, U, V, T, A,
            p=p,
            safety_max_iter=theta_max_iter,
        )
        c = np.asarray(theta_result["c"], dtype=float).copy()
        U = np.asarray(theta_result["U"], dtype=float).copy()
        V = np.asarray(theta_result["V"], dtype=float).copy()
        T = np.asarray(theta_result["T"], dtype=float).copy()
        A = np.asarray(theta_result["A"], dtype=float).copy()

        R, Q = compute_residual(X, c, U, V, T, A)

        # Step 2: update eta at the current p.
        eta = update_eta(R, p)

        # Step 3: optimize p using the profile likelihood.
        p_new = update_p(R, p_bounds)

        # Recompute eta at p_new so the returned pair is a profile-MLE pair.
        eta_new = update_eta(R, p_new)
        nll_new = negative_log_likelihood(R, p_new, eta_new)

        p, eta, nll = p_new, eta_new, nll_new

        history.append({
            "iteration": k,
            "p": float(p),
            "eta": float(eta),
            "nll": float(nll),
            "residual_p_mean": float(np.mean(np.abs(R) ** p)),
        })

        previous_nll = history[-2]["nll"]
        relative_change = abs(previous_nll - nll) / max(1.0, abs(previous_nll))

        if verbose:
            print(
                f"Iter {k:03d}: NLL={nll:.6e}, p={p:.6f}, "
                f"eta={eta:.6e}, rel_change={relative_change:.3e}"
            )

        if relative_change <= tol:
            converged = True
            stop_reason = f"relative NLL change <= {tol:g}"
            break

    elapsed = time.perf_counter() - start_time
    R, Q = compute_residual(X, c, U, V, T, A)
    final_eta = update_eta(R, p)
    final_nll = negative_log_likelihood(R, p, final_eta)

    if verbose:
        print(
            f"Finished: NLL={final_nll:.6e}, p={p:.6f}, "
            f"eta={final_eta:.6e}, time={elapsed:.2f}s, "
            f"converged={converged}"
        )

    return {
        "c": c,
        "U": U,
        "V": V,
        "T": T,
        "A": A,
        "Q": Q,
        "R": R,
        "p": float(p),
        "eta": float(final_eta),
        "nll": float(final_nll),
        "history": history,
        "n_iter": len(history) - 1,
        "converged": converged,
        "stop_reason": stop_reason,
        "elapsed_time": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="sqmf_optimization",
                        help="Python module containing irgu_sqmf.")
    parser.add_argument("--optimizer", default="irgu_sqmf",
                        help="Theta optimizer function name.")
    parser.add_argument("--p-min", type=float, default=0.2)
    parser.add_argument("--p-max", type=float, default=4.0)
    parser.add_argument("--p0", type=float, default=2.0)
    parser.add_argument("--outer-iter", type=int, default=50)
    parser.add_argument("--theta-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    module = importlib.import_module(args.module)
    theta_solver = getattr(module, args.optimizer)

    raise SystemExit(
        "Import alternating_mle_sqmf from this file and call it with your data "
        "and initial parameters. Example usage is documented in the file header."
    )


if __name__ == "__main__":
    main()
