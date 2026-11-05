import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import loadmat
from scipy.linalg import svd
from scipy.special import gammaln
from threadpoolctl import threadpool_limits

from alternating_mle_sqmf import alternating_mle_sqmf
from frey_common import PARAMETERS, query_centers, write_csv
from irgu_sqmf import irgu_sqmf
from manifold_baselines import project_baseline
from sqmf_common import (
    backtracking, compute_residual, latent_direction, objective_value,
    quadratic_features, residual_gradient,
)


NOISES = ("ggd_p1_25", "ggd_p1_5", "ggd_p1_75", "logistic", "student_t10")
SHAPES = (1.25, 1.5, 1.75, np.nan, np.nan)
NOISE_SEEDS = (125, 0, 175, 8, 10)
METHODS = ("AMLE-SQMF", "SQMF-p1", "SQMF-p2", "LPCA", "SPH", "Q-LS",
           "MFIT", "LOG-KDE")


@lru_cache(maxsize=1)
def load_isomap(path):
    return np.asarray(loadmat(path)["images"], dtype=float)


def noise_parameters(index, sigma):
    p = SHAPES[index]
    if np.isfinite(p):
        alpha = sigma * np.exp((gammaln(1.0 / p) - gammaln(3.0 / p)) / 2.0)
        return p, float(alpha), float(alpha ** p)
    return np.nan, np.nan, np.nan


@lru_cache(maxsize=1)
def make_cloud(path, index, sigma):
    clean = load_isomap(path)
    rng = np.random.default_rng(np.random.SeedSequence([2026103101, NOISE_SEEDS[index]]))
    if index < 3:
        p, alpha, eta = noise_parameters(index, sigma)
        noise = alpha * rng.gamma(1.0 / p, size=clean.T.shape) ** (1.0 / p)
        noise *= 2 * rng.integers(0, 2, size=clean.T.shape) - 1
    elif index == 3:
        noise = rng.logistic(scale=sigma * np.sqrt(3.0) / np.pi, size=clean.T.shape)
    else:
        noise = sigma * np.sqrt(8.0 / 10.0) * rng.standard_t(10, size=clean.T.shape)
    return clean + noise.T


def make_problem(noisy, query, n):
    distances = np.sum((noisy - noisy[:, query, None]) ** 2, axis=0)
    distances[query] = np.inf
    ids = np.argsort(distances, kind="stable")[:n]
    X = noisy[:, ids]
    c = X.mean(axis=1)
    Y = X - c[:, None]
    W = svd(Y, full_matrices=False)[0][:, :6]
    U, V = W[:, :3], W[:, 3:6]
    rng = np.random.default_rng(np.random.SeedSequence([2026102633, query]))
    A = rng.normal(scale=0.01, size=(3, 3, 3))
    A = 0.5 * (A + A.swapaxes(1, 2))
    return X, (c, U, V, U.T @ Y, A), ids


def project_query(x, model, p, max_iter=180):
    c, U, V, T, A = (model[key] for key in PARAMETERS)
    tau = (U.T @ (x - c))[:, None]
    R, Q = compute_residual(x[:, None], c, U, V, tau, A)
    F = objective_value(R, p)
    history = [float(F)]
    converged = False
    step = 0.15
    for iteration in range(max_iter):
        direction = latent_direction(residual_gradient(R, p), U, V, tau, A)

        def objective(a):
            residual, _ = compute_residual(x[:, None], c, U, V, tau + a * direction, A)
            return objective_value(residual, p)

        a = backtracking(objective, F, direction, step, 0.5, 1e-4, 1e-12)
        if not a:
            converged = bool(np.linalg.norm(direction) <= 1e-8)
            break
        change = a * direction
        tau += change
        R, Q = compute_residual(x[:, None], c, U, V, tau, A)
        F = objective_value(R, p)
        history.append(float(F))
        step = min(a / 0.5, 1e6)
        if np.linalg.norm(change) <= 1e-8:
            converged = True
            break
    prediction = c + U @ tau[:, 0] + V @ quadratic_features(A, tau)[:, 0]
    return prediction, tau[:, 0], dict(
        iterations=len(history) - 1, converged=converged, history=history,
    )


def fit_query(job):
    index, order, query, args = job
    start = perf_counter()
    noise = NOISES[index]
    tag = f"{noise}_q{order:02d}"
    saved = {}
    rows = []
    with threadpool_limits(limits=1):
        noisy = make_cloud(args.data, index, args.sigma)
        X, initial, ids = make_problem(noisy, query, args.n)
        y = noisy[:, query]
        model = alternating_mle_sqmf(
            X, *initial, p0=1.4, p_bounds=(1.0, 2.0),
            max_outer_iter=args.outer_iter, theta_max_iter=args.max_iter,
            tol=args.tol,
        )
        p_true, alpha_true, eta_true = noise_parameters(index, args.sigma)
        estimate = dict(
            noise=noise, query_order=order, face_id=query + 1, n=args.n,
            d=3, s=3, sigma=args.sigma, p_true=p_true,
            alpha_true=alpha_true, eta_true=eta_true,
            p_hat=float(model["p"]), alpha_hat=float(model["alpha"]),
            eta_hat=float(model["eta"]), nll=float(model["nll"]),
            iterations=model["n_iter"], converged=model["converged"],
        )
        clean_query = load_isomap(args.data)[:, query]
        input_mse = float(np.mean((y - clean_query) ** 2))
        for method in METHODS:
            method_start = perf_counter()
            if method == "AMLE-SQMF":
                fitted = model
                p_fit = float(model["p"])
            elif method in ("SQMF-p1", "SQMF-p2"):
                p_fit = 1.0 if method == "SQMF-p1" else 2.0
                fitted = irgu_sqmf(
                    X, *initial, p=p_fit, max_iter=args.outer_iter * args.max_iter,
                    tol=args.tol,
                )
            else:
                fitted = None
                p_fit = np.nan
            if fitted is not None:
                prediction, tau, diagnostic = project_query(y, fitted, p_fit)
                iterations = fitted["n_iter"]
                converged = fitted["converged"]
                saved.update({f"{method}_{key}": fitted[key] for key in PARAMETERS})
                saved[f"{method}_tau"] = tau
                saved[f"{method}_projection_history"] = diagnostic["history"]
                if method != "AMLE-SQMF":
                    saved[f"{method}_history"] = fitted["history"]
            else:
                prediction, diagnostic = project_baseline(method, X, y, d=3)
                iterations = diagnostic["iterations"]
                converged = diagnostic["converged"]
            saved[f"{method}_prediction"] = prediction
            saved[f"{method}_converged"] = converged
            saved[f"{method}_iterations"] = iterations
            saved[f"{method}_projection_converged"] = diagnostic["converged"]
            saved[f"{method}_projection_iterations"] = diagnostic["iterations"]
            seconds = perf_counter() - method_start
            if method == "AMLE-SQMF":
                seconds = perf_counter() - start
            rows.append(dict(
                noise=noise, query_order=order, face_id=query + 1, n=args.n,
                d=3, sigma=args.sigma, method=method, p_fit=p_fit,
                mse=float(np.mean((prediction - clean_query) ** 2)),
                input_mse=input_mse, iterations=iterations, converged=converged,
                projection_iterations=diagnostic["iterations"],
                projection_converged=diagnostic["converged"], seconds=seconds,
            ))
    np.savez_compressed(
        args.output / "fits" / f"{tag}.npz", query_order=order, face_id=query + 1,
        neighbor_ids=ids, clean_query=clean_query, noisy_query=y,
        p_hat=model["p"], alpha_hat=model["alpha"], eta_hat=model["eta"],
        history_iteration=[item["iteration"] for item in model["history"]],
        history_p=[item["p"] for item in model["history"]],
        history_alpha=[item["alpha"] for item in model["history"]],
        history_eta=[item["eta"] for item in model["history"]],
        history_nll=[item["nll"] for item in model["history"]], **saved,
    )
    write_csv(args.output / "fits" / f"{tag}_history.csv", model["history"])
    return estimate, rows


def summarize(estimates, rows, args):
    summaries = []
    parameters = []
    print("\nParameter estimates")
    print(f"{'Noise':>13} {'Count':>7} {'Mean p':>10} {'SD p':>10} "
          f"{'Mean alpha':>12} {'SD alpha':>12} {'Mean eta':>12} {'SD eta':>12}")
    for noise in args.noises:
        selected = [row for row in estimates if row["noise"] == noise]
        row = dict(noise=noise, count=len(selected))
        for name in ("p", "alpha", "eta"):
            values = np.array([item[f"{name}_hat"] for item in selected])
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        parameters.append(row)
        print(f"{noise:>13} {row['count']:7d} {row['p_mean']:10.6f} {row['p_sd']:10.6f} "
              f"{row['alpha_mean']:12.6f} {row['alpha_sd']:12.6f} "
              f"{row['eta_mean']:12.6f} {row['eta_sd']:12.6f}")
    print("\nReconstruction MSE")
    print(f"{'Noise':>13} {'Method':>12} {'Count':>7} {'Mean MSE':>14} {'SD MSE':>14}")
    for noise in args.noises:
        for method in METHODS:
            selected = [row for row in rows if (row["noise"], row["method"]) == (noise, method)]
            values = np.array([row["mse"] for row in selected])
            sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            summaries.append(dict(
                noise=noise, method=method, count=len(values),
                mse_mean=float(values.mean()), mse_sd=sd,
                converged_count=sum(row["converged"] for row in selected),
            ))
            print(f"{noise:>13} {method:>12} {len(values):7d} {values.mean():14.8f} {sd:14.8f}")
    write_csv(args.output / "parameter_summary.csv", parameters)
    write_csv(args.output / "summary.csv", summaries)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("isomap.mat"))
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--noises", nargs="+", choices=NOISES, default=NOISES)
    parser.add_argument("--outer-iter", type=int, default=360)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results_isomap_reconstruction")
    return parser


def prepare(args):
    clean = load_isomap(args.data)
    args.noises = list(dict.fromkeys(args.noises))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fits").mkdir(exist_ok=True)
    print(f"Isomap Face: D={clean.shape[0]}, images={clean.shape[1]}, d=3, s=3, p0=1.4")
    print(f"Queries={args.queries}, n={args.n}, sigma={args.sigma:g}")
    centers = query_centers(args.queries, clean.shape[1])
    write_csv(args.output / "queries.csv", [
        dict(query_order=i, face_id=int(query) + 1)
        for i, query in enumerate(centers)
    ])
    return centers


def run_jobs(centers, args):
    jobs = [(NOISES.index(noise), order, int(query), args)
            for noise in args.noises for order, query in enumerate(centers)]
    print(f"Neighborhoods: {len(jobs)}, reconstructions: {len(jobs) * len(METHODS)}")
    estimates, rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for estimate, fitted_rows in pool.map(fit_query, jobs):
            estimates.append(estimate)
            rows.extend(fitted_rows)
            print(f"\n{estimate['noise']}, face={estimate['face_id']}, "
                  f"p_hat={estimate['p_hat']:.6f}, alpha_hat={estimate['alpha_hat']:.6f}, "
                  f"eta_hat={estimate['eta_hat']:.6f}, "
                  f"iterations={estimate['iterations']}, converged={estimate['converged']}",
                  flush=True)
            for row in fitted_rows:
                print(f"{row['method']:>12} MSE={row['mse']:.8f}, "
                      f"iterations={row['iterations']}, converged={row['converged']}",
                      flush=True)
            write_csv(args.output / "estimates.csv", estimates)
            write_csv(args.output / "mse.csv", rows)
    summarize(estimates, rows, args)
    print(f"\nResults: {args.output.resolve()}")
