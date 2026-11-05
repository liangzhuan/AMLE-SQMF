import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import svd
from scipy.optimize import least_squares
from scipy.special import gammaln
from threadpoolctl import threadpool_limits

from irgu_sqmf import irgu_sqmf
from ordinary_gd_sqmf import ordinary_gd_sqmf
from sqmf_common import quadratic_features


P_VALUES = (1.25, 1.5, 1.75, 2.0)
PARAMETERS = ("c", "U", "V", "T", "A")


def make_sphere(n=1000, seed=42):
    X = np.random.default_rng(seed).standard_normal((3, n))
    return X / np.linalg.norm(X, axis=0)


def make_noise(p, index, shape=(3, 1000), sigma=0.05, seed=42):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    alpha = sigma * np.exp((gammaln(1.0 / p) - gammaln(3.0 / p)) / 2.0)
    noise = alpha * rng.gamma(1.0 / p, size=shape) ** (1.0 / p)
    return noise * (2 * rng.integers(0, 2, size=shape) - 1)


def make_problem(noisy, query, n=200):
    distances = np.sum((noisy - noisy[:, query, None]) ** 2, axis=0)
    distances[query] = np.inf
    ids = np.argsort(distances, kind="stable")[:n]
    X = noisy[:, ids]
    c = X.mean(axis=1)
    Y = X - c[:, None]
    W = svd(Y, full_matrices=False)[0]
    U, V = W[:, :2], W[:, 2:3]
    initial = (c, U, V, U.T @ Y, np.zeros((1, 2, 2)))
    return X, initial, ids


def euclidean_projection(x, model):
    c, U, V, T, A = (model[key] for key in PARAMETERS)

    def residual(tau):
        return c + U @ tau + V @ quadratic_features(A, tau[:, None])[:, 0] - x

    def jacobian(tau):
        return U + V @ np.einsum("sij,j->si", A + A.swapaxes(1, 2), tau)

    fitted = c[:, None] + U @ T + V @ quadratic_features(A, T)
    nearest = np.argsort(np.sum((fitted - x[:, None]) ** 2, axis=0))[:5]
    starts = [U.T @ (x - c), np.zeros(2), *T[:, nearest].T]
    solutions = [
        least_squares(
            residual, start, jac=jacobian, method="lm",
            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000,
        )
        for start in starts
    ]
    solution = min(solutions, key=lambda item: np.sum(item.fun ** 2))
    return x + solution.fun, solution.x


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_speed(clouds, query, max_iter, output):
    rows = []
    methods = (("GD", ordinary_gd_sqmf), ("IRGU", irgu_sqmf))
    print("\nOptimization time")
    print(f"{'p':>5} {'Method':>8} {'Time (s)':>12} {'Iterations':>12}")
    with threadpool_limits(limits=1):
        for index, (p, noisy) in enumerate(zip(P_VALUES, clouds)):
            X, initial, ids = make_problem(noisy, query)
            for name, solver in methods[::1 if index % 2 == 0 else -1]:
                start = perf_counter()
                model = solver(
                    X, *(value.copy() for value in initial), p=p,
                    tol=1e-9, patience=10, max_iter=max_iter,
                )
                seconds = perf_counter() - start
                print(f"{p:5.2f} {name:>8} {seconds:12.4f} {model['n_iter']:12d}")
                rows.append(dict(
                    p=p, method=name, seconds=seconds,
                    iterations=model["n_iter"], converged=model["converged"],
                ))
                np.savez_compressed(
                    output / f"timing_p{index}_{name}.npz",
                    history=model["history"], neighbor_ids=ids,
                    **{key: model[key] for key in PARAMETERS},
                )
                history = [dict(iteration=k, objective=float(value))
                           for k, value in enumerate(model["history"])]
                write_csv(output / f"history_p{index}_{name}.csv", history)
    write_csv(output / "timing.csv", rows)


def fit_query(job):
    index, order, query, noisy, max_iter, output = job
    p = P_VALUES[index]
    with threadpool_limits(limits=1):
        X, initial, ids = make_problem(noisy, query)
        model = irgu_sqmf(
            X, *initial, p=p, tol=1e-9, patience=10, max_iter=max_iter,
        )
        projected, tau = euclidean_projection(noisy[:, query], model)
    delta = float(np.linalg.norm(noisy[:, query] - projected))
    np.savez_compressed(
        output / "fits" / f"p{index}_q{order:02d}.npz",
        p=p, query_id=query, query=noisy[:, query], projected_query=projected,
        projected_tau=tau, delta=delta, neighbor_ids=ids,
        history=model["history"], n_iter=model["n_iter"],
        converged=model["converged"],
        **{key: model[key] for key in PARAMETERS},
    )
    row = dict(
        p=p, query_order=order, query_id=int(query), delta=delta,
        iterations=model["n_iter"], converged=model["converged"],
    )
    example = {key: model[key] for key in ("c", "U", "V", "A")} if order == 0 else None
    return row, example


def fit_spheres(clouds, centers, max_iter, workers, output):
    (output / "fits").mkdir(exist_ok=True)
    jobs = [
        (index, order, int(query), noisy, max_iter, output)
        for index, noisy in enumerate(clouds)
        for order, query in enumerate(centers)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fit_query, jobs))
    rows = [row for row, example in results]
    write_csv(output / "delta.csv", rows)
    summaries = []
    print("\nCenter-to-surface distance")
    print(f"{'p':>5} {'Mean delta':>14} {'Sample std':>14}")
    for p in P_VALUES:
        values = np.array([row["delta"] for row in rows if row["p"] == p])
        summaries.append(dict(
            p=p, count=values.size, delta_mean=float(values.mean()),
            delta_std=float(values.std(ddof=1)),
        ))
        print(f"{p:5.2f} {values.mean():14.6f} {values.std(ddof=1):14.6f}")
    write_csv(output / "delta_summary.csv", summaries)
    np.set_printoptions(precision=6, suppress=True)
    for row, example in results:
        if example is not None:
            print(f"\np = {row['p']:g}, query = {row['query_id']}")
            for key, value in example.items():
                print(f"{key} =\n{value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    clean = make_sphere()
    clouds = [clean + make_noise(p, index) for index, p in enumerate(P_VALUES)]
    centers = np.random.default_rng(2026102631).permutation(clean.shape[1])[:args.queries]
    for index, noisy in enumerate(clouds):
        np.savez_compressed(args.output / f"cloud_p{index}.npz", clean=clean, noisy=noisy)
    compare_speed(clouds, int(centers[0]), args.max_iter, args.output)
    fit_spheres(clouds, centers, args.max_iter, args.workers, args.output)
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
