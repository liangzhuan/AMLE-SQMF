import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import loadmat
from scipy.linalg import svd
from threadpoolctl import threadpool_limits

from alternating_mle_sqmf import alternating_mle_sqmf


P_VALUES = (1.0, 1.25, 1.5, 1.75, 2.0)
PARAMETERS = ("c", "U", "V", "T", "A")


@lru_cache(maxsize=1)
def load_frey(path):
    return np.asarray(loadmat(path)["ff"], dtype=float) / 255.0


@lru_cache(maxsize=1)
def make_cloud(path, p, index, alpha):
    clean = load_frey(path)
    rng = np.random.default_rng(np.random.SeedSequence([2026102632, index]))
    signs = rng.choice([-1.0, 1.0], size=clean.shape)
    noise = signs * rng.gamma(1.0 / p, size=clean.shape) ** (1.0 / p)
    return clean + alpha * noise


def make_problem(noisy, query, n):
    distances = np.sum((noisy - noisy[:, query, None]) ** 2, axis=0)
    distances[query] = np.inf
    ids = np.argsort(distances, kind="stable")[:n]
    X = noisy[:, ids]
    c = X.mean(axis=1)
    Y = X - c[:, None]
    W = svd(Y, full_matrices=False)[0][:, :3]
    U, V = W[:, :2], W[:, 2:3]
    rng = np.random.default_rng(np.random.SeedSequence([2026102633, query]))
    A = rng.normal(scale=0.01, size=(1, 2, 2))
    A = 0.5 * (A + A.swapaxes(1, 2))
    return X, (c, U, V, U.T @ Y, A), ids


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_query(job):
    index, alpha, order, query, n, args = job
    p = P_VALUES[index]
    output = args.output
    tag = f"p{index}_alpha{alpha:g}_q{order:02d}_n{n}"
    start = perf_counter()
    with threadpool_limits(limits=1):
        noisy = make_cloud(args.data, p, index, alpha)
        X, initial, ids = make_problem(noisy, query, n)
        model = alternating_mle_sqmf(
            X, *initial, p0=1.4, p_bounds=(0.5, 3.5),
            max_outer_iter=args.outer_iter, theta_max_iter=args.max_iter,
            tol=args.tol,
        )
    row = dict(
        p_true=p, alpha_true=alpha, query_order=order, face_id=query + 1,
        n=n, p_hat=float(model["p"]), alpha_hat=float(model["alpha"]),
        eta_hat=float(model["eta"]), nll=float(model["nll"]),
        iterations=model["n_iter"], converged=model["converged"],
        seconds=perf_counter() - start,
    )
    np.savez_compressed(
        output / "fits" / f"{tag}.npz",
        p_true=p, alpha_true=alpha, p0=1.4, face_id=query + 1,
        query=noisy[:, query], neighbor_ids=ids,
        p_hat=model["p"], alpha_hat=model["alpha"], eta_hat=model["eta"],
        nll=model["nll"], n_iter=model["n_iter"],
        converged=model["converged"],
        history_iteration=[item["iteration"] for item in model["history"]],
        history_p=[item["p"] for item in model["history"]],
        history_alpha=[item["alpha"] for item in model["history"]],
        history_nll=[item["nll"] for item in model["history"]],
        **{key: model[key] for key in PARAMETERS},
    )
    write_csv(output / "fits" / f"{tag}_history.csv", model["history"])
    return row


def summarize(rows, output):
    summaries = []
    conditions = sorted({(row["p_true"], row["alpha_true"], row["n"])
                         for row in rows})
    print("\nParameter estimates")
    print(f"{'p':>5} {'alpha':>7} {'n':>5} {'Count':>7} "
          f"{'Mean p':>10} {'SD p':>10} {'Mean alpha':>12} {'SD alpha':>12}")
    for p, alpha, n in conditions:
        selected = [row for row in rows if
                    (row["p_true"], row["alpha_true"], row["n"]) == (p, alpha, n)]
        shapes = np.array([row["p_hat"] for row in selected])
        scales = np.array([row["alpha_hat"] for row in selected])
        p_sd = float(shapes.std(ddof=1)) if shapes.size > 1 else float("nan")
        alpha_sd = float(scales.std(ddof=1)) if scales.size > 1 else float("nan")
        summaries.append(dict(
            p_true=p, alpha_true=alpha, n=n, count=shapes.size,
            p_mean=float(shapes.mean()), p_sd=p_sd,
            alpha_mean=float(scales.mean()), alpha_sd=alpha_sd,
            converged_count=sum(row["converged"] for row in selected),
        ))
        print(f"{p:5.2f} {alpha:7.3f} {n:5d} {shapes.size:7d} "
              f"{shapes.mean():10.3f} {p_sd:10.3f} "
              f"{scales.mean():12.3f} {alpha_sd:12.3f}")
    write_csv(output / "summary.csv", summaries)


def argument_parser(output):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).with_name("frey_rawface.mat"))
    parser.add_argument("--outer-iter", type=int, default=360)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / output)
    return parser


def prepare(args):
    clean = load_frey(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Frey Face: D={clean.shape[0]}, images={clean.shape[1]}, p0=1.4")
    return clean


def query_centers(count, total):
    draw = min(total, max(30, count))
    return np.random.default_rng(41).choice(total, draw, replace=False)[:count]


def run_jobs(jobs, args, summary=True):
    (args.output / "fits").mkdir(parents=True, exist_ok=True)
    print(f"Fits: {len(jobs)}")
    print(f"{'p':>5} {'alpha':>7} {'Face':>6} {'n':>5} "
          f"{'p_hat':>10} {'alpha_hat':>12} {'Iterations':>12} {'Converged':>10}")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(fit_query, jobs):
            rows.append(row)
            print(f"{row['p_true']:5.2f} {row['alpha_true']:7.3f} "
                  f"{row['face_id']:6d} {row['n']:5d} "
                  f"{row['p_hat']:10.6f} {row['alpha_hat']:12.6f} "
                  f"{row['iterations']:12d} {str(row['converged']):>10}",
                  flush=True)
    write_csv(args.output / "estimates.csv", rows)
    if summary:
        summarize(rows, args.output)
    print(f"\nResults: {args.output.resolve()}")
