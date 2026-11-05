import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from dimension_mle_sqmf import alternating_mle_sqmf
from dimension_optimization import irgu_sqmf, quadratic_features
from frey_common import write_csv


DIMENSIONS = (30, 60, 90, 120, 150)
SAMPLE_SIZES = (1000, 1500, 2000, 2500, 3000)
PARAMETERS = ("c", "U", "V", "T", "A")
P_TRUE = 1.4
ETA_TRUE = 0.02


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=DIMENSIONS)
    parser.add_argument("--sizes", type=int, nargs="+", default=SAMPLE_SIZES)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--outer-iter", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results_dimension_sample_size")
    return parser


def make_problem(D, n, seed):
    rng = np.random.default_rng(seed)
    W = np.linalg.qr(rng.normal(size=(D, 5)), mode="reduced")[0]
    c = rng.normal(scale=0.4, size=D)
    T = rng.uniform(-1.0, 1.0, size=(2, n))
    A = rng.normal(scale=0.12, size=(3, 2, 2))
    A = 0.5 * (A + A.swapaxes(1, 2))
    clean = c[:, None] + W[:, :2] @ T + W[:, 2:] @ quadratic_features(A, T)
    magnitudes = rng.gamma(1.0 / P_TRUE, size=clean.shape) ** (1.0 / P_TRUE)
    signs = rng.choice(np.array([-1.0, 1.0]), size=clean.shape)
    noise = signs * magnitudes * ETA_TRUE ** (1.0 / P_TRUE)
    c0 = c + rng.normal(scale=0.04, size=D)
    T0 = T + rng.normal(scale=0.04, size=T.shape)
    A0 = A + rng.normal(scale=0.025, size=A.shape)
    A0 = 0.5 * (A0 + A0.swapaxes(1, 2))
    W0 = np.linalg.qr(W + rng.normal(scale=0.02, size=W.shape), mode="reduced")[0]
    return dict(
        X=clean + noise, clean_X=clean, noise=noise,
        c_true=c, U_true=W[:, :2], V_true=W[:, 2:], T_true=T, A_true=A,
        c0=c0, U0=W0[:, :2], V0=W0[:, 2:], T0=T0, A0=A0,
    )


def fit_task(job):
    D, repeat, args = job
    seed = 20000000000 + 1000 * repeat + D
    tag = f"D{D}_r{repeat:02d}"
    rows = []
    with threadpool_limits(limits=1):
        data = make_problem(D, max(args.sizes), seed)
        np.savez_compressed(args.output / "inputs" / f"{tag}.npz", seed=seed, **data)
        for n in args.sizes:
            start = perf_counter()
            initial = (
                data["c0"], data["U0"], data["V0"],
                data["T0"][:, :n], data["A0"],
            )
            model = alternating_mle_sqmf(
                data["X"][:, :n], *initial,
                theta_solver=partial(irgu_sqmf, verbose=False),
                p0=2.0, p_bounds=(0.5, 3.5), max_outer_iter=args.outer_iter,
                theta_max_iter=args.max_iter, tol=args.tol, verbose=False,
            )
            alpha = model["eta"] ** (1.0 / model["p"])
            row = dict(
                D=D, n=n, r=repeat, seed=seed, p_true=P_TRUE,
                eta_true=ETA_TRUE, alpha_true=ETA_TRUE ** (1.0 / P_TRUE),
                p_hat=model["p"], eta_hat=model["eta"], alpha_hat=alpha,
                abs_error_p=abs(model["p"] - P_TRUE), nll=model["nll"],
                reconstruction_rmse=float(np.sqrt(np.mean(
                    (data["X"][:, :n] - model["R"] - data["clean_X"][:, :n]) ** 2,
                ))),
                outer_iterations=model["n_iter"], converged=model["converged"],
                seconds=perf_counter() - start,
            )
            rows.append(row)
            name = f"{tag}_n{n}"
            np.savez_compressed(
                args.output / "fits" / f"{name}.npz",
                D=D, n=n, r=repeat, seed=seed, p0=2.0,
                p_true=P_TRUE, eta_true=ETA_TRUE,
                alpha_true=ETA_TRUE ** (1.0 / P_TRUE),
                p_hat=model["p"], eta_hat=model["eta"], alpha_hat=alpha,
                R=model["R"], Q=model["Q"], nll=model["nll"],
                n_iter=model["n_iter"], converged=model["converged"],
                **{key: model[key] for key in PARAMETERS},
            )
            write_csv(args.output / "fits" / f"{name}_history.csv", model["history"])
            print(f"D={D}, n={n}, repeat={repeat + 1}, p_hat={model['p']:.6f}, "
                  f"alpha_hat={alpha:.6f}, eta_hat={model['eta']:.6f}, "
                  f"iterations={model['n_iter']}, converged={model['converged']}",
                  flush=True)
    return rows


def summarize(raw):
    rows = []
    for (D, n), group in raw.groupby(["D", "n"], sort=True):
        estimates = group.p_hat.to_numpy()
        error = estimates - group.p_true.to_numpy()
        bias = float(error.mean())
        rows.append(dict(
            D=int(D), n=int(n), Dn=int(D * n), repetitions=len(group),
            mean_p=float(estimates.mean()), bias=bias, abs_bias=abs(bias),
            sd=float(estimates.std(ddof=1)), mse=float(np.mean(error ** 2)),
        ))
    return pd.DataFrame(rows)


def run_jobs(args):
    for name in ("inputs", "fits"):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    jobs = [(D, repeat, args) for D in args.dimensions for repeat in range(args.repeats)]
    print(f"Tasks={len(jobs)}, fits={len(jobs) * len(args.sizes)}, "
          f"p_true={P_TRUE:g}, alpha_true={ETA_TRUE ** (1.0 / P_TRUE):g}, p0=2")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fitted in pool.map(fit_task, jobs):
            rows.extend(fitted)
            write_csv(args.output / "estimates.csv", rows)
    summary = summarize(pd.DataFrame(rows))
    summary.to_csv(args.output / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"Results: {args.output.resolve()}")
