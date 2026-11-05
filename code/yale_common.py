import argparse
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.linalg import svd
from threadpoolctl import threadpool_limits

from alternating_mle_sqmf import alternating_mle_sqmf
from frey_common import PARAMETERS, write_csv
from sqmf_common import quadratic_features


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).with_name("yale_faces.mat"))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--outer-iter", type=int, default=360)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results_yale_interpolation")
    return parser


def make_problem(X, seed):
    c = X.mean(axis=1)
    Y = X - c[:, None]
    W = svd(Y, full_matrices=False)[0][:, :3]
    U, V = W[:, :2], W[:, 2:3]
    A = np.random.default_rng(seed).normal(scale=0.01, size=(1, 2, 2))
    A = 0.5 * (A + A.swapaxes(1, 2))
    return c, U, V, U.T @ Y, A


def add_grid(fig, images, image_shape, bounds):
    left, bottom, width, height = bounds
    layout = fig.add_gridspec(10, 10, left=left, right=left + width,
                             bottom=bottom, top=bottom + height,
                             hspace=0.025, wspace=0.025)
    for index in range(100):
        ax = fig.add_subplot(layout[index // 10, index % 10])
        ax.imshow(images[:, index].reshape(image_shape, order="F"),
                  cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_axis_off()


def add_coordinates(fig, T, azimuth, bounds, color_bounds):
    ax = fig.add_axes(bounds)
    limit = float(np.max(np.abs(azimuth)))
    points = ax.scatter(T[0], T[1], c=azimuth, cmap="coolwarm",
                        vmin=-limit, vmax=limit, s=12,
                        edgecolors="#333333", linewidths=0.25, zorder=3)
    ax.set_xlabel(r"$\tau_1$", labelpad=1)
    ax.set_ylabel(r"$\tau_2$", labelpad=1)
    ax.margins(0.12)
    ax.tick_params(length=2, width=0.5, pad=1)
    ax.grid(color="#e5e5e5", linewidth=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.55)
    bar = fig.colorbar(points, cax=fig.add_axes(color_bounds),
                       orientation="horizontal", ticks=[-limit, 0, limit])
    bar.ax.tick_params(length=1.5, width=0.45, pad=1)
    bar.outline.set_linewidth(0.45)
    bar.set_label("Lighting azimuth (degrees)", labelpad=1)


def save_figures(images, T, azimuth, image_shape, output):
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 7,
        "pdf.fonttype": 42, "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
    })
    fig = plt.figure(figsize=(3.5, 4.0))
    add_grid(fig, images, image_shape, (0.005, 0.005, 0.99, 0.99))
    fig.savefig(output / "yale_interpolation.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(3.5, 2.8))
    add_coordinates(fig, T, azimuth, (0.15, 0.34, 0.82, 0.63),
                    (0.23, 0.14, 0.68, 0.035))
    fig.savefig(output / "yale_latent_coordinates.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(3.5, 2.23))
    add_grid(fig, images, image_shape, (0.005, 0.10, 0.469, 0.885))
    add_coordinates(fig, T, azimuth, (0.605, 0.35, 0.377, 0.60),
                    (0.62, 0.205, 0.34, 0.035))
    fig.text(0.24, 0.027, "(a)", ha="center", va="center", fontsize=8)
    fig.text(0.79, 0.027, "(b)", ha="center", va="center", fontsize=8)
    fig.savefig(output / "yale_latent_interpolation_and_coordinates.pdf")
    fig.savefig(output / "yale_latent_interpolation_and_coordinates.png", dpi=300)
    plt.close(fig)


def run_experiment(args):
    args.output.mkdir(parents=True, exist_ok=True)
    data = loadmat(args.data, squeeze_me=True)
    clean = np.asarray(data["images"], dtype=float)
    ids = np.random.default_rng(args.seed).choice(clean.shape[1], 40, replace=False)
    X = clean[:, ids]
    image_shape = tuple(np.asarray(data["image_shape"], dtype=int))
    filenames = np.asarray(data["filenames"]).ravel()[ids]
    azimuth = np.asarray(data["azimuth_deg"]).ravel()[ids]
    elevation = np.asarray(data["elevation_deg"]).ravel()[ids]
    print(f"Yale Face B: D={X.shape[0]}, available={clean.shape[1]}, samples=40")
    print(f"AQMF: d=2, s=1, p0=1.4, seed={args.seed}", flush=True)
    start = perf_counter()
    with threadpool_limits(limits=1):
        initial = make_problem(X, args.seed)
        model = alternating_mle_sqmf(
            X, *initial, p0=1.4, p_bounds=(1.0, 2.0),
            max_outer_iter=args.outer_iter, theta_max_iter=args.max_iter,
            tol=args.tol,
        )
    seconds = perf_counter() - start
    T = model["T"]
    limits = np.column_stack((T.min(axis=1), T.max(axis=1)))
    axes = [np.linspace(low, high, 10) for low, high in limits]
    grid = np.stack(np.meshgrid(*axes, indexing="ij")).reshape(2, -1)
    images = model["c"][:, None] + model["U"] @ grid
    images += model["V"] @ quadratic_features(model["A"], grid)
    rows = [dict(sample_order=order + 1, face_id=int(index) + 1,
                 filename=str(filenames[order]).strip(),
                 azimuth_deg=float(azimuth[order]),
                 elevation_deg=float(elevation[order]),
                 tau1=float(T[0, order]), tau2=float(T[1, order]))
            for order, index in enumerate(ids)]
    summary = dict(model="AQMF", samples=40, d=2, s=1, seed=args.seed,
                   p0=1.4, p_hat=float(model["p"]),
                   alpha_hat=float(model["alpha"]), eta_hat=float(model["eta"]),
                   nll=float(model["nll"]), mse=float(np.mean(model["R"] ** 2)),
                   iterations=model["n_iter"], converged=model["converged"],
                   seconds=seconds)
    write_csv(args.output / "samples.csv", rows)
    write_csv(args.output / "summary.csv", [summary])
    write_csv(args.output / "history.csv", model["history"])
    np.savez_compressed(
        args.output / "fit.npz", samples=X, sample_ids=ids,
        filenames=filenames, azimuth_deg=azimuth, elevation_deg=elevation,
        image_shape=image_shape, grid=grid, latent_limits=limits,
        interpolated_images=images, R=model["R"],
        p_hat=model["p"], alpha_hat=model["alpha"], eta_hat=model["eta"],
        nll=model["nll"], n_iter=model["n_iter"], converged=model["converged"],
        history_iteration=[item["iteration"] for item in model["history"]],
        history_p=[item["p"] for item in model["history"]],
        history_alpha=[item["alpha"] for item in model["history"]],
        history_eta=[item["eta"] for item in model["history"]],
        history_nll=[item["nll"] for item in model["history"]],
        **{key: model[key] for key in PARAMETERS},
    )
    save_figures(images, T, azimuth, image_shape, args.output)
    print(f"p_hat={summary['p_hat']:.6f}, alpha_hat={summary['alpha_hat']:.6f}, "
          f"eta_hat={summary['eta_hat']:.6f}, MSE={summary['mse']:.8f}")
    print(f"Iterations={model['n_iter']}, converged={model['converged']}, "
          f"seconds={seconds:.3f}")
    print(f"Results: {args.output.resolve()}")
