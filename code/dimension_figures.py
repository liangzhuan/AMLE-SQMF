import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd

from dimension_common import summarize


COLORS = ("#0072B2", "#D55E00", "#009E73", "#8B5AA5", "#444444")
MARKERS = ("o", "s", "^", "D", "v")
LINES = ("-", "--", "-.", ":", "-")
METRICS = (("abs_bias", "Absolute bias"), ("sd", "SD"), ("mse", "MSE"))


def set_style(size=7.3):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "font.size": size,
        "axes.labelsize": size, "xtick.labelsize": size, "ytick.labelsize": size,
        "legend.fontsize": size, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False,
    })


def export(fig, name, output):
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=280)
    plt.close(fig)


def overview(summary, output):
    set_style()
    fig, ax = plt.subplots(figsize=(3.5, 1.73))
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.25, top=0.76)
    for index, D in enumerate(sorted(summary.D.unique())):
        group = summary[summary.D == D].sort_values("n")
        ax.plot(group.Dn / 1000, group.mse, linestyle="none",
                marker=MARKERS[index], color=COLORS[index], markersize=3.4,
                label=rf"$D={D}$")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1,)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0, summary.Dn.max() / 1000 * 1.055)
    ax.set_xlabel(r"$Dn$ ($\times 10^3$)", labelpad=2)
    ax.set_ylabel("MSE", labelpad=2)
    ax.grid(axis="y", color="#DCE1E5", linewidth=0.4)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), ncol=5, loc="upper center",
               bbox_to_anchor=(0.565, 1.005), columnspacing=0.75,
               handletextpad=0.25, handlelength=0.8)
    export(fig, "Fig1_joint_scale_mse_compact", output)


def effects(summary, output):
    set_style()
    fig, axes = plt.subplots(3, 2, figsize=(3.5, 4.8))
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.09, top=0.86,
                        wspace=0.57, hspace=0.8)
    dimensions = sorted(summary.D.unique())
    sizes = sorted(summary.n.unique())
    for column, (fixed, varying, levels) in enumerate(
            (("D", "n", dimensions), ("n", "D", sizes))):
        for row, (metric, label) in enumerate(METRICS):
            ax = axes[row, column]
            for index, level in enumerate(levels):
                group = summary[summary[fixed] == level].sort_values(varying)
                x = group[varying] / 1000 if varying == "n" else group[varying]
                ax.plot(x, group[metric], color=COLORS[index], marker=MARKERS[index],
                        linestyle=LINES[index], linewidth=0.85, markersize=2.6,
                        label=rf"${fixed}={level}$")
            ax.set_ylabel(label if column == 0 else "")
            ax.grid(axis="y", color="#DCE1E5", linewidth=0.4)
            ax.set_axisbelow(True)
            if metric == "mse":
                ax.set_yscale("log")
                ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(r"$n$ ($\times 10^3$)" if varying == "n" else r"$D$")
            ax.set_xticks(np.array(sizes) / 1000 if varying == "n" else dimensions)
    for column in range(2):
        handles, labels = axes[0, column].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=1, loc="upper center",
                   bbox_to_anchor=(0.36 if column == 0 else 0.83, 1.005),
                   handlelength=1.7, labelspacing=0.15, fontsize=6.5)
    export(fig, "Fig2_sample_size_and_dimension_effects", output)


def matched_products(summary, output):
    set_style()
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.9))
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.17, top=0.79, hspace=1.12)
    matched = summary[summary.Dn.duplicated(keep=False)].Dn.unique()
    products = (90000, 180000) if set((90000, 180000)) <= set(matched) else sorted(matched)[:2]
    dimensions = sorted(summary.D.unique())
    shown = set()
    for panel, (ax, product) in enumerate(zip(axes, products)):
        group = summary[summary.Dn == product].sort_values("D")
        for position, entry in enumerate(group.itertuples()):
            index = dimensions.index(entry.D)
            shown.add(entry.D)
            ax.plot(entry.mse, position, marker=MARKERS[index],
                    color=COLORS[index], linestyle="none", markersize=3.7)
        ax.set_xscale("log")
        ax.set_yticks(range(len(group)), [
            rf"$({int(entry.D)},\,{int(entry.n)})$" for entry in group.itertuples()
        ])
        ax.set_ylim(len(group) - 0.55, -0.45)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.grid(axis="x", which="major", color="#DCE1E5", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.set_title(f"({'ab'[panel]}) " + rf"$Dn={int(product):,}$", loc="left", pad=3)
        ax.set_xlabel("MSE (log scale)", labelpad=2)
    levels = sorted(shown)
    handles = [Line2D([], [], color=COLORS[dimensions.index(D)],
                      marker=MARKERS[dimensions.index(D)], linestyle="none", markersize=3.7)
               for D in levels]
    fig.legend(handles, [rf"$D={D}$" for D in levels], ncol=4, loc="upper center",
               bbox_to_anchor=(0.59, 0.99), columnspacing=0.9,
               handletextpad=0.3, handlelength=0.8)
    export(fig, "Fig3_equal_product_mse_vertical_compact", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).with_name("dimension_results.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "figures_dimension_sample_size")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input, float_precision="round_trip")
    summary = summarize(raw)
    summary.to_csv(args.output / "condition_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    overview(summary, args.output)
    effects(summary, args.output)
    matched_products(summary, args.output)
    print(f"Figures: {args.output.resolve()}")


if __name__ == "__main__":
    main()
