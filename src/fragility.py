#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest, wilcoxon

def enumerate_motifs(edges, labels):
    label_map = labels.drop_duplicates("address").set_index("address")["class"]
    edges = edges.drop_duplicates()
    edges = edges[edges["input_address"] != edges["output_address"]].copy()
    edges["src_class"] = edges["input_address"].map(label_map)
    edges["dst_class"] = edges["output_address"].map(label_map)

    iu = edges[(edges.src_class == 1) & (edges.dst_class == 3)][
        ["input_address", "output_address"]
    ].rename(columns={"input_address": "s", "output_address": "u1"})

    uu = edges[(edges.src_class == 3) & (edges.dst_class == 3)][
        ["input_address", "output_address"]
    ].rename(columns={"input_address": "u_from", "output_address": "u_to"})

    ul = edges[(edges.src_class == 3) & (edges.dst_class == 2)][
        ["input_address", "output_address"]
    ].rename(columns={"input_address": "u_last", "output_address": "t"})

    m1 = iu.merge(ul, left_on="u1", right_on="u_last")[
        ["s", "u1", "t"]
    ].drop_duplicates()

    m2 = (
        iu.merge(uu, left_on="u1", right_on="u_from")
          .merge(ul, left_on="u_to", right_on="u_last")
          [["s", "u1", "u_to", "t"]]
          .rename(columns={"u_to": "u2"})
    )
    m2 = m2[(m2.s != m2.t) & (m2.s != m2.u2) & (m2.u1 != m2.t)].drop_duplicates()

    prefix2 = iu.merge(uu, left_on="u1", right_on="u_from")[
        ["s", "u1", "u_to"]
    ].rename(columns={"u_to": "u2"})
    m3 = (
        prefix2.merge(uu, left_on="u2", right_on="u_from")
               .merge(ul, left_on="u_to", right_on="u_last")
               [["s", "u1", "u2", "u_to", "t"]]
               .rename(columns={"u_to": "u3"})
    )
    cols = ["s", "u1", "u2", "u3", "t"]
    keep = np.ones(len(m3), dtype=bool)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            keep &= m3[cols[i]].to_numpy() != m3[cols[j]].to_numpy()
    m3 = m3[keep].drop_duplicates()
    return m1, m2, m3

def analyze(df, unknown_cols, stages, probability):
    grouped = defaultdict(list)
    for row in df[["s", "t"] + unknown_cols].itertuples(index=False, name=None):
        s, t, *unknowns = row
        grouped[(s, t)].append(tuple(unknowns))

    records = []
    for motif_id, row in enumerate(
        df[["s", "t"] + unknown_cols].itertuples(index=False, name=None)
    ):
        s, t, *unknowns = row
        baseline_paths = grouped[(s, t)]
        p0 = len(baseline_paths)
        ranked = sorted(unknowns, key=lambda a: (-probability[a], a))
        first_failure = None

        for proportion, removed_count in stages:
            removed = set(ranked[:removed_count])
            remaining = sum(
                1 for path in baseline_paths if removed.isdisjoint(path)
            )
            disconnected = remaining == 0
            if disconnected and first_failure is None:
                first_failure = proportion
            records.append({
                "motif_id": motif_id,
                "removal_proportion": proportion,
                "baseline_paths": p0,
                "remaining_paths": remaining,
                "path_loss_ratio": (p0 - remaining) / p0,
                "disconnected": disconnected,
                "first_failure_threshold": first_failure,
            })
    return pd.DataFrame(records)

def bootstrap_ci(values, rng, repetitions=2000):
    values = np.asarray(values, dtype=float)
    means = np.empty(repetitions)
    for i in range(repetitions):
        means[i] = values[rng.integers(0, len(values), len(values))].mean()
    return np.quantile(means, [0.025, 0.975])

def summarize(result_map):
    rng = np.random.default_rng(42)
    rows = []
    for motif_type, result in result_map.items():
        for level, group in result.groupby("removal_proportion"):
            low, high = bootstrap_ci(group["path_loss_ratio"], rng)
            n = len(group)
            disconnected = int(group["disconnected"].sum())
            rows.append({
                "motif_type": motif_type,
                "removal_proportion": level,
                "motifs": n,
                "connected_n": n - disconnected,
                "connected_pct": 100 * (n - disconnected) / n,
                "disconnected_n": disconnected,
                "disconnected_pct": 100 * disconnected / n,
                "mean_remaining_paths": group["remaining_paths"].mean(),
                "median_remaining_paths": group["remaining_paths"].median(),
                "mean_path_loss": group["path_loss_ratio"].mean(),
                "median_path_loss": group["path_loss_ratio"].median(),
                "ci_low": low,
                "ci_high": high,
            })
    return pd.DataFrame(rows)

def paired_tests(result_map):
    rows = []
    for motif_type in (2, 3):
        result = result_map[motif_type]
        disc = result.pivot(
            index="motif_id", columns="removal_proportion", values="disconnected"
        )
        paths = result.pivot(
            index="motif_id", columns="removal_proportion", values="remaining_paths"
        )
        levels = sorted(disc.columns)
        for left, right in zip(levels[:-1], levels[1:]):
            a = disc[left].to_numpy(bool)
            b = disc[right].to_numpy(bool)
            b01 = int(((~a) & b).sum())
            b10 = int((a & (~b)).sum())
            p_mcnemar = (
                1.0 if b01 + b10 == 0
                else binomtest(min(b01, b10), b01 + b10, 0.5).pvalue
            )
            x = paths[left].to_numpy(float)
            y = paths[right].to_numpy(float)
            if np.all(x == y):
                statistic, p_wilcoxon = 0.0, 1.0
            else:
                test = wilcoxon(
                    x, y, zero_method="pratt",
                    alternative="greater", method="approx"
                )
                statistic, p_wilcoxon = test.statistic, test.pvalue
            rows.append({
                "motif_type": motif_type,
                "comparison": f"{left:.2f} vs {right:.2f}",
                "mcnemar_b01": b01,
                "mcnemar_b10": b10,
                "mcnemar_exact_p": p_mcnemar,
                "wilcoxon_statistic": statistic,
                "wilcoxon_one_sided_p": p_wilcoxon,
            })
    return pd.DataFrame(rows)

def make_plots(summary, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, group in summary.groupby("motif_type"):
        group = group.sort_values("removal_proportion")
        ax.plot(
            100 * group["removal_proportion"],
            group["disconnected_pct"],
            marker="o", label=f"Motif {k}"
        )
    ax.set_xlabel("Removed unknown intermediaries (%)")
    ax.set_ylabel("Disconnected motifs (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fragility_disconnection.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fragility_disconnection.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, group in summary.groupby("motif_type"):
        group = group.sort_values("removal_proportion")
        x = 100 * group["removal_proportion"]
        y = group["mean_path_loss"]
        error = np.vstack([y - group["ci_low"], group["ci_high"] - y])
        ax.errorbar(x, y, yerr=error, marker="o", capsize=3, label=f"Motif {k}")
    ax.set_xlabel("Removed unknown intermediaries (%)")
    ax.set_ylabel("Mean path-loss ratio")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fragility_path_loss.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fragility_path_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("motif_fragility_results"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(args.edges, dtype=str)
    labels = pd.read_csv(args.labels, dtype={"address": str, "class": int})
    probabilities = pd.read_csv(
        args.probabilities,
        usecols=["address", "predicted_illicit_probability"],
        dtype={"address": str, "predicted_illicit_probability": float},
    )
    probability = probabilities.drop_duplicates("address").set_index("address")[
        "predicted_illicit_probability"
    ].to_dict()

    m1, m2, m3 = enumerate_motifs(edges, labels)
    m1.to_csv(args.out_dir / "motif_1_instances.csv", index=False)
    m2.to_csv(args.out_dir / "motif_2_instances.csv", index=False)
    m3.to_csv(args.out_dir / "motif_3_instances.csv", index=False)

    result_map = {
        1: analyze(m1, ["u1"], [(1.00, 1)], probability),
        2: analyze(m2, ["u1", "u2"], [(0.50, 1), (1.00, 2)], probability),
        3: analyze(m3, ["u1", "u2", "u3"], [(0.33, 1), (0.67, 2), (1.00, 3)], probability),
    }

    for motif_type, result in result_map.items():
        result.to_csv(
            args.out_dir / f"motif_{motif_type}_node_removal_results.csv",
            index=False
        )

    failure_rows = []
    for k, result in result_map.items():
        for _, group in result.groupby("motif_id", sort=True):
            failed = group.loc[group["disconnected"], "removal_proportion"]
            stage = "No failure" if failed.empty else f"{round(100 * failed.min())}%"
            failure_rows.append({"motif_type": k, "first_failure_stage": stage})
    pd.DataFrame(failure_rows).groupby(
        ["motif_type", "first_failure_stage"], sort=False
    ).size().reset_index(name="motifs").to_csv(
        args.out_dir / "first_failure_thresholds.csv", index=False
    )
    summary = summarize(result_map)
    tests = paired_tests(result_map)
    summary.to_csv(args.out_dir / "fragility_summary.csv", index=False)
    tests.to_csv(args.out_dir / "paired_tests.csv", index=False)
    make_plots(summary, args.out_dir)
    print(summary.to_string(index=False))
    print(tests.to_string(index=False))

if __name__ == "__main__":
    main()
