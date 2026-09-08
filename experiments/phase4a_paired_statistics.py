
"""
Phase 4-A Paired Statistical Analysis

Purpose
-------
Statistical validation of the Phase-4A polarization-aware PHY benchmark.

Compares:
    OFF = polarization disabled
    ON  = polarization enabled

Analysis:
    1. Paired mean difference
    2. Paired t-test
    3. Wilcoxon signed-rank test
    4. Cohen's dz effect size
    5. 95% CI of paired mean difference
    6. Per-seed paired differences

Important
---------
The six seeds were evaluated using common random numbers.
Therefore, OFF and ON observations are paired by seed.

This script does NOT retrain SAC.
It analyzes the controlled PHY benchmark only.
"""

from pathlib import Path
import sys

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Benchmark data
# ---------------------------------------------------------------------
#
# These values are the frozen results from:
# results/phase4a_multiseed_benchmark.txt
#
# Seeds:
#   42, 123, 2026, 4096, 7777, 9999
#
# Values are ordered identically for OFF and ON.
# ---------------------------------------------------------------------

SEEDS = np.array(
    [42, 123, 2026, 4096, 7777, 9999],
    dtype=int,
)


DATA = {
    "total_reward": {
        "OFF": np.array([
            56.952406,
            58.294871,
            58.069850,
            56.804922,
            55.791591,
            58.243936,
        ]),
        "ON": np.array([
            56.563265,
            57.076171,
            56.241303,
            56.526969,
            55.321823,
            57.004617,
        ]),
        "unit": "reward",
    },

    "mean_reward": {
        "OFF": np.array([
            0.28476203,
            0.29147436,
            0.29034925,
            0.28402461,
            0.27895796,
            0.29121968,
        ]),
        "ON": np.array([
            0.28281633,
            0.28538086,
            0.28120652,
            0.28263485,
            0.27660912,
            0.28502309,
        ]),
        "unit": "reward/step",
    },

    "spectral_efficiency": {
        "OFF": np.array([
            0.07672685,
            0.07755047,
            0.07817515,
            0.07681200,
            0.07532364,
            0.07835962,
        ]),
        "ON": np.array([
            0.07629637,
            0.07598713,
            0.07567136,
            0.07632336,
            0.07470000,
            0.07642185,
        ]),
        "unit": "bit/s/Hz",
    },

    "energy_efficiency": {
        "OFF": np.array([
            769483.38,
            796728.65,
            786164.36,
            778457.33,
            772036.90,
            797185.28,
        ]),
        "ON": np.array([
            767182.79,
            778326.91,
            761304.64,
            772936.35,
            762142.81,
            780431.80,
        ]),
        "unit": "bit/J",
    },

    "mean_trust": {
        "OFF": np.array([
            0.5974736293,
        ] * 6),
        "ON": np.array([
            0.5970474551,
        ] * 6),
        "unit": "trust",
    },

    "trust_change": {
        "OFF": np.array([
            -0.541,
            -0.537,
            -0.538,
            -0.540,
            -0.540,
            -0.541,
        ]),
        "ON": np.array([
            -0.541,
            -0.537,
            -0.539,
            -0.540,
            -0.541,
            -0.541,
        ]),
        "unit": "trust delta",
    },

    "mean_queue": {
        "OFF": np.array([
            0.9304722762,
        ] * 6),
        "ON": np.array([
            0.9308816778,
        ] * 6),
        "unit": "normalized queue",
    },
}


# ---------------------------------------------------------------------
# Correct exact values for aggregate-only metrics
# ---------------------------------------------------------------------
#
# For metrics where only aggregate values were retained in the previous
# report, do not pretend that identical per-seed values existed.
#
# We therefore exclude those metrics from inferential testing unless
# exact per-seed observations are available.
# ---------------------------------------------------------------------

INFERENTIAL_METRICS = [
    "total_reward",
    "mean_reward",
    "spectral_efficiency",
    "energy_efficiency",
]


# ---------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------

def paired_statistics(off, on):
    """
    Calculate paired statistical quantities.

    Difference convention:
        difference = ON - OFF

    Negative values therefore indicate degradation after enabling
    polarization-aware PHY under the current configuration.
    """

    off = np.asarray(off, dtype=float)
    on = np.asarray(on, dtype=float)

    diff = on - off

    n = len(diff)

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))

    # Standard error
    se = std_diff / np.sqrt(n)

    # 95% CI using Student's t distribution
    t_critical = stats.t.ppf(0.975, df=n - 1)

    ci_low = mean_diff - t_critical * se
    ci_high = mean_diff + t_critical * se

    # Paired t-test
    t_result = stats.ttest_rel(on, off)

    # Cohen's dz
    if std_diff > 0:
        cohens_dz = mean_diff / std_diff
    else:
        cohens_dz = np.nan

    # Wilcoxon signed-rank test
    #
    # With n=6, exact testing is preferable.
    # zero_method="wilcox" excludes zero differences.
    try:
        w_result = stats.wilcoxon(
            on,
            off,
            zero_method="wilcox",
            alternative="two-sided",
            method="exact",
        )
    except ValueError:
        w_result = stats.wilcoxon(
            on,
            off,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )

    # Percent change based on OFF mean
    off_mean = float(np.mean(off))

    if abs(off_mean) > 1e-15:
        percent_change = 100.0 * mean_diff / off_mean
    else:
        percent_change = np.nan

    return {
        "n": n,
        "off_mean": float(np.mean(off)),
        "on_mean": float(np.mean(on)),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se": se,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "percent_change": percent_change,
        "t_stat": float(t_result.statistic),
        "t_p": float(t_result.pvalue),
        "wilcoxon_stat": float(w_result.statistic),
        "wilcoxon_p": float(w_result.pvalue),
        "cohens_dz": float(cohens_dz),
        "differences": diff,
    }


# ---------------------------------------------------------------------
# Effect interpretation
# ---------------------------------------------------------------------

def interpret_effect(d):
    if np.isnan(d):
        return "undefined"

    magnitude = abs(d)

    if magnitude < 0.20:
        return "negligible"
    elif magnitude < 0.50:
        return "small"
    elif magnitude < 0.80:
        return "medium"
    else:
        return "large"


def significance_label(p):
    if p < 0.001:
        return "highly significant"
    elif p < 0.01:
        return "very significant"
    elif p < 0.05:
        return "statistically significant"
    else:
        return "not statistically significant"


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main():

    print("=" * 78)
    print("PHASE-4A PAIRED STATISTICAL ANALYSIS")
    print("=" * 78)

    print()
    print("Experimental conditions:")
    print("  OFF = polarization-aware PHY disabled")
    print("  ON  = polarization-aware PHY enabled")
    print()
    print("Seeds:")
    print(" ", SEEDS.tolist())
    print()
    print("Pairing:")
    print("  Same seed -> OFF/ON paired observation")
    print("  Common random numbers -> PASS")
    print()
    print("Difference convention:")
    print("  ON - OFF")
    print("  Negative = performance degradation")
    print()

    results = {}

    # -----------------------------------------------------------------
    # Inferential statistics
    # -----------------------------------------------------------------

    for metric in INFERENTIAL_METRICS:

        off = DATA[metric]["OFF"]
        on = DATA[metric]["ON"]

        result = paired_statistics(off, on)

        results[metric] = result

        print("-" * 78)
        print(metric.upper())
        print("-" * 78)

        print(f"OFF mean              : {result['off_mean']:.10f}")
        print(f"ON mean               : {result['on_mean']:.10f}")
        print(f"Mean difference       : {result['mean_diff']:.10f}")
        print(f"Std difference        : {result['std_diff']:.10f}")
        print(f"95% CI difference     : "
              f"[{result['ci_low']:.10f}, "
              f"{result['ci_high']:.10f}]")

        print(f"Percent change        : "
              f"{result['percent_change']:.4f}%")

        print()
        print("Paired t-test")
        print(f"  t                   : {result['t_stat']:.6f}")
        print(f"  p                   : {result['t_p']:.6f}")
        print(f"  interpretation      : "
              f"{significance_label(result['t_p'])}")

        print()
        print("Wilcoxon signed-rank")
        print(f"  W                   : "
              f"{result['wilcoxon_stat']:.6f}")
        print(f"  p                   : "
              f"{result['wilcoxon_p']:.6f}")
        print(f"  interpretation      : "
              f"{significance_label(result['wilcoxon_p'])}")

        print()
        print("Effect size")
        print(f"  Cohen's dz          : "
              f"{result['cohens_dz']:.6f}")
        print(f"  magnitude            : "
              f"{interpret_effect(result['cohens_dz'])}")

        print()
        print("Per-seed ON - OFF:")

        for seed, diff in zip(SEEDS, result["differences"]):
            print(
                f"  seed={seed:<5d} "
                f"difference={diff:+.10f}"
            )

    # -----------------------------------------------------------------
    # Generate report
    # -----------------------------------------------------------------

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    report_path = results_dir / "phase4a_paired_statistics.txt"

    with report_path.open("w", encoding="utf-8") as f:

        f.write("=" * 78 + "\n")
        f.write("PHASE-4A PAIRED STATISTICAL ANALYSIS\n")
        f.write("=" * 78 + "\n\n")

        f.write("Project: TA-FDRL-IRF\n")
        f.write("Phase: 4-A Polarization-Aware PHY\n\n")

        f.write("Experimental conditions:\n")
        f.write("  OFF = polarization-aware PHY disabled\n")
        f.write("  ON  = polarization-aware PHY enabled\n\n")

        f.write(
            "The OFF and ON experiments are paired by random seed "
            "using common random numbers.\n"
        )

        f.write(
            "Difference convention: ON - OFF\n"
        )

        f.write(
            "Negative differences indicate degradation after enabling "
            "polarization-aware PHY.\n\n"
        )

        f.write("Seeds:\n")
        f.write("  " + ", ".join(map(str, SEEDS)) + "\n\n")

        # -------------------------------------------------------------
        # Summary table
        # -------------------------------------------------------------

        f.write("-" * 78 + "\n")
        f.write("SUMMARY\n")
        f.write("-" * 78 + "\n\n")

        f.write(
            f"{'Metric':<25}"
            f"{'OFF':>14}"
            f"{'ON':>14}"
            f"{'Change %':>14}"
            f"{'p(t)':>12}"
            f"{'p(W)':>12}"
            f"{'dz':>12}\n"
        )

        f.write("-" * 78 + "\n")

        for metric in INFERENTIAL_METRICS:

            r = results[metric]

            f.write(
                f"{metric:<25}"
                f"{r['off_mean']:>14.6f}"
                f"{r['on_mean']:>14.6f}"
                f"{r['percent_change']:>13.4f}%"
                f"{r['t_p']:>12.6f}"
                f"{r['wilcoxon_p']:>12.6f}"
                f"{r['cohens_dz']:>12.4f}\n"
            )

        f.write("\n")

        # -------------------------------------------------------------
        # Detailed results
        # -------------------------------------------------------------

        for metric in INFERENTIAL_METRICS:

            r = results[metric]

            f.write("=" * 78 + "\n")
            f.write(metric.upper() + "\n")
            f.write("=" * 78 + "\n\n")

            f.write(
                f"OFF mean             : {r['off_mean']:.12f}\n"
            )

            f.write(
                f"ON mean              : {r['on_mean']:.12f}\n"
            )

            f.write(
                f"Mean difference      : {r['mean_diff']:.12f}\n"
            )

            f.write(
                f"Std difference       : {r['std_diff']:.12f}\n"
            )

            f.write(
                f"95% CI               : "
                f"[{r['ci_low']:.12f}, "
                f"{r['ci_high']:.12f}]\n"
            )

            f.write(
                f"Percent change       : "
                f"{r['percent_change']:.6f}%\n"
            )

            f.write("\n")

            f.write("Paired t-test:\n")
            f.write(
                f"  t                  : "
                f"{r['t_stat']:.8f}\n"
            )

            f.write(
                f"  p                  : "
                f"{r['t_p']:.8f}\n"
            )

            f.write(
                f"  interpretation     : "
                f"{significance_label(r['t_p'])}\n"
            )

            f.write("\n")

            f.write("Wilcoxon signed-rank test:\n")
            f.write(
                f"  W                  : "
                f"{r['wilcoxon_stat']:.8f}\n"
            )

            f.write(
                f"  p                  : "
                f"{r['wilcoxon_p']:.8f}\n"
            )

            f.write(
                f"  interpretation     : "
                f"{significance_label(r['wilcoxon_p'])}\n"
            )

            f.write("\n")

            f.write("Effect size:\n")
            f.write(
                f"  Cohen's dz         : "
                f"{r['cohens_dz']:.8f}\n"
            )

            f.write(
                f"  magnitude           : "
                f"{interpret_effect(r['cohens_dz'])}\n"
            )

            f.write("\n")

            f.write("Per-seed paired differences (ON - OFF):\n")

            for seed, diff in zip(SEEDS, r["differences"]):

                f.write(
                    f"  seed={seed:<5d} "
                    f"{diff:+.12f}\n"
                )

            f.write("\n")

        # -------------------------------------------------------------
        # Scientific interpretation
        # -------------------------------------------------------------

        f.write("=" * 78 + "\n")
        f.write("SCIENTIFIC INTERPRETATION\n")
        f.write("=" * 78 + "\n\n")

        reward = results["total_reward"]
        se = results["spectral_efficiency"]
        ee = results["energy_efficiency"]

        f.write(
            "Across six controlled random seeds, enabling the "
            "polarization-aware PHY model changed the mean total reward "
            f"by {reward['percent_change']:.2f}% relative to the "
            "polarization-disabled condition.\n\n"
        )

        f.write(
            "The corresponding spectral-efficiency change was "
            f"{se['percent_change']:.2f}%, while the energy-efficiency "
            f"change was {ee['percent_change']:.2f}%.\n\n"
        )

        f.write(
            "The paired statistical tests should be interpreted "
            "cautiously because the benchmark contains only six "
            "independent random seeds.\n\n"
        )

        f.write(
            "The purpose of this analysis is therefore to quantify "
            "the direction, magnitude, and uncertainty of the "
            "polarization effect rather than to claim broad statistical "
            "generalization from a small sample.\n\n"
        )

        f.write(
            "Phase-4A status: statistical validation generated.\n"
        )

    # -----------------------------------------------------------------
    # Final console message
    # -----------------------------------------------------------------

    print()
    print("=" * 78)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 78)

    print()
    print(f"Report:")
    print(f"  {report_path}")

    print()
    print("Next recommended step:")
    print("  1. Inspect the p-values and Cohen's dz.")
    print("  2. Freeze Phase-4A if the PHY implementation is accepted.")
    print("  3. Start Phase-4B with polarization-aware state representation.")


if __name__ == "__main__":
    main()

