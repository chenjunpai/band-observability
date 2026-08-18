"""51 -- dual-axis plot of the discontinuous rank transition vs the gradual
rate recovery just above the stripe threshold.

Reads the frozen exp41 stripe ladder at nu = 5e-3; no dynamics."""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "51_offset_rank_rate"
FIG = ROOT / "Figure"
RES = ROOT / "results"
NU = 0.005


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    d = json.loads((RES / "41_stripe_nyquist_v3" / "results.json").read_text())
    rows = [r for r in d["rows"] if r["nu"] == NU]
    rows.sort(key=lambda r: r["p"])
    ps = [r["p"] for r in rows]
    dk = [r["delta_K"] for r in rows]
    MU_RATE = 5.0
    rate = []
    for r in rows:
        vals = [x["rate"] for x in r["per_run"]
                if abs(x["mu"] - MU_RATE) < 1e-9]
        if r["p"] <= 10:
            rate.append(np.nan)   # no meaningful exponential fit; leave a gap
        else:
            rate.append(float(np.median(vals)) if vals else np.nan)
    pstar = rows[0]["p_star"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    fig, ax1 = plt.subplots(figsize=(4.8, 3.0))
    color_dk = "#1f77b4"
    color_rate = "#d62728"

    ax1.plot(ps, dk, "-o", color=color_dk, label=r"$\delta_K$")
    ax1.axvline(pstar, color="0.35", linestyle="--", lw=0.9)
    ax1.set_xlabel("stripe pitch $p$")
    ax1.set_ylabel(r"$\delta_K$", color=color_dk)
    ax1.tick_params(axis="y", labelcolor=color_dk)

    ax2 = ax1.twinx()
    ax2.plot(ps, rate, "-s", color=color_rate, label="median rate at $\\mu=5$")
    ax2.set_ylabel("median decay rate", color=color_rate)
    ax2.tick_params(axis="y", labelcolor=color_rate)
    ax2.set_ylim(bottom=0)

    ylim = ax1.get_ylim()
    ax1.annotate(r"$p^*=2K_c+1$", xy=(pstar + 1.0, ylim[0]),
                 xytext=(0, 18), textcoords="offset points",
                 fontsize=8, ha="center", clip_on=False)
    ax1.annotate("rank transition", xy=(pstar, 0.16), xytext=(pstar + 0.7, 0.30),
                 fontsize=8, color=color_dk, ha="left")
    ax1.annotate("rate recovery", xy=(12.5, rate[ps.index(12)]),
                 xytext=(12.6, rate[ps.index(12)] - 0.10),
                 fontsize=8, color=color_rate, ha="left")

    ax1.grid(alpha=0.2, lw=0.5)
    ax1.set_xlim(min(ps) - 0.5, max(ps) + 0.5)
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(FIG / f"fig7_offset_rank_rate{ext}", bbox_inches="tight",
                    dpi=200 if ext == ".png" else None)
    plt.close(fig)

    (OUT / "results.json").write_text(json.dumps(dict(
        meta=dict(nu=NU, pstar=pstar, mu_rate=MU_RATE,
                  note="exp41 stripe ladder, median rate over all runs at mu=5; p<=10 left as NaN"),
        rows=[dict(p=p, delta_K=dk_, median_rate=r_) for p, dk_, r_ in zip(ps, dk, rate)]), indent=1))
    print("wrote", OUT / "results.json")
    print("figure", FIG / "fig7_offset_rank_rate.png")
    for p, dk_, r_ in zip(ps, dk, rate):
        print(f"  p={p:>2}  dK={dk_:+.4f}  rate={r_:.3f}")


if __name__ == "__main__":
    main()
