

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from daltonlens import simulate

import benchmarks as B
import colour_math as cm
from problem import Problem

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2,
})
COL1, COL2 = 3.5, 7.16                     
OBS = ("normal", "protan", "deutan", "tritan")
OBS_LABEL = {"normal": "Normal", "protan": "Protan", "deutan": "Deutan", "tritan": "Tritan"}

STYLE = {
    "GA": ("#0072B2", "-"), "Random search": ("#E69F00", ":"),
    "(1+20)-ES": ("#009E73", "-."), "CMA-ES": ("#D55E00", "--"),
}
OBS_STYLE = {"normal": ("#000000", "-"), "protan": ("#D55E00", "--"),
             "deutan": ("#009E73", "-."), "tritan": ("#0072B2", ":")}
SHOWCASE = ["Red-green-brown", "RdYlGn-5 (ordered)", "Pastel UI",
            "RGB+CMY primaries", "Tol Bright", "Tableau 10"]


# ---------------------------------------------------------------------------
def load(res_dir, group):
    path = os.path.join(res_dir, f"{group}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pd.DataFrame(pickle.load(f))
    partial = os.path.join(res_dir, f"{group}.partial")   
    if os.path.exists(partial):
        from experiments import load_partial
        rows = load_partial(partial)
        print(f"  using {len(rows)} partial results for '{group}'")
        return pd.DataFrame(rows) if rows else None
    return None


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(out, name.replace(".pdf", ".png")), dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  figure", name)


SHORT = {"GA, all colours mutable": "All colours mutable", "GA, greedy cover": "Greedy cover",
         "GA, no normal observer": "No normal obs.", "GA, no polish": "No polish",
         "Draft (maximin)": "Draft (maximin)", "GA": "GA (full)" }


def write(out, name, text):
    width = r"\textwidth" if "table*" in text else r"\columnwidth"
    text = text.replace("\\begin{tabular}", "\\resizebox{" + width + "}{!}{%\n\\begin{tabular}", 1)
    text = text.replace("\\end{tabular}", "\\end{tabular}}", 1)
    with open(os.path.join(out, name), "w") as f:
        f.write(text)
    print("  table ", name)


def med_iqr(x, fmt="{:.2f}"):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "--"
    q1, m, q3 = np.percentile(x, [25, 50, 75])
    return f"{fmt.format(m)} [{fmt.format(q1)}, {fmt.format(q3)}]"


def tex_name(s):
    return s.replace("&", r"\&").replace("_", r"\_")


def problem_for(name, hexes, **kw):
    return Problem(cm.hex_to_lab(hexes), ordered=name in B.ORDERED, **kw)


def representative(df, instance, method="GA"):
    """The successful run with the median total drift (not the best run)."""
    d = df[(df.instance == instance) & (df.method == method)]
    ok = d[d.success] if d.success.any() else d
    ok = ok.sort_values("total_drift")
    return ok.iloc[len(ok) // 2]


# ---------------------------------------------------------------------------
# Tables that need no optimisation runs
# ---------------------------------------------------------------------------
def table_audit(out):
    rows = []
    for name, hx in {**B.ILLUSTRATIVE, **B.REAL}.items():
        p = problem_for(name, hx)
        ev = p.evaluate(p.original, margin=0.0)
        mins = ev["min_obs"]
        cells = [f"\\textbf{{{v:.2f}}}" if v < p.delta else f"{v:.2f}" for v in mins]
        exact, greedy = len(p.exact_cover()), len(p.greedy_cover())
        rows.append(f"{tex_name(name)} & {p.k} & " + " & ".join(cells) +
                    f" & {len(p.conflict_edges())} & {exact} & {greedy} & "
                    + (r"\cmark" if ev["worst"] >= p.delta else r"\xmark") + r" \\")
    text = r"""\begin{table}[t]
\centering
\caption{Audit of the test palettes: worst-pair CIEDE2000 per observer
(bold: below $\delta=10$), conflicting pairs, and the size of the exact
and greedy vertex covers.}
\label{tab:audit}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccccccc}
\toprule
Palette & $k$ & N & P & D & T & Pairs & MVC & Greedy & Pass \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, "tab_audit.tex", text)


# ---------------------------------------------------------------------------
# Main results
# ---------------------------------------------------------------------------
def table_repair(df, out):
    ga = df[df.method == "GA"]
    rows = []
    groups = [(n, ga[ga.instance == n]) for n in {**B.ILLUSTRATIVE, **B.REAL}]
    syn = ga[ga.instance.str.startswith("Rand-")]
    for k in sorted(syn.k.unique()):
        groups.append((f"Random, $k={k}$ (n={syn[syn.k == k].instance.nunique()})",
                       syn[syn.k == k]))
    for name, d in groups:
        if d.empty:
            continue
        p0 = d.iloc[0]
        base = problem_for(p0.instance, p0.original_hex).evaluate(
            cm.hex_to_lab(p0.original_hex), margin=0.0)["worst"] \
            if not name.startswith("Random") else np.nan
        base_s = f"{base:.2f}" if not np.isnan(base) else "--"
        feas = d.evals_to_feasible.dropna()
        bound = d.cover_size.fillna(0) if "cover_size" in d else pd.Series(0, index=d.index)
        at_bound = 100 * np.mean(d.n_modified.values == bound.values)
        rows.append(
            f"{tex_name(name)} & {int(p0.k)} & {base_s} & "
            f"{np.median(d.n_modified):.0f} & {at_bound:.0f} & {100 * d.success.mean():.0f} & "
            f"{med_iqr(d.worst)} & {med_iqr(d.total_drift)} & "
            f"{np.median(d.max_drift):.2f} & "
            f"{(np.median(feas) / 1000 if len(feas) else float('nan')):.1f} & "
            f"{np.median(d.runtime):.2f}" + r" \\")
    text = r"""\begin{table*}[t]
\centering
\caption{Repair results for the proposed pipeline (GA, exact cover,
escalation, polish), """ + f"{int(ga.groupby('instance').size().median())}" + r""" independent runs per palette.
Worst $\Delta E$ and total drift are median [IQR], measured on the final
8-bit hex palette. Success means worst $\Delta E\ge\delta$ and every
colour within $\tau$. Base: worst $\Delta E$ of the original. Mod.: colours
changed. Bound: runs that change exactly as many colours as the minimum
vertex cover, which is provably the fewest possible. Evals: evaluations to
the first accessible palette.}
\label{tab:repair}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccccc}
\toprule
Palette & $k$ & Base & Mod. & Bound (\%) & Succ.\ (\%) & Worst $\Delta E$ &
Total drift & Max drift & Evals ($10^3$) & Time (s) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write(out, "tab_repair.tex", text)


def _score(d):
    """Lexicographic per-run score used for statistics: any success beats
    any failure; successes compared on total drift."""
    return np.where(d.success, d.total_drift, 1e3 + np.maximum(0, 10 - d.worst))


def _rel_drift(df, ref, m):
    """Geometric mean over palettes of median drift(m) / median drift(ref),
    using successful runs, on palettes where both methods succeed."""
    ratios = []
    for inst in df.instance.unique():
        a = df[(df.instance == inst) & (df.method == ref) & df.success].total_drift
        b = df[(df.instance == inst) & (df.method == m) & df.success].total_drift
        if len(a) and len(b) and np.median(a) > 0:
            ratios.append(np.median(b) / np.median(a))
    return float(np.exp(np.mean(np.log(ratios)))) if ratios else float("nan")


def compare_table(df, out, ref, methods, name, caption, label):
    """Per-instance two-sided Mann-Whitney U vs the reference, Holm
    corrected across instances. W = reference significantly better."""
    rows = []
    instances = df.instance.unique()
    for m in methods:
        pvals, signs = [], []
        for inst in instances:
            a = df[(df.instance == inst) & (df.method == ref)]
            b = df[(df.instance == inst) & (df.method == m)]
            if a.empty or b.empty:
                continue
            sa, sb = _score(a), _score(b)
            if np.allclose(sa, sa[0]) and np.allclose(sb, sb[0]) and np.isclose(sa[0], sb[0]):
                pvals.append(1.0)
            else:
                pvals.append(mannwhitneyu(sa, sb, alternative="two-sided").pvalue)
            signs.append(np.sign(np.median(sb) - np.median(sa)))
        pvals = np.array(pvals)
        order = np.argsort(pvals)
        adj = np.empty_like(pvals)
        running = 0.0
        for rank, i in enumerate(order):          # Holm step-down
            running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
            adj[i] = running
        sig = adj < 0.05
        w = int(np.sum(sig & (np.array(signs) > 0)))
        l = int(np.sum(sig & (np.array(signs) < 0)))
        d = df[df.method == m]
        wtl = "--" if m == ref else f"{w}/{len(pvals) - w - l}/{l}"
        rows.append(f"{tex_name(SHORT.get(m, m) if 'tab_ablation' in name else m)} & {100 * d.success.mean():.1f} & "
                    f"{d.n_modified.mean():.2f} & {_rel_drift(df, ref, m):.2f} & "
                    f"{d.max_drift.median():.2f} & {wtl} & {d.runtime.median():.2f}" + r" \\")
    text = r"""\begin{table}[t]
\centering
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}
\toprule
Method & Succ.\ (\%) & Modified & Rel.\ drift & Max drift & W/T/L & Time (s) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, name, text)


def table_robustness(df, out):
    """Re-score the GA's successful palettes under models it never saw.
    DaltonLens is called on float linear RGB (no 8-bit rounding)."""
    ga = df[(df.method == "GA") & df.success]
    sims = [("Brettel 1997, sev. 1.0 (optimised)", simulate.Simulator_Brettel1997(), 1.0),
            ("Brettel 1997, sev. 0.5", simulate.Simulator_Brettel1997(), 0.5),
            ("Machado 2009, sev. 1.0", simulate.Simulator_Machado2009(), 1.0),
            ("Machado 2009, sev. 0.5", simulate.Simulator_Machado2009(), 0.5)]
    defs = (simulate.Deficiency.PROTAN, simulate.Deficiency.DEUTAN, simulate.Deficiency.TRITAN)
    rows = []
    for label, sim, sev in sims:
        worst = []
        for hexes in ga.hex:
            lin = cm.lab_to_linear(cm.hex_to_lab(hexes))[None].astype(np.float32)
            mins = []
            for d in defs:
                out_lin = np.clip(sim._simulate_cvd_linear_rgb(lin.copy(), d, sev)[0], 0, 1)
                lab = cm.linear_to_lab(out_lin.astype(np.float64))
                i, j = np.triu_indices(len(lab), 1)
                mins.append(cm.ciede2000(lab[i], lab[j]).min())
            worst.append(min(mins))
        worst = np.array(worst)
        rows.append(f"{label} & {med_iqr(worst)} & {100 * np.mean(worst >= 10 - 0.05):.1f} & "
                    f"{100 * np.mean(worst >= 8):.1f}" + r" \\")
    text = r"""\begin{table}[t]
\centering
\caption{Robustness of repaired palettes to the simulation model and to
severity: worst CVD $\Delta E$ (median [IQR]) and the share of palettes
at or above $\delta=10$ and $8$.}
\label{tab:robust}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccc}
\toprule
Evaluation model & Worst $\Delta E$ & $\ge 10$ (\%) & $\ge 8$ (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, "tab_robustness.tex", text)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_conflict_graph(out, name="Red-green-brown"):
    hx = {**B.ILLUSTRATIVE, **B.REAL}[name]
    p = problem_for(name, hx)
    cover = set(p.exact_cover())
    rgb = cm.lab_to_srgb(p.original)
    ang = np.pi / 2 - 2 * np.pi * np.arange(p.k) / p.k
    xy = np.c_[np.cos(ang), np.sin(ang)]
    conf = {}
    for i, j, o, d in p.conflicts():
        if (i, j) not in conf or d < conf[(i, j)][1]:
            conf[(i, j)] = (o, d)
    fig, ax = plt.subplots(figsize=(COL1 * 0.62, COL1 * 0.62))
    for (i, j), (o, d) in conf.items():
        c, ls = OBS_STYLE[o]
        ax.plot(*xy[[i, j]].T, color=c, ls=ls, lw=1.4, zorder=1)
        mid = xy[[i, j]].mean(axis=0) * 0.92
        ax.text(*mid, f"{d:.1f}", fontsize=6, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"), zorder=2)
    for i in range(p.k):
        ax.scatter(*xy[i], s=260, c=[rgb[i]], edgecolors="black",
                   linewidths=2.4 if i in cover else 0.6, zorder=3)
        ax.text(*(xy[i] * 1.28), f"{i}", ha="center", va="center", fontsize=7)
    for o in OBS:
        if any(v[0] == o for v in conf.values()):
            c, ls = OBS_STYLE[o]
            ax.plot([], [], color=c, ls=ls, label=OBS_LABEL[o])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False,
              handlelength=1.8, columnspacing=0.8)
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, out, "fig_conflict_graph.pdf")


def _cumulative_best(h, big=1e6):
    """History (evals, S, D) -> lexicographic best-so-far across restarts."""
    key = np.where(h[:, 1] > 0, big + h[:, 1], h[:, 2])
    idx = np.array([int(np.argmin(key[:t + 1])) for t in range(len(key))])
    return h[:, 0], h[idx, 1], h[idx, 2]


def fig_convergence(df, out, instances=("Tableau 10", "Red-green-brown")):
    instances = [i for i in instances if i in set(df.instance)]
    if not instances:
        return
    fig, axes = plt.subplots(2, len(instances), figsize=(COL1, 2.8),
                             sharex="col", squeeze=False)
    for c, inst in enumerate(instances):
        d = df[df.instance == inst]
        emax = max(r[-1, 0] for r in d.history)
        grid = np.linspace(0, emax, 200)
        for m in [m for m in STYLE if m in set(d.method)]:
            runs = d[d.method == m].history
            S = np.full((len(runs), len(grid)), np.nan)
            D = np.full_like(S, np.nan)
            for r, h in enumerate(runs):
                e, s, dd = _cumulative_best(np.asarray(h))
                idx = np.searchsorted(e, grid, side="right") - 1
                ok = idx >= 0
                S[r, ok], D[r, ok] = s[idx[ok]], dd[idx[ok]]
            D[S > 0] = np.nan
            col, ls = STYLE[m]
            q = np.nanpercentile(S, [25, 50, 75], axis=0)
            axes[0, c].plot(grid, q[1], color=col, ls=ls, label=m)
            axes[0, c].fill_between(grid, q[0], q[2], color=col, alpha=0.15, lw=0)
            enough = np.mean(~np.isnan(D), axis=0) >= 0.5
            with np.errstate(all="ignore"):
                qd = np.nanpercentile(D, [25, 50, 75], axis=0)
            qd[:, ~enough] = np.nan
            axes[1, c].plot(grid, qd[1], color=col, ls=ls)
            axes[1, c].fill_between(grid, qd[0], qd[2], color=col, alpha=0.15, lw=0)
        axes[0, c].set_yscale("symlog", linthresh=0.1)
        axes[0, c].set_ylim(bottom=0)
        axes[0, c].set_title(inst)
        axes[1, c].set_xlabel("Function evaluations")
        for ax in axes[:, c]:
            ax.grid(alpha=0.25, lw=0.4)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    axes[0, 0].set_ylabel("Shortfall $S$")
    axes[1, 0].set_ylabel("Total drift $D$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.55, 1.1))
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(h_pad=0.4, w_pad=0.8)
    save(fig, out, "fig_convergence.pdf")


def fig_sensitivity(df, out):
    if df is None or df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(COL1, 2.3), sharex="col")
    for c, (var, fixed, xlabel) in enumerate([("tau", "delta", r"Fidelity budget $\tau$"),
                                              ("delta", "tau", r"Accessibility target $\delta$")]):
        d = df[df[fixed] == 10.0]
        g = d.groupby(var)
        x = np.array(sorted(g.groups))
        axes[0, c].plot(x, g.success.mean().reindex(x).values * 100, "o-",
                        color="#0072B2", ms=3)
        axes[1, c].plot(x, g.n_modified.mean().reindex(x).values, "s--",
                        color="#D55E00", ms=3)
        axes[0, c].set_ylim(-3, 103)
        axes[1, c].set_xlabel(xlabel)
        for ax in axes[:, c]:
            ax.axvline(10, color="grey", lw=0.6, ls=":")
            ax.grid(alpha=0.25, lw=0.4)
    axes[0, 0].set_ylabel("Success (%)")
    axes[1, 0].set_ylabel("Colours modified")
    axes[0, 1].set_yticklabels([])
    top = max(max(l.get_ydata().max() for l in ax.get_lines() if len(l.get_ydata()) > 2)
              for ax in axes[1])
    for ax in axes[1]:
        ax.set_ylim(0, np.ceil(top + 0.5))
    axes[1, 1].set_yticklabels([])
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(h_pad=0.3, w_pad=0.5)
    save(fig, out, "fig_sensitivity.pdf")


def fig_qualitative(df, out, instances=SHOWCASE, name="fig_qualitative.pdf"):
    instances = [i for i in instances if i in set(df.instance)]
    fig, axes = plt.subplots(len(instances), 4, figsize=(COL2, 0.62 * len(instances) + 0.3),
                             squeeze=False)
    for r, inst in enumerate(instances):
        run = representative(df, inst)
        orig, opt = cm.hex_to_lab(run.original_hex), cm.hex_to_lab(run.hex)
        changed = np.array(run.hex) != np.array(run.original_hex)
        k = len(orig)
        for c, o in enumerate(OBS):
            ax = axes[r, c]
            so, sp = cm.simulate_lab(orig, o), cm.simulate_lab(opt, o)
            i, j = np.triu_indices(k, 1)
            wo, wp = cm.ciede2000(so[i], so[j]).min(), cm.ciede2000(sp[i], sp[j]).min()
            ax.imshow(np.stack([cm.lab_to_srgb(so), cm.lab_to_srgb(sp)]), aspect="auto",
                      extent=(0, k, 2, 0), interpolation="nearest")
            ax.axhline(1, color="white", lw=2.5)
            for x in np.where(changed)[0]:
                ax.plot(x + 0.5, 2.18, marker="^", ms=3.2, color="black", clip_on=False)
            ax.set_xlim(0, k); ax.set_ylim(2, 0)
            ax.set_xticks([]); ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels([f"{wo:.1f}", f"{wp:.1f}"], fontsize=6.5)
            ax.tick_params(length=0, pad=1.5)
            for tl, v in zip(ax.get_yticklabels(), (wo, wp)):
                tl.set_color("#B00000" if v < 10 - 1e-9 else "black")
                tl.set_fontweight("bold" if v < 10 - 1e-9 else "normal")
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if r == 0:
                ax.set_title({"normal": "Normal vision", "protan": "Protanopia",
                              "deutan": "Deuteranopia", "tritan": "Tritanopia"}[o])
            if c == 0:
                ax.set_ylabel(inst.replace(" (ordered)", ""), rotation=0, ha="right",
                              va="center", labelpad=22, fontsize=7)
    fig.tight_layout(h_pad=0.9, w_pad=0.6)
    save(fig, out, name)


def fig_movement(df, out, inst="Tableau 10"):
    if inst not in set(df.instance):
        return
    run = representative(df, inst)
    orig, opt = cm.hex_to_lab(run.original_hex), cm.hex_to_lab(run.hex)
    fig, ax = plt.subplots(figsize=(COL1 * 0.75, COL1 * 0.75))
    ax.axhline(0, color="grey", lw=0.4); ax.axvline(0, color="grey", lw=0.4)
    for i in range(len(orig)):
        moved = run.hex[i] != run.original_hex[i]
        ax.scatter(orig[i, 1], orig[i, 2], s=55, c=[cm.lab_to_srgb(orig[i])],
                   edgecolors="black", linewidths=0.5, zorder=3)
        if moved:
            ax.annotate("", xy=opt[i, 1:], xytext=orig[i, 1:],
                        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black",
                                        shrinkA=3.5, shrinkB=1.5), zorder=4)
            ax.scatter(opt[i, 1], opt[i, 2], s=28, c=[cm.lab_to_srgb(opt[i])],
                       edgecolors="black", linewidths=0.5, marker="s", zorder=5)
    ax.set_xlabel("$a^*$"); ax.set_ylabel("$b^*$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25, lw=0.4)
    save(fig, out, "fig_movement.pdf")


def fig_ablation(main, abl, out):
    if abl is None:
        return
    df = pd.concat([main[main.method == "GA"].assign(method="GA (full)"), abl])
    order = ["GA (full)", "GA, all colours mutable", "GA, greedy cover",
             "GA, no normal observer", "GA, no polish", "Draft (maximin)"]
    order = [m for m in order if m in set(df.method)]
    ok = df[df.success]
    med = ok.groupby(["method", "instance"]).total_drift.median()
    ref = med.loc["GA (full)"]
    ref = ref[ref > 0]                         # palettes that needed a repair
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.45), sharey=True)
    y = np.arange(len(order))[::-1]
    succ = [100 * df[df.method == m].success.mean() for m in order]
    mod = [df[df.method == m].n_modified.mean() for m in order]
    rel = []                                   # one ratio of medians per palette
    for m in order:
        mm = med.loc[m].reindex(ref.index) if m in med.index.get_level_values(0) else None
        r = (mm / ref).values if mm is not None else np.array([])
        rel.append(r[np.isfinite(r)])
    axes[0].barh(y, succ, color="#0072B2", height=0.6)
    axes[0].set_xlabel("Success (%)"); axes[0].set_xlim(0, 100)
    axes[1].barh(y, mod, color="#E69F00", height=0.6)
    axes[1].set_xlabel("Mean colours modified")
    axes[2].boxplot(rel, positions=y, vert=False, widths=0.55,
                    showfliers=True, flierprops=dict(markersize=2),
                    medianprops=dict(color="black"))
    axes[2].axvline(1, color="grey", ls=":", lw=0.8)
    axes[2].set_xlabel("Drift ratio to full GA")
    axes[0].set_yticks(y); axes[0].set_yticklabels(order)
    for ax in axes:
        ax.grid(axis="x", alpha=0.25, lw=0.4)
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig_ablation.pdf")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--out", default="paper_figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    main_df, abl, sens = load(a.res, "main"), load(a.res, "ablation"), load(a.res, "sensitivity")
    table_audit(a.out)
    fig_conflict_graph(a.out)
    if main_df is not None:
        table_repair(main_df, a.out)
        compare_table(main_df, a.out, "GA", list(STYLE), "tab_methods.tex",
                      "Optimiser comparison under identical budgets, repair and ranking. "
                      "Rel.\\ drift is the geometric mean, over palettes where both succeed, of median drift relative to the GA. W/T/L counts palettes where "
                      "the GA is significantly better / not different / worse (two-sided "
                      "Mann-Whitney U, Holm-corrected, $\\alpha=0.05$).", "tab:methods")
        table_robustness(main_df, a.out)
        fig_convergence(main_df, a.out)
        fig_qualitative(main_df, a.out)
        fig_movement(main_df, a.out)
    if abl is not None and main_df is not None:
        both = pd.concat([main_df[main_df.method == "GA"], abl])
        keep = [m for m in ["GA, all colours mutable", "Draft (maximin)"]
                if m in set(abl.method)]
        compare_table(both, a.out, "GA", ["GA"] + keep, "tab_ablation.tex",
                      "Ablation: the full pipeline against letting every colour change, and "
                      "against the earlier maximin formulation (maximin objective, greedy "
                      "cover, CVD observers only, no escalation). W/T/L as in "
                      "Table~\\ref{tab:methods}.", "tab:ablation")
        fig_ablation(main_df, abl, a.out)
    fig_sensitivity(sens, a.out)


if __name__ == "__main__":
    main()