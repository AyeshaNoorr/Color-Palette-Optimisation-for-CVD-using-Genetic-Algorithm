# Minimal-change CVD palette repair

Repairs a categorical palette so every pair of colours stays at least
`delta` CIEDE2000 apart for normal, protan, deutan and tritan observers
(Brettel 1997), while changing as few colours as possible, each by as
little as possible and never by more than `tau`.

## Setup

    pip install -r requirements.txt
    python verify.py        # maths check against colour-science / DaltonLens

## Run the paper experiments

    python experiments.py --smoke      # 10 seconds: checks the install
    python experiments.py              # full: 30 seeds, ~3 CPU-hours total
    python report.py                   # figures + LaTeX tables -> paper_figures/




## Repair one palette

    python main.py --hex "#B22222" "#228B22" "#8B4513" "#D2691E" "#808000"
    python main.py --image photo.png --colours 6 --save repaired.pdf

## Files

| File | Purpose |
|---|---|
| `colour_math.py` | Vectorised float64 sRGB/LAB, gamut mapping, CIEDE2000, Brettel 1997 |
| `problem.py` | Formulation: observers, conflicts, exact/greedy vertex cover, lexicographic scoring, guaranteed drift repair |
| `optimisers.py` | GA, random search, (1+20)-ES, CMA-ES, polish, full pipeline (`repair_palette`) |
| `benchmarks.py` | Illustrative, real and seeded synthetic palettes |
| `experiments.py` | Main comparison, ablation, sensitivity (saves `results/*.pkl`) |
| `report.py` | All figures (PDF + PNG) and LaTeX tables |
| `main.py` | Command-line repair of one palette |
| `palette_extraction.py` | k-means palette from an image |
| `verify.py` | Checks the maths against reference libraries |

## Pipeline

1. **Audit.** Simulate the palette for the four observers; every pair with
   dE00 < delta under any observer is a conflict. No conflicts -> return
   the palette unchanged (C1).
2. **Mutable set.** Exact minimum vertex cover of the conflict graph
   (enumeration; k <= 15 is instant). Ties go to the colours carrying the
   most shortfall (C2).
3. **Optimise** only the mutable colours. Lexicographic objective (Deb's
   feasibility rules): minimise accessibility shortfall
   S = sum max(0, delta - dE00), then total drift D (C3).
   Every candidate is gamut-mapped and contracted toward its original so
   drift <= tau holds by construction.
4. **Escalate** if the cover cannot be repaired within tau: add the frozen
   colour carrying the most remaining shortfall and restart from scratch
   (warm starts got trapped in testing).
5. **Polish:** pull each changed colour back along its ray to the original
   as far as accessibility allows.
6. **Round to hex** and measure everything on the rounded palette with the
   true delta and tau. The optimiser uses delta + 0.3 and tau - 0.3
   internally so rounding cannot break either constraint.

## Defaults

delta = 10, tau = 10, budget = 10,000 evaluations per optimisation stage.
GA: population 100, 10% elitism, tournament 4, uniform crossover on whole
colours, Gaussian mutation p = 0.25, sigma = 3.0 (CIELAB units).

## Outputs of report.py

| Output | Paper use |
|---|---|
| `tab_audit.tex` | Audit of test palettes (no runs needed) |
| `tab_repair.tex` | Main results per palette + synthetic summary |
| `tab_methods.tex` | GA vs baselines, Mann-Whitney U + Holm |
| `tab_ablation.tex` | Component ablation incl. the draft formulation |
| `tab_robustness.tex` | Re-scoring under Machado 2009 and severity 0.5 |
| `fig_conflict_graph` | Conflict graph + minimum cover (method figure) |
| `fig_convergence` | Median + IQR vs evaluations, GA vs baselines |
| `fig_qualitative` | Before/after under all observers (representative run) |
| `fig_sensitivity` | Success and colours modified vs tau and delta |
| `fig_ablation` | Success, colours modified, relative drift |
| `fig_movement` | a*b* displacement of changed colours |

Tables use `\cmark`/`\xmark`; add to the preamble:

    \usepackage{booktabs,pifont}
    \newcommand{\cmark}{\ding{51}}
    \newcommand{\xmark}{\ding{55}}


