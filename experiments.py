"""
experiments.py
==============
Runs every experiment in the paper and saves raw results to results/.

    python experiments.py --smoke          # 1 palette, 1 seed
    python experiments.py --quick          # 3 seeds, named palettes only
    python experiments.py                  # full run 
    python experiments.py --only main      # one experiment group
    
main         GA vs Random search, (1+20)-ES, CMA-ES          (all instances)
ablation     GA variants and the draft's maximin formulation (all instances)
sensitivity  GA over a tau sweep and a delta sweep            (illustrative + real)
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import benchmarks as B
import colour_math as cm
from problem import Problem
from optimisers import repair_palette

CVD_ONLY = ("protan", "deutan", "tritan")

# name -> (optimiser, repair_palette kwargs, Problem kwargs)
MAIN = {
    "GA": ("GA", {}, {}),
    "Random search": ("Random search", {}, {}),
    "(1+20)-ES": ("(1+20)-ES", {}, {}),
    "CMA-ES": ("CMA-ES", {}, {}),
}
ABLATION = {
    "GA, all colours mutable": ("GA", dict(cover="all"), {}),
    "GA, greedy cover": ("GA", dict(cover="greedy"), {}),
    "GA, no normal observer": ("GA", {}, dict(observers=CVD_ONLY)),
    "GA, no polish": ("GA", dict(use_polish=False), {}),
    "Draft (maximin)": ("GA", dict(cover="greedy", use_polish=False, escalate=False),
                        dict(objective="maximin", observers=CVD_ONLY)),
}
TAU_SWEEP = (4, 6, 8, 10, 12, 15)
DELTA_SWEEP = (6, 8, 10, 12, 14)

HISTORY_POINTS = 60


def _task(instance, hexes, method, spec, seed, delta, tau):
    optimiser, run_kw, prob_kw = spec
    p = Problem(cm.hex_to_lab(hexes), delta=delta, tau=tau,
                ordered=instance in B.ORDERED, **prob_kw)
    r = repair_palette(p, optimiser, seed=seed, **run_kw)
    h = np.array(r.pop("history") or [(0, 0.0, 0.0)])
    keep = np.unique(np.linspace(0, len(h) - 1, HISTORY_POINTS).astype(int))
    r["history"] = h[keep]
    r.update(instance=instance, method=method, seed=seed, delta=delta, tau=tau,
             k=len(hexes), original_hex=list(hexes))
    return r


def _low_priority():
    """Run workers below normal priority so the laptop stays responsive."""
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  
        else:
            os.nice(10)
    except Exception:
        pass


def _key(t):
    return (t[0], t[2], t[4], t[5], t[6])          # instance, method, seed, delta, tau


def load_partial(path):
    out = []
    if os.path.exists(path):
        with open(path, "rb") as f:
            while True:
                try:
                    out.append(pickle.load(f))
                except (EOFError, pickle.UnpicklingError):
                    break
    return out


def _run(tasks, jobs, label, partial_path):
    done = load_partial(partial_path)
    seen = {(r["instance"], r["method"], r["seed"], r["delta"], r["tau"]) for r in done}
    todo = [t for t in tasks if _key(t) not in seen]
    if seen:
        print(f"  {label}: resuming, {len(seen)} done, {len(todo)} left")
    t0, n = time.perf_counter(), 0

    def record(f, r):
        nonlocal n
        pickle.dump(r, f)
        f.flush()
        done.append(r)
        n += 1
        if n % 25 == 0 or n == len(todo):
            el = time.perf_counter() - t0
            eta = el / n * (len(todo) - n)
            print(f"  {label}: {len(done)}/{len(tasks)}  "
                  f"elapsed {el / 60:.1f} min, ~{eta / 60:.0f} min left", flush=True)

    with open(partial_path, "ab") as f:
        if jobs == 1:
            _low_priority()
            for t in todo:
                record(f, _task(*t))
        else:
            with ProcessPoolExecutor(max_workers=jobs, initializer=_low_priority) as ex:
                
                it = iter(todo)
                pending = set()
                for t in it:
                    pending.add(ex.submit(_task, *t))
                    if len(pending) >= 4 * jobs:
                        break
                while pending:
                    fut = next(as_completed(pending))
                    pending.remove(fut)
                    record(f, fut.result())
                    nxt = next(it, None)
                    if nxt is not None:
                        pending.add(ex.submit(_task, *nxt))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--sens-runs", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)),
                    help="parallel workers (default: half the cores, at most 4)")
    ap.add_argument("--smoke", action="store_true",
                    help="1 palette, 1 seed, 1 worker: checks the install in seconds")
    ap.add_argument("--quick", action="store_true",
                    help="3 seeds, no synthetic palettes; for checking the pipeline")
    ap.add_argument("--only", choices=["main", "ablation", "sensitivity"], default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    runs = 3 if a.quick else a.runs
    sens_runs = 2 if a.quick else a.sens_runs
    instances = B.all_instances(include_synthetic=not a.quick)
    small = {**B.ILLUSTRATIVE, **B.REAL}
    if a.smoke:
        runs, sens_runs, a.jobs = 1, 1, 1
        instances = small = {"Tol Bright": B.REAL["Tol Bright"]}
    print(f"{len(instances)} instances, {runs} runs, {a.jobs} workers")

    groups = [a.only] if a.only else ["main", "ablation", "sensitivity"]
    for g in groups:
        if g in ("main", "ablation"):
            methods = MAIN if g == "main" else ABLATION
            tasks = [(n, hx, m, spec, s, 10.0, 10.0)
                     for n, hx in instances.items() for m, spec in methods.items()
                     for s in range(runs)]
        else:
            ga = MAIN["GA"]
            tasks = ([(n, hx, "GA", ga, s, 10.0, float(t)) for n, hx in small.items()
                      for t in TAU_SWEEP for s in range(sens_runs)] +
                     [(n, hx, "GA", ga, s, float(d), 10.0) for n, hx in small.items()
                      for d in DELTA_SWEEP if d != 10 for s in range(sens_runs)])
        res = _run(tasks, a.jobs, g, os.path.join(a.out, f"{g}.partial"))
        with open(os.path.join(a.out, f"{g}.pkl"), "wb") as f:
            pickle.dump(res, f)
        print(f"saved {a.out}/{g}.pkl ({len(res)} runs)")


if __name__ == "__main__":
    main()
