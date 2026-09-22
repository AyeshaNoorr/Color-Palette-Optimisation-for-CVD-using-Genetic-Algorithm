
import time

import numpy as np

import colour_math as cm

GA_DEFAULTS = dict(pop_size=100, elite_frac=0.10, p_mut=0.25, sigma=3.0,
                   tournament=4)


class _Tracker:
    

    def __init__(self, problem):
        self.p = problem
        self.evals = 0
        self.best_key = np.inf
        self.best = problem.original.copy()
        self.best_S = self.best_D = np.inf
        self.history = []
        self.first_feasible = None

    def update(self, pop, ev):
        self.evals += len(pop)
        i = int(np.argmin(ev["key"]))
        if ev["key"][i] < self.best_key:
            self.best_key = ev["key"][i]
            self.best = pop[i].copy()
            self.best_S, self.best_D = ev["S"][i], ev["D"][i]
        if self.first_feasible is None and self.best_S == 0:
            self.first_feasible = self.evals
        self.history.append((self.evals, self.best_S, self.best_D))


def _start_population(problem, mutable, rng, n, sigma, init):
    """Individual 0 = warm start (or the original), 1 = the original,
    the rest = warm start (or original) + Gaussian noise on mutable colours."""
    base = problem.original if init is None else init
    pop = np.repeat(base[None], n, axis=0)
    noise = rng.normal(0.0, sigma, size=(n, len(mutable), 3))
    pop[:, mutable] = problem.repair_drift(pop[:, mutable] + noise, mutable)
    pop[0], pop[1] = base, problem.original
    return pop


#constrained GA
def genetic_algorithm(problem, mutable, rng, budget=10_000, init=None,
                      pop_size=100, elite_frac=0.10, p_mut=0.25, sigma=3.0,
                      tournament=4):
    mutable = np.asarray(mutable)
    m = len(mutable)
    n_elite = max(1, int(round(pop_size * elite_frac)))
    n_child = pop_size - n_elite
    tr = _Tracker(problem)

    pop = _start_population(problem, mutable, rng, pop_size, sigma, init)
    ev = problem.evaluate(pop)
    tr.update(pop, ev)
    key = ev["key"]

    while tr.evals + n_child <= budget:
        order = np.argsort(key, kind="stable")
        elites, elite_key = pop[order[:n_elite]], key[order[:n_elite]]

        # Tournament selectiono
        cand = rng.random((2 * n_child, pop_size)).argsort(axis=1)[:, :tournament]
        winners = cand[np.arange(2 * n_child), np.argmin(key[cand], axis=1)]
        pa, pb = pop[winners[:n_child]], pop[winners[n_child:]]

        # Uniform crossover over whole colour triplets (mutable colours only)
        children = pa.copy()
        swap = rng.random((n_child, m)) < 0.5
        children[:, mutable] = np.where(swap[..., None], pb[:, mutable], pa[:, mutable])

        # Gaussian mutation, then gamut mapping and guaranteed drift repair
        hit = rng.random((n_child, m)) < p_mut
        noise = rng.normal(0.0, sigma, size=(n_child, m, 3)) * hit[..., None]
        children[:, mutable] = problem.repair_drift(children[:, mutable] + noise, mutable)

        ev_c = problem.evaluate(children)
        tr.update(children, ev_c)
        pop = np.concatenate([elites, children])
        key = np.concatenate([elite_key, ev_c["key"]])

    return tr.best, tr



def random_search(problem, mutable, rng, budget=10_000, init=None, batch=1000):
    mutable = np.asarray(mutable)
    tr = _Tracker(problem)
    base = problem.original if init is None else init
    ev = problem.evaluate(base[None])
    tr.update(base[None], ev)
    while tr.evals < budget:
        n = min(batch, budget - tr.evals)
        # uniform in a CIELAB ball of radius tau around each original colour
        d = rng.normal(size=(n, len(mutable), 3))
        d *= (problem.tau * rng.random((n, len(mutable), 1)) ** (1 / 3)
              / np.linalg.norm(d, axis=-1, keepdims=True))
        pop = np.repeat(base[None], n, axis=0)
        pop[:, mutable] = problem.repair_drift(problem.original[mutable] + d, mutable)
        tr.update(pop, problem.evaluate(pop))
    return tr.best, tr


def one_plus_lambda_es(problem, mutable, rng, budget=10_000, init=None,
                       lam=20, sigma=3.0):
    
    mutable = np.asarray(mutable)
    tr = _Tracker(problem)
    parent = (problem.original if init is None else init).copy()
    ev0 = problem.evaluate(parent[None])
    tr.update(parent[None], ev0)
    pkey = ev0["key"][0]
    while tr.evals + lam <= budget:
        kids = np.repeat(parent[None], lam, axis=0)
        noise = rng.normal(0.0, sigma, size=(lam, len(mutable), 3))
        kids[:, mutable] = problem.repair_drift(parent[mutable] + noise, mutable)
        ev = problem.evaluate(kids)
        tr.update(kids, ev)
        p_success = np.mean(ev["key"] < pkey)
        sigma = float(np.clip(sigma * np.exp((p_success - 0.2) / 0.8 / 3), 0.05, 20.0))
        i = int(np.argmin(ev["key"]))
        if ev["key"][i] <= pkey:
            parent, pkey = kids[i].copy(), ev["key"][i]
    return tr.best, tr


def cma_es(problem, mutable, rng, budget=10_000, init=None, sigma=3.0, popsize=20):
 
    import cma
    mutable = np.asarray(mutable)
    tr = _Tracker(problem)
    base = problem.original if init is None else init
    x0 = (base[mutable] - problem.original[mutable]).ravel()
    tr.update(base[None], problem.evaluate(base[None]))
    while tr.evals < budget - 10:
        es = cma.CMAEvolutionStrategy(
            x0, sigma, {"seed": int(rng.integers(1, 2**31 - 1)), "verbose": -9,
                        "maxfevals": budget - tr.evals, "popsize": popsize})
        while not es.stop() and tr.evals < budget:
            X = np.asarray(es.ask())
            pop = np.repeat(base[None], len(X), axis=0)
            off = X.reshape(len(X), len(mutable), 3)
            pop[:, mutable] = problem.repair_drift(problem.original[mutable] + off, mutable)
            ev = problem.evaluate(pop)
            tr.update(pop, ev)
            es.tell(list(X), list(ev["key"]))
        x0 = (tr.best[mutable] - problem.original[mutable]).ravel()
    return tr.best, tr


OPTIMISERS = {
    "GA": genetic_algorithm,
    "Random search": random_search,
    "(1+20)-ES": one_plus_lambda_es,
    "CMA-ES": cma_es,
}



def polish(problem, lab, mutable, passes=2, grid=33):
   
    if problem.evaluate(lab)["S"] > 0:
        return lab, 0
    cur = lab.copy()
    evals = 0
    ts_coarse = np.linspace(0.0, 1.0, grid)
    for _ in range(passes):
        drift = problem.evaluate(cur)["drift"]
        for i in sorted(mutable, key=lambda c: -drift[c]):
            o, c = problem.original[i], cur[i]
            if cm.ciede2000(o, c) < 1e-9:
                continue
            lo, hi = 0.0, 1.0
            for ts in (ts_coarse, None):
                if ts is None:
                    ts = np.linspace(lo, hi, grid)
                trial = np.repeat(cur[None], len(ts), axis=0)
                trial[:, i] = cm.gamut_map(o + ts[:, None] * (c - o))
                ok = problem.evaluate(trial)["S"] == 0
                evals += len(ts)
                first = int(np.argmax(ok))              # smallest feasible t
                hi = ts[first]
                lo = ts[max(first - 1, 0)]
            cur[i] = cm.gamut_map(o + hi * (c - o))
    return cur, evals



def repair_palette(problem, optimiser="GA", seed=0, budget=10_000,
                   cover="exact", use_polish=True, escalate=True, **opt_kwargs):
    
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    edges = problem.conflict_edges()
    res = dict(n_conflict_pairs=len(edges), n_conflicts=len(problem.conflicts()))

    if not edges:                                   # C1: already accessible
        best, mutable, hist, evals, first, esc = problem.original, [], [], 0, 0, 0
    else:
        mutable = {"exact": problem.exact_cover, "greedy": problem.greedy_cover,
                   "all": lambda: list(range(problem.k))}[cover]()
        res["cover_size"] = len(mutable)
        run = OPTIMISERS[optimiser]
        best, tr = run(problem, mutable, rng, budget=budget, **opt_kwargs)
        hist, evals, first, esc = list(tr.history), tr.evals, tr.first_feasible, 0
        while (escalate and problem.objective == "lexicographic"
               and problem.evaluate(best)["S"] > 0):
            add = problem.escalation_colour(best, mutable)
            if add is None:
                break
            mutable = sorted(mutable + [add])
            esc += 1
            best, tr = run(problem, mutable, rng, budget=budget, **opt_kwargs)
            hist += [(e + evals, s, d) for e, s, d in tr.history]
            if first is None and tr.first_feasible is not None:
                first = evals + tr.first_feasible
            evals += tr.evals
        if use_polish and problem.objective == "lexicographic":
            best, pe = polish(problem, best, mutable)
            evals += pe

    final = cm.quantise_lab(best)
    final[[i for i in range(problem.k) if i not in mutable]] = \
        problem.original[[i for i in range(problem.k) if i not in mutable]]
    ev = problem.evaluate(final, margin=0.0, observers=("normal", "protan", "deutan", "tritan"))
    res.update(
        final_lab=final, hex=cm.lab_to_hex(final),
        worst=float(ev["worst"]), min_obs=ev["min_obs"].tolist(),
        accessible=bool(ev["worst"] >= problem.delta - 1e-9),
        max_drift=float(ev["drift"].max()), total_drift=float(ev["D"]),
        faithful=bool(ev["drift"].max() <= problem.tau + 1e-9),
        n_modified=int((ev["drift"] > 1e-6).sum()), mutable=list(mutable),
        escalations=esc, evals=int(evals), evals_to_feasible=first,
        runtime=time.perf_counter() - t0, history=hist,
    )
    if problem.ordered:
        res["order_kept"] = bool(np.all(np.sign(np.diff(final[:, 0]))
                                        == problem._orig_dL_sign))
    res["success"] = res["accessible"] and res["faithful"]
    return res
