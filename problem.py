


from itertools import combinations

import numpy as np

import colour_math as cm


class Problem:
    def __init__(self, original_lab, delta=10.0, tau=10.0,
                 observers=cm.OBSERVERS, margin=0.3, ordered=False,
                 objective="lexicographic"):
        self.original = np.asarray(original_lab, dtype=np.float64)
        self.k = len(self.original)
        self.delta, self.tau = float(delta), float(tau)
        self.observers = tuple(observers)
        self.margin = float(margin)
        self.ordered = ordered              # preserve lightness order (ramps)
        self.objective = objective          # "lexicographic" or "maximin"
        self.pi, self.pj = np.triu_indices(self.k, 1)
        self.max_drift_total = self.k * self.tau + 1.0
        self._orig_dL_sign = np.sign(np.diff(self.original[:, 0]))

    
    def evaluate(self, pop, margin=None, observers=None):
        """

        Returns a dict of arrays with leading dimension N:
            dists   (N, O, P)  pairwise dE00 per observer
            min_obs (N, O)     worst pair per observer
            worst   (N,)       worst pair over all observers
            S       (N,)       accessibility shortfall (0 = accessible)
            drift   (N, k)     per-colour dE00 from the original
            D       (N,)       total drift
            key     (N,)       scalar ranking key, lower is better
        """
        pop = np.asarray(pop, dtype=np.float64)
        single = pop.ndim == 2
        if single:
            pop = pop[None]
        margin = self.margin if margin is None else margin
        observers = self.observers if observers is None else observers

        lin = cm.lab_to_linear(pop)
        dists = []
        for o in observers:
            lab_o = pop if o == "normal" else cm.linear_to_lab(cm.simulate_linear(lin, o))
            dists.append(cm.ciede2000(lab_o[:, self.pi], lab_o[:, self.pj]))
        dists = np.stack(dists, axis=1)                       # (N, O, P)

        min_obs = dists.min(axis=2)
        worst = min_obs.min(axis=1)
        S = np.maximum(0.0, self.delta + margin - dists).sum(axis=(1, 2))
        if self.ordered:
            dL = np.diff(pop[:, :, 0], axis=1)
            S = S + np.maximum(0.0, -self._orig_dL_sign * dL).sum(axis=1)

        drift = cm.ciede2000(self.original[None], pop)          # (N, k)
        D = drift.sum(axis=1)

        if self.objective == "maximin":
            # Original draft formulation: maximise the worst pair (drift is
            # capped by repair, so the old penalty term was always zero).
            key = -worst
        else:
            
            key = np.where(S > 0, self.max_drift_total + S, D)

        out = dict(dists=dists, min_obs=min_obs, worst=worst, S=S,
                   drift=drift, D=D, key=key)
        if single:
            out = {n: v[0] for n, v in out.items()}
        return out

 
    def conflicts(self, lab=None):
        """All (i, j, observer, dE) with dE < delta, using the true delta."""
        lab = self.original if lab is None else lab
        ev = self.evaluate(lab, margin=0.0)
        out = []
        for o_idx, o in enumerate(self.observers):
            for p, d in enumerate(ev["dists"][o_idx]):
                if d < self.delta:
                    out.append((int(self.pi[p]), int(self.pj[p]), o, float(d)))
        return out

    def conflict_edges(self, lab=None):
        return sorted({(i, j) for i, j, _, _ in self.conflicts(lab)})

    def exact_cover(self):
       
        edges = self.conflict_edges()
        if not edges:
            return []
        weight = np.zeros(self.k)
        for i, j, _, d in self.conflicts():
            weight[i] += self.delta - d
            weight[j] += self.delta - d
        for size in range(1, self.k + 1):
            covers = [set(c) for c in combinations(range(self.k), size)
                      if all(i in c or j in c for i, j in edges)]
            if covers:
                best = max(covers, key=lambda c: (sum(weight[list(c)]), -min(c)))
                return sorted(best)
        return list(range(self.k))

    def greedy_cover(self):
       
        remaining = set(self.conflict_edges())
        cover = set()
        while remaining:
            deg = {}
            for i, j in remaining:
                deg[i] = deg.get(i, 0) + 1
                deg[j] = deg.get(j, 0) + 1
            c = max(deg, key=deg.get)
            cover.add(c)
            remaining = {e for e in remaining if c not in e}
        return sorted(cover)

    def escalation_colour(self, lab, mutable):
       
        ev = self.evaluate(lab)
        short = np.maximum(0.0, self.delta + self.margin - ev["dists"]).sum(axis=0)
        load = np.zeros(self.k)
        np.add.at(load, self.pi, short)
        np.add.at(load, self.pj, short)
        load[list(mutable)] = 0.0
        return int(np.argmax(load)) if load.max() > 0 else None

   
    def repair_drift(self, cand, idx, grid=48, refine=16):
        
        cand = cm.gamut_map(cand)
        orig = self.original[idx]
        limit = self.tau - self.margin
        bad = cm.ciede2000(orig, cand) > limit
        if not bad.any():
            return cand
        start, end = np.broadcast_to(orig, cand.shape)[bad], cand[bad]
        step = end - start
        lo, hi = np.zeros(len(end)), np.ones(len(end))
        for n in (grid, refine):
            ts = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, n + 1)[None, 1:]
            c = cm.gamut_map(start[:, None] + ts[..., None] * step[:, None])
            ok = cm.ciede2000(start[:, None], c) <= limit
            last = n - 1 - np.argmax(ok[:, ::-1], axis=1)        # largest ok step
            found = ok.any(axis=1)
            new_lo = np.where(found, ts[np.arange(len(ts)), last], lo)
            hi = np.where(found & (last < n - 1),
                          ts[np.arange(len(ts)), np.minimum(last + 1, n - 1)],
                          np.where(found, hi, ts[:, 0]))
            lo = new_lo
        cand = cand.copy()
        cand[bad] = cm.gamut_map(start + lo[:, None] * step)
        return cand
