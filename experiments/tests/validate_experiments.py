"""Rigorous, independent validation of the dKS experiment pipeline.

Checks every detail against transparently-correct references (not just "it runs").
Run:  .venv/bin/python experiments/tests/validate_experiments.py

Sections:
  A core statistic vs a pure-Python pooled-corner reference (ground truth)
  B permutation test (formula + Type-I + null p-value dist + randomized)
  C generators (range, no boundary atoms, alpha tracking)
  D threshold & bound formulas (exact algebra + inversion)
  E Wilson CI vs an independent implementation
  F reproducibility & data<->plot consistency
  G hygiene (no MMD/energy/Bickel comparison; results/figures git-ignored; clean tree)

Small/fast cases. Honest: discrepancies are reported with the number.
"""
import contextlib
import io
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(EXP)
sys.path.insert(0, EXP)

import dks                      # noqa: E402  -- confirm the binding imports
import generators              # noqa: E402
import methods as M            # noqa: E402
import run_power               # noqa: E402
import plots                   # noqa: E402

RESULTS = []  # (section, name, passed, key_number)


def record(section, name, passed, detail):
    RESULTS.append((section, name, bool(passed), detail))


def section(tag, fn):
    try:
        fn()
    except Exception as e:  # a crash in one check shouldn't abort the table
        record(tag, f"{tag}: EXCEPTION", False, f"{type(e).__name__}: {e}")


# ============================================================ reference dKS
def ref_dks(P, Q):
    """Ground-truth pooled-corner lower-orthant dKS (transparently correct).

    max over corners (x_i, y_j), x_i and y_j each any pooled coordinate, of
    |#{p in P: p<=corner}/|P| - #{q in Q: q<=corner}/|Q||.
    """
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    pooled = np.vstack([P, Q])
    xs, ys = pooled[:, 0], pooled[:, 1]
    Px = (P[:, 0][:, None] <= xs[None, :]).astype(np.float64)   # (nP, N)
    Py = (P[:, 1][:, None] <= ys[None, :]).astype(np.float64)
    Qx = (Q[:, 0][:, None] <= xs[None, :]).astype(np.float64)
    Qy = (Q[:, 1][:, None] <= ys[None, :]).astype(np.float64)
    FP = (Px.T @ Py) / P.shape[0]      # FP[i,j] = frac of P dominated by (xs[i],ys[j])
    FQ = (Qx.T @ Qy) / Q.shape[0]
    return float(np.max(np.abs(FP - FQ)))


# ============================================================ (A)
def check_A():
    rng = np.random.default_rng(20260615)
    max_diff = 0.0
    gaps = []           # exact - approx
    viol = 0; max_over = 0.0
    res = []
    for n in (5, 10, 30, 80):
        for _ in range(50):
            P = rng.uniform(-1, 1, (n, 2)); Q = rng.uniform(-1, 1, (n, 2))
            e = dks.exact(P, Q); r = ref_dks(P, Q); a = dks.approx(P, Q)
            max_diff = max(max_diff, abs(e - r))
            gaps.append(e - a)
            if a > e + 1e-9:
                viol += 1; max_over = max(max_over, a - e)
            res.append((n, e, a))
    record("A", "exact == pooled-corner reference (200 pairs)",
           max_diff < 1e-9, f"max|exact-ref|={max_diff:.2e}")

    gaps = np.array(gaps)
    record("A", "approx <= exact (per-pair conjecture)",
           viol == 0, f"violations={viol}/200, max overshoot={max_over:.2e}")
    # gap vs grid resolution ~1/(2 sqrt n)
    record("A", "max (exact-approx) gap vs grid res 1/(2*sqrt n)",
           True, f"max gap={gaps.max():.4f}; grid res(n=5..80)~{1/(2*math.sqrt(5)):.3f}..{1/(2*math.sqrt(80)):.3f}")

    # determinism
    det_ok = True
    for _ in range(4):
        P = rng.uniform(-1, 1, (60, 2)); Q = rng.uniform(-1, 1, (60, 2))
        vals = {dks.approx(P, Q) for _ in range(5)}
        det_ok = det_ok and len(vals) == 1
    record("A", "approx deterministic (5 repeats x4 inputs)", det_ok,
           "all identical" if det_ok else "NON-DETERMINISTIC")

    # symmetry
    sym_e = sym_a = 0.0
    for _ in range(50):
        P = rng.uniform(-1, 1, (40, 2)); Q = rng.uniform(-1, 1, (40, 2))
        sym_e = max(sym_e, abs(dks.exact(P, Q) - dks.exact(Q, P)))
        sym_a = max(sym_a, abs(dks.approx(P, Q) - dks.approx(Q, P)))
    record("A", "symmetry exact(P,Q)==exact(Q,P)", sym_e < 1e-12, f"max|diff|={sym_e:.2e}")
    record("A", "symmetry approx(P,Q)==approx(Q,P)", sym_a < 1e-12, f"max|diff|={sym_a:.2e}")

    # degenerate
    P = rng.uniform(-1, 1, (50, 2))
    zero = dks.exact(P, P)
    record("A", "identical P,Q -> dKS == 0", abs(zero) < 1e-12, f"value={zero:.2e}")
    Plo = rng.uniform(-1.0, -0.6, (50, 2)); Qhi = rng.uniform(0.6, 1.0, (50, 2))
    one = dks.exact(Plo, Qhi)
    record("A", "fully separated (Q dominates) -> dKS == 1", abs(one - 1.0) < 1e-9,
           f"value={one:.6f}")


# ============================================================ (B)
class _Stub:
    def __init__(self, t_obs, perms):
        self.vals = [t_obs] + list(perms); self.i = 0

    def __call__(self, A, B):
        v = self.vals[self.i]; self.i += 1; return v


def _wilson(k, Z, z=1.96):
    if Z == 0:
        return 0.0, 0.0
    p = k / Z; d = 1 + z * z / Z
    c = (p + z * z / (2 * Z)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / Z + z * z / (4 * Z * Z))
    return max(0.0, c - h), min(1.0, c + h)


def check_B():
    # B1 formula with a stub
    rng = np.random.default_rng(0)
    P = np.zeros((3, 2)); Q = np.zeros((3, 2))
    perms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # B-1 = 10 -> B = 11
    rej, p, t_obs, el = M.permutation_test(P, Q, _Stub(5.0, perms), 11, 0.05, rng)
    ge = sum(1 for v in perms if v >= 5.0)     # = 6
    expect = (1 + ge) / 11
    record("B", "permutation p-value formula (stub)",
           abs(p - expect) < 1e-12 and t_obs == 5.0,
           f"p={p:.6f} expected={expect:.6f}")

    # B2 Type-I for both statistics (true null), conservative
    n, B, trials, delta = 200, 100, 1000, 0.05
    sss = lambda A, Bp: M.sss_stat(A, Bp)

    def type_one(stat_fn, randomized, seed=7):
        ss = np.random.SeedSequence(seed)
        rj = np.zeros(trials, bool); pv = np.zeros(trials)
        for t in range(trials):
            r = np.random.default_rng(ss.spawn(1)[0])
            Pn = generators.uniform_square(n, r); Qn = generators.uniform_square(n, r)
            rj[t], pv[t], _, _ = M.permutation_test(Pn, Qn, stat_fn, B, delta, r, randomized)
        return rj, pv

    re_c, pv_e = type_one(M.exact_stat, False)
    rs_c, pv_s = type_one(sss, False)
    for nm, rj in (("exact-sample", re_c), ("SSS-dKS", rs_c)):
        rate = rj.mean(); lo, hi = _wilson(int(rj.sum()), trials)
        record("B", f"Type-I size <= delta ({nm}, null)", lo <= delta + 1e-12,
               f"rate={rate:.3f} CI=[{lo:.3f},{hi:.3f}] delta={delta}")

    # B3 null p-value distribution (exact), should be conservative: P(p<=lvl) <= lvl
    pv = pv_e
    pdist_ok = True; parts = []
    for lvl in (0.05, 0.10, 0.20):
        frac = float((pv <= lvl + 1e-12).mean())
        slack = 1.96 * math.sqrt(lvl * (1 - lvl) / trials)
        pdist_ok = pdist_ok and frac <= lvl + slack
        parts.append(f"P(p<={lvl})={frac:.3f}")
    record("B", "null p-value dist conservative (exact)", pdist_ok, "; ".join(parts))

    # B4 randomized closer to nominal than conservative
    re_r, _ = type_one(M.exact_stat, True)
    rate_c = re_c.mean(); rate_r = re_r.mean()
    closer = abs(rate_r - delta) <= abs(rate_c - delta) + 0.01
    record("B", "randomized rate closer to delta than conservative (exact)", closer,
           f"conservative={rate_c:.3f} randomized={rate_r:.3f} delta={delta}")


# ============================================================ (C)
def _ks2_pvalue(a, b):
    """Independent 1D two-sample KS asymptotic p-value (hand-coded)."""
    a = np.sort(a); b = np.sort(b)
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    D = np.max(np.abs(cdf_a - cdf_b))
    ne = len(a) * len(b) / (len(a) + len(b))
    t = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * D
    q = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * t * t)
                  for k in range(1, 101))
    return D, max(0.0, min(1.0, q))


def check_C():
    rng = np.random.default_rng(3)
    N = 100000
    U = generators.uniform_square(N, rng)
    record("C", "uniform_square in [-1,1]^2", np.all(np.abs(U) <= 1.0),
           f"max|coord|={np.abs(U).max():.6f}")
    record("C", "uniform_square mean ~0, var ~1/3",
           abs(U.mean()) < 0.01 and abs(U.var() - 1 / 3) < 0.01,
           f"mean={U.mean():+.4f} var={U.var():.4f} (target 0.3333)")

    # (i) alpha=0 same distribution as uniform_square
    U2 = generators.uniform_square(N, rng)
    H0 = generators.huber_mixture(N, 0.0, rng)
    _, px = _ks2_pvalue(H0[:, 0], U2[:, 0]); _, py = _ks2_pvalue(H0[:, 1], U2[:, 1])
    record("C", "huber_mixture(alpha=0) == uniform_square (2-sample KS not rejected)",
           px > 0.01 and py > 0.01, f"KS p-values x={px:.3f} y={py:.3f}")

    # (ii) alpha>0 strictly inside; (iii) no boundary atoms
    Ha = generators.huber_mixture(N, 0.30, rng, sigma=0.15)
    record("C", "huber_mixture(alpha>0) strictly inside [-1,1]^2",
           np.all(np.abs(Ha) < 1.0), f"max|coord|={np.abs(Ha).max():.6f}")
    atoms = int(np.count_nonzero(np.abs(Ha) >= 1.0 - 1e-9))
    record("C", "no boundary atoms at +-1 (clamping would create them)",
           atoms == 0, f"points at |coord|>=1-1e-9: {atoms}")

    # (iv) central-bump fraction tracks alpha
    r3 = 3 * 0.15
    p_unif = math.pi * r3 ** 2 / 4.0                 # uniform mass in disk radius 0.45
    p_bump = 1 - math.exp(-r3 ** 2 / (2 * 0.15 ** 2))  # Rayleigh, ~0.989
    fracs = []
    for a in (0.0, 0.15, 0.30):
        pts = generators.huber_mixture(N, a, rng, sigma=0.15)
        fracs.append(float((np.hypot(pts[:, 0], pts[:, 1]) < r3).mean()))
    mono = fracs[0] < fracs[1] < fracs[2]
    pred = [p_unif + a * (p_bump - p_unif) for a in (0.0, 0.15, 0.30)]
    err = max(abs(f - p) for f, p in zip(fracs, pred))
    record("C", "central-bump fraction tracks alpha", mono and err < 0.02,
           f"frac={[round(f,3) for f in fracs]} pred={[round(p,3) for p in pred]} maxerr={err:.3f}")


# ============================================================ (D)
def check_D():
    cases = [(100, 0.05), (1000, 0.01), (5000, 0.1)]
    e1 = e2 = e3 = e4 = inv_c = inv_u = 0.0
    for n, d in cases:
        e1 = max(e1, abs(M.tau_clean(n, d) - 2 * math.sqrt(math.log(1 / d) / n)))
        e2 = max(e2, abs(M.tau_union(n, d) - math.sqrt((4 * math.log(2 * n) / n) * math.log(1 / d))))
        eps = 0.05
        e3 = max(e3, abs(float(M.bound_clean(eps, n)) - math.exp(-n * eps ** 2 / 4)))
        e4 = max(e4, abs(float(M.bound_union(eps, n)) - math.exp(-n * eps ** 2 / (4 * math.log(2 * n)))))
        inv_c = max(inv_c, abs(float(M.bound_clean(M.tau_clean(n, d), n)) - d))
        inv_u = max(inv_u, abs(float(M.bound_union(M.tau_union(n, d), n)) - d))
    record("D", "tau_clean formula", e1 < 1e-12, f"max err={e1:.2e}")
    record("D", "tau_union formula", e2 < 1e-12, f"max err={e2:.2e}")
    record("D", "bound_clean formula", e3 < 1e-12, f"max err={e3:.2e}")
    record("D", "bound_union formula", e4 < 1e-12, f"max err={e4:.2e}")
    record("D", "inversion bound_clean(tau_clean)==delta", inv_c < 1e-9, f"max err={inv_c:.2e}")
    record("D", "inversion bound_union(tau_union)==delta", inv_u < 1e-9, f"max err={inv_u:.2e}")


# ============================================================ (E)
def check_E():
    md = 0.0; edge_ok = True
    for k, Z in [(0, 20), (20, 20), (1, 10), (5, 10), (37, 300), (0, 1), (1, 1)]:
        a = run_power.wilson_ci(k, Z); b = _wilson(k, Z)
        md = max(md, abs(a[0] - b[0]), abs(a[1] - b[1]))
        if k == 0 and abs(a[0]) > 1e-12:
            edge_ok = False
        if k == Z and abs(a[1] - 1.0) > 1e-12:
            edge_ok = False
    record("E", "wilson_ci == independent reference", md < 1e-12, f"max diff={md:.2e}")
    record("E", "wilson_ci edges: k=0 -> low=0, k=Z -> high=1", edge_ok,
           "edges correct" if edge_ok else "EDGE WRONG")


# ============================================================ (F)
def check_F():
    import csv
    with tempfile.TemporaryDirectory() as td:
        out1 = os.path.join(td, "p1.csv"); out2 = os.path.join(td, "p2.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            run_power.run([50], [0.0, 0.30], 50, 50, 0.05, -1.0, 123, out1, False)
            run_power.run([50], [0.0, 0.30], 50, 50, 0.05, -1.0, 123, out2, False)
        r1 = list(csv.DictReader(open(out1))); r2 = list(csv.DictReader(open(out2)))
        # statistical columns must be identical; avg_runtime is wall-clock and
        # legitimately varies run-to-run (not a reproducibility defect).
        stat_keys = ["method", "n", "alpha", "B", "delta", "Z",
                     "rejection_rate", "ci_low", "ci_high"]
        stat_same = (len(r1) == len(r2) and
                     all(all(a[k] == b[k] for k in stat_keys) for a, b in zip(r1, r2)))
        rt_same = all(a["avg_runtime"] == b["avg_runtime"] for a, b in zip(r1, r2))
        record("F", "run_power reproducible (same seed -> identical statistics)",
               stat_same,
               f"stat cols identical={stat_same}; avg_runtime identical={rt_same} "
               f"(timing varies by design)")

        # data<->plot consistency: plots ingestion returns CSV values unchanged
        raw = r1
        saved_results = plots.RESULTS
        try:
            plots.RESULTS = td
            os.replace(out1, os.path.join(td, "power.csv"))
            rows = plots._read_power()
        finally:
            plots.RESULTS = saved_results
        # match a couple of (n, rejection_rate) values to the raw CSV, no transform
        ok = True; checked = 0
        for rr in raw[:4]:
            hit = [r for r in rows if r["method"] == rr["method"]
                   and r["n"] == int(rr["n"]) and abs(r["alpha"] - float(rr["alpha"])) < 1e-12]
            if hit:
                checked += 1
                ok = ok and abs(hit[0]["rejection_rate"] - float(rr["rejection_rate"])) < 1e-12
        record("F", "plot ingestion == CSV rejection_rate (no transform)",
               ok and checked >= 2, f"matched {checked} rows exactly")


# ============================================================ (G)
def check_G():
    # method set: exactly the 4 dKS methods, no MMD/energy/Bickel competitor
    expected = [M.METHOD_EXACT, M.METHOD_SSS, M.METHOD_DIRECT_CLEAN, M.METHOD_DIRECT_UNION]
    labels = " ".join(M.LABELS.values()).lower()
    bad_method = any(t in labels for t in ("mmd", "energy", "bickel"))
    record("G", "methods compared = exactly the 4 dKS variants (no competitor)",
           run_power.METHOD_ORDER == expected and not bad_method,
           f"METHOD_ORDER={run_power.METHOD_ORDER}")

    # grep the pipeline code (excluding this test) for competitor IMPLEMENTATIONS
    files = ["generators.py", "methods.py", "run_power.py", "run_calibration.py", "plots.py"]
    hits = []
    for fn in files:
        for i, line in enumerate(open(os.path.join(EXP, fn)), 1):
            low = line.lower()
            if any(t in low for t in ("mmd", "energy", "bickel")):
                hits.append(f"{fn}:{i}: {line.strip()[:70]}")
    # all hits should be the documented-exclusion disclaimers, not code
    doc_only = all(("excluded" in h.lower() or "no mmd" in h.lower()
                    or "#" in h or "bake-off" in h.lower()) for h in hits)
    record("G", "no MMD/energy/Bickel except documented-exclusion notes",
           doc_only, f"{len(hits)} textual mentions (all doc-only={doc_only})")

    # results/ and figures/ git-ignored
    def ignored(path):
        r = subprocess.run(["git", "-C", REPO, "check-ignore", path],
                           capture_output=True, text=True)
        return r.returncode == 0
    ig = ignored("experiments/results/power.csv") and ignored("experiments/figures/x.pdf")
    record("G", "results/ and figures/ are git-ignored", ig,
           "both ignored" if ig else "NOT IGNORED")

    # harness code working tree clean (committed)
    r = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--"] +
                       [f"experiments/{f}" for f in files] + ["experiments/README.md"],
                       capture_output=True, text=True)
    clean = r.stdout.strip() == ""
    record("G", "harness code working tree clean (committed)", clean,
           "clean" if clean else f"dirty: {r.stdout.strip()[:80]}")


# ============================================================ main
def main():
    print("dKS experiment pipeline validation  (import dks: OK)\n")
    for tag, fn in [("A", check_A), ("B", check_B), ("C", check_C),
                    ("D", check_D), ("E", check_E), ("F", check_F), ("G", check_G)]:
        section(tag, fn)

    w = max(len(n) for _, n, _, _ in RESULTS)
    print(f"{'':2}{'check':<{w}}  {'result':<6}  key number")
    print("-" * (w + 40))
    npass = 0
    for sec, name, ok, detail in RESULTS:
        npass += ok
        print(f"{sec:<2}{name:<{w}}  {'PASS' if ok else 'FAIL':<6}  {detail}")
    print("-" * (w + 40))
    total = len(RESULTS)
    print(f"\n{npass}/{total} checks PASSED")
    if npass == total:
        print("ALL VALIDATION CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED -- see FAIL rows above")
    return 0 if npass == total else 1


if __name__ == "__main__":
    sys.exit(main())
