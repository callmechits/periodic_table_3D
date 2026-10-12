"""Verification suite for the orbital / atom modules.   Run:  python test_orbitals.py [--full]

Every check compares the code against an exact result (analytic formula, normalisation,
textbook value) or against an independent method. scipy, mendeleev and scikit-image are used for
extra cross-checks only if they are installed; --full also builds all 78 pickable meshes."""
import sys, time, math, importlib
import numpy as np
import hydrogenic as H, atomdata as A, sampler as S, isosurface as I, orbital_view as V

FAILS, T0 = [], time.time()
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok: FAILS.append(name)

def optional(modname):
    try: return importlib.import_module(modname)
    except ImportError: print(f"[skip] cross-check needing {modname} (not installed)"); return None

rng = np.random.default_rng(2024)

# ---------------- radial functions ----------------
x = np.linspace(0, 40, 801)
def laguerre_series(k, a, x):                      # explicit sum, independent of the recurrence
    return sum((-1) ** i * math.comb(k + a, k - i) * x ** i / math.factorial(i) for i in range(k + 1))
err = max(np.max(np.abs(H.genlaguerre(k, a, x) - laguerre_series(k, a, x)) / (1 + np.abs(laguerre_series(k, a, x))))
          for k in range(8) for a in range(1, 9))
check("Laguerre recurrence == explicit series (k<=7, alpha<=8)", err < 1e-9, f"max rel err {err:.1e}")
sp = optional("scipy.special")
if sp:
    err = max(np.max(np.abs(H.genlaguerre(k, a, x) - sp.eval_genlaguerre(k, a, x)) / (1 + np.abs(sp.eval_genlaguerre(k, a, x))))
              for k in range(8) for a in range(1, 9))
    check("Laguerre recurrence == scipy.special.eval_genlaguerre", err < 1e-9, f"max rel err {err:.1e}")

worst = worst_r = worst_r2 = 0.0
for n in range(1, 8):
    for l in range(n):
        r = H.radial_grid(n, 1.0, 200001); p = H.R(n, l, r) ** 2 * r ** 2
        worst = max(worst, abs(np.trapezoid(p, r) - 1))
        worst_r = max(worst_r, abs(np.trapezoid(p * r, r) / H.r_mean(n, l) - 1))
        worst_r2 = max(worst_r2, abs(np.trapezoid(p * r * r, r) / H.r2_mean(n, l) - 1))
check("R_nl normalised (all n<=7, l<n)", worst < 1e-10, f"max |norm-1| {worst:.1e}")
check("<r> = (3n^2 - l(l+1))/2Z and <r^2> formulas", max(worst_r, worst_r2) < 1e-10, f"max rel err {max(worst_r, worst_r2):.1e}")
Z = 3.7
r = H.radial_grid(4, Z, 200001)
check("Z scaling: R_nl(r; Z) normalised and <r> ~ 1/Z", abs(np.trapezoid(H.R(4, 2, r, Z) ** 2 * r ** 2, r) - 1) < 1e-10
      and abs(np.trapezoid(H.R(4, 2, r, Z) ** 2 * r ** 3, r) / H.r_mean(4, 2, Z) - 1) < 1e-10)
worst = max(abs(np.trapezoid(H.R(n1, l, rr := np.linspace(0, 400, 400001)) * H.R(n2, l, rr) * rr ** 2, rr))
            for l in range(4) for n1 in range(l + 1, 8) for n2 in range(n1 + 1, 8))
check("R_n1l orthogonal to R_n2l (Laguerre orthogonality)", worst < 1e-9, f"max overlap {worst:.1e}")
bad = 0
for n in range(1, 8):
    for l in range(n):
        rr = np.linspace(1e-3, H.radial_quantile(n, l, 1.0, 1 - 1e-9), 200001)
        bad += np.count_nonzero(np.diff(np.sign(H.R(n, l, rr))) != 0) != n - l - 1
check("radial node count == n - l - 1", bad == 0)
# the normalisation printed on note 13 uses (n+l)!^3, which belongs to the older Laguerre convention
n, l = 3, 2; r = H.radial_grid(n, 1.0, 200001)
if sp:
    Rn = math.sqrt((2 / n) ** 3 * math.factorial(n - l - 1) / (2 * n * math.factorial(n + l) ** 3)) * np.exp(-2 * r / n / 2) \
         * (2 * r / n) ** l * sp.eval_genlaguerre(n - l - 1, 2 * l + 1, 2 * r / n)
    check("note-13 factorial^3 with modern Laguerre is NOT normalised (ratio 1/(n+l)!^2)",
          abs(np.trapezoid(Rn ** 2 * r ** 2, r) * math.factorial(n + l) ** 2 - 1) < 1e-8)

# ---------------- angular functions ----------------
xg, wg = np.polynomial.legendre.leggauss(24); ph = np.linspace(0, 2 * np.pi, 48, endpoint=False)
CT, PH = np.meshgrid(xg, ph, indexing="ij"); ST = np.sqrt(1 - CT ** 2)
W = wg[:, None] * np.ones_like(PH) * (2 * np.pi / 48); u = (ST * np.cos(PH), ST * np.sin(PH), CT)
worst = 0
for l1 in range(4):
    for l2 in range(4):
        G = np.einsum("aij,bij,ij->ab", H.real_ylm(l1, *u), H.real_ylm(l2, *u), W)
        worst = max(worst, np.max(np.abs(G - (np.eye(2 * l1 + 1) if l1 == l2 else 0))))
check("real Y_lm orthonormal on the sphere (l<=3, all pairs)", worst < 1e-12, f"max dev {worst:.1e}")
v = rng.normal(size=(1_000_000, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
check("rejection envelope bounds |Y|^2 for all 16 orbitals",
      all((H.ylm(l, mi, v[:, 0], v[:, 1], v[:, 2]) ** 2).max() <= S._ang_max(l, mi) for l in range(4) for mi in range(2 * l + 1)))
check("ylm() == real_ylm() component-wise", all(np.allclose(H.ylm(l, mi, *u), H.real_ylm(l, *u)[mi]) for l in range(4) for mi in range(2 * l + 1)))
# whole wavefunction: integral |psi|^2 d^3r = 1 on a 3D grid (spherical shells x sphere quadrature)
rg = np.linspace(0, H.radial_quantile(3, 2, 1.0, 1 - 1e-10), 4001)
tot = 0
for (n, l, mi) in [(3, 2, 0), (3, 2, 3), (2, 1, 2), (3, 0, 0)]:
    rg = np.linspace(0, H.radial_quantile(n, l, 1.0, 1 - 1e-10), 4001)
    f = np.array([np.einsum("ij,ij->", H.psi(n, l, mi, ri * u[0], ri * u[1], ri * u[2]) ** 2, W) * ri ** 2 for ri in rg])
    tot = max(tot, abs(np.trapezoid(f, rg) - 1))
check("integral |psi_nlm(x,y,z)|^2 d^3r = 1 (Cartesian psi)", tot < 1e-8, f"max |I-1| {tot:.1e}")

# ---------------- per-element data ----------------
check("every configuration sums to Z (118 elements)", all(sum(A.config(z).values()) == z for z in range(1, 119)))
def madelung(z):
    cfg, left = {}, z
    for n, l in A.AUFBAU:
        t = min(left, 2 * (2 * l + 1)); cfg[(n, l)] = t; left -= t
    return {k: v for k, v in cfg.items() if v}
exc = [z for z in range(1, 104) if A.config(z) != madelung(z)]
check("20 Madelung exceptions up to Z=103", len(exc) == 20, str(exc))
md = optional("mendeleev")
if md:
    import re
    def parse(ec):
        out = {}
        for tok in ec.split():
            m = re.fullmatch(r"\[(\w+)\]", tok)
            if m:
                for k, v in parse(md.element(m.group(1)).econf).items(): out[k] = out.get(k, 0) + v
            else:
                m = re.fullmatch(r"(\d)([spdf])(\d*)", tok); k = (int(m.group(1)), "spdf".index(m.group(2)))
                out[k] = out.get(k, 0) + int(m.group(3) or 1)
        return {k: v for k, v in out.items() if v}
    bad = [z for z in range(1, 119) if parse(md.element(z).econf) != A.config(z)]
    check("all 118 configurations == mendeleev reference", not bad, f"mismatches {bad}")
text = {("C", 6, 2, 1): 3.25, ("N", 7, 2, 1): 3.90, ("O", 8, 2, 1): 4.55, ("F", 9, 2, 1): 5.20, ("Ne", 10, 2, 1): 5.85,
        ("Na", 11, 3, 0): 2.20, ("Cl", 17, 3, 1): 6.10, ("K", 19, 4, 0): 2.20, ("Fe3d", 26, 3, 2): 6.25, ("Fe4s", 26, 4, 0): 3.75,
        ("Cu3d", 29, 3, 2): 7.85, ("Cu4s", 29, 4, 0): 3.70, ("H", 1, 1, 0): 1.00, ("He", 2, 1, 0): 1.70}
check("Slater Z_eff == textbook values (14 cases)", max(abs(A.slater_zeff(z, n, l) - val) for (_, z, n, l), val in text.items()) < 1e-12)
check("hydrogen: Z_eff = 1 for every pickable (n, l)", all(abs(A.slater_zeff(1, n, l) - 1) < 1e-12 for n in range(1, 8) for l in range(min(n, 4))))
mn = min(A.slater_zeff(z, n, l) for z in range(1, 119) for n in range(1, 8) for l in range(min(n, 4)))
check("Z_eff >= 1 for every element x pickable orbital", mn >= 1 - 1e-12, f"min {mn:.3f}")

# ---------------- samplers ----------------
st = optional("scipy.stats")
def exact_radial_cdf(n, l, Z):
    rg = np.linspace(0, H.radial_grid(n, Z)[-1], 400001); p = H.R(n, l, rg, Z) ** 2 * rg ** 2
    F = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(rg))]); return rg, F / F[-1]
worst_z = 0
for (n, l, Z) in [(1, 0, 1.0), (2, 1, 1.0), (3, 2, 1.0), (4, 3, 1.0), (6, 0, 2.2), (7, 1, 8.25)]:
    xs = S.sample_radial(n, l, Z, 400000, rng)
    worst_z = max(worst_z, abs(xs.mean() - H.r_mean(n, l, Z)) / (xs.std() / np.sqrt(len(xs))))
    if st:
        rg, F = exact_radial_cdf(n, l, Z)
        assert st.kstest(xs, lambda t: np.interp(t, rg, F)).pvalue > 1e-3, (n, l, Z)
check("radial sampler: <r> within 4 sigma of analytic; KS vs exact CDF", worst_z < 4, f"worst z-score {worst_z:.2f}")
if st:
    xg2, wg2 = np.polynomial.legendre.leggauss(40)
    def expected(l, mi, nb=12):
        e = np.zeros((nb, nb))
        for i in range(nb):
            a, b = -1 + 2 * i / nb, -1 + 2 * (i + 1) / nb; ct = .5 * (b - a) * xg2 + .5 * (b + a); w = .5 * (b - a) * wg2
            for j in range(nb):
                c, d = 2 * np.pi * j / nb, 2 * np.pi * (j + 1) / nb; pp = .5 * (d - c) * xg2 + .5 * (d + c); wp = .5 * (d - c) * wg2
                C2, P2 = np.meshgrid(ct, pp, indexing="ij"); S2 = np.sqrt(1 - C2 ** 2)
                e[i, j] = np.einsum("ab,a,b", H.ylm(l, mi, S2 * np.cos(P2), S2 * np.sin(P2), C2) ** 2, w, wp)
        return e
    ps = []
    for l in range(4):
        for mi in range(2 * l + 1):
            N = 600000; uu = S.sample_angular(l, mi, N, rng); phi = np.mod(np.arctan2(uu[:, 1], uu[:, 0]), 2 * np.pi)
            obs, _, _ = np.histogram2d(uu[:, 2], phi, bins=[12, 12], range=[[-1, 1], [0, 2 * np.pi]]); e = expected(l, mi) * N; m = e > 5
            ps.append(1 - st.chi2.cdf(np.sum((obs[m] - e[m]) ** 2 / e[m]), m.sum() - 1))
    check("angular sampler: chi-square vs |Y|^2 for all 16 orbitals (Fisher-combined p)",
          st.combine_pvalues(ps, method="fisher")[1] > 1e-3, f"min p {min(ps):.3f}, combined {st.combine_pvalues(ps, method='fisher')[1]:.3f}")
# 2p_z against its exact z^2 CDF: P(|cos th| <= a) = a^3  ->  P(z^2 <= t) = integral p(r) min(1, (sqrt(t)/r)^3) dr
rg = np.linspace(1e-9, H.radial_grid(2, 1.0)[-1], 1_000_001); pr = H.R(2, 1, rg) ** 2 * rg ** 2; pr /= np.trapezoid(pr, rg)
for name, pts in [("separable", S.sample_orbital(2, 1, 2, 1.0, 1_000_000, rng)[0]), ("3D rejector", S.mc_rejection_3d(2, 1, 2, 1.0, 400_000, rng))]:
    z2 = np.sort(pts[:, 2] ** 2); grid = np.quantile(z2, np.linspace(0.001, 0.999, 200))
    ex = np.array([np.trapezoid(pr * np.minimum(1.0, (np.sqrt(t) / rg) ** 3), rg) for t in grid])
    d = np.max(np.abs(np.searchsorted(z2, grid) / len(z2) - ex))
    check(f"2p_z z^2 distribution vs exact CDF ({name})", d < 1.8 / np.sqrt(len(z2)), f"max |ECDF-exact| {d:.5f}")

# ---------------- isosurface ----------------
r90 = H.radial_quantile(1, 0, 1.0, 0.9); m = I.OrbitalMesh(1, 0, 0, 1.0)
s_ = I.mesh_stats(m.verts, m.faces)
check("1s 90% surface is a closed sphere (Euler 2)", s_["closed"] == 1.0 and s_["euler"] == 2)
rr = np.linalg.norm(m.verts, axis=1) / r90 - 1
check("1s surface: every vertex within 1 % of the exact radius", np.abs(rr).max() < 0.01, f"max {np.abs(rr).max():.2%}, rms {np.sqrt((rr ** 2).mean()):.2%}")
check("1s surface area / volume vs exact sphere (within 2 %)",
      abs(s_["area"] / (4 * np.pi * r90 ** 2) - 1) < .02 and abs(s_["volume"] / (4 / 3 * np.pi * r90 ** 3) - 1) < .02,
      f"area {s_['area'] / (4 * np.pi * r90 ** 2) - 1:+.2%}, volume {s_['volume'] / (4 / 3 * np.pi * r90 ** 3) - 1:+.2%}")
for (n, l, mi, euler, name, why) in [(2, 1, 2, 4, "2pz", "2 lobes"), (3, 2, 3, 8, "3dxy", "4 lobes"), (4, 3, 3, 16, "4fxyz", "8 lobes"),
                                     (3, 2, 4, 8, "3dx2-y2", "4 lobes"), (3, 2, 0, 4, "3dz2", "2 caps + 1 torus"), (3, 0, 0, 10, "3s", "5 nested spheres")]:
    mm = I.OrbitalMesh(n, l, mi, 1.0); ss = I.mesh_stats(mm.verts, mm.faces)
    check(f"{name} mesh is watertight with the right topology ({why}, Euler {euler})", ss["closed"] == 1.0 and ss["euler"] == euler, f"Euler {ss['euler']}")
fr = []
for (n, l, mi, Zz) in [(2, 1, 2, 1.0), (3, 2, 0, 1.0), (3, 2, 3, 1.0), (4, 3, 3, 1.0), (3, 0, 0, 1.0), (2, 0, 0, 1.0), (5, 2, 4, 3.3)]:
    mm = I.OrbitalMesh(n, l, mi, Zz); pts, _ = S.sample_orbital(n, l, mi, Zz, 600000, np.random.default_rng(77))
    fr.append(np.mean(H.psi(n, l, mi, pts[:, 0], pts[:, 1], pts[:, 2], Zz) ** 2 > mm.level))
check("surfaces enclose 90 % of the probability (independent Monte Carlo, 7 orbitals)", max(abs(f - 0.9) for f in fr) < 0.004,
      f"range {min(fr):.4f}..{max(fr):.4f}")
worst = 0
for (n, l, mi) in [(1, 0, 0), (2, 1, 2), (3, 2, 0), (3, 2, 4), (4, 3, 3)]:
    fine = I.mesh_stats(*[getattr(I.OrbitalMesh(n, l, mi, 1.0, N=101), k) for k in ("verts", "faces")])
    coarse = I.OrbitalMesh(n, l, mi, 1.0); cs = I.mesh_stats(coarse.verts, coarse.faces)
    worst = max(worst, abs(cs["area"] / fine["area"] - 1)); assert cs["euler"] == fine["euler"], (n, l, mi)
check("production-grid area within 3 % of an N=101 reference, same topology (5 orbitals)", worst < .03, f"worst {worst:.2%}")
nrm_ok = all(abs(np.linalg.norm(mm.normals, axis=1) - 1).max() < 1e-9 for mm in [I.OrbitalMesh(3, 2, 0, 1.0)])
check("shading normals are unit length", nrm_ok)

# whole atoms and the scene
t = time.time(); ok = True; slow = 0
for z in range(1, 119):
    t1 = time.time(); d = S.atom_cloud(z); slow = max(slow, time.time() - t1)
    ok &= bool(np.isfinite(d["points"]).all() and abs(d["weight"].sum() - z) < 1e-6 and d["fit_radius"] > 0)
check("atom_cloud for all 118 elements: finite, weights sum to Z", ok, f"slowest {slow * 1000:.0f} ms")

seen = set(); sc = V.OrbitalScene(26)

for _ in range(7):
    sc.cycle_n()
    for _ in range(4):
        sc.cycle_l()
        for _ in range(7): sc.cycle_m(); seen.add((sc.n, sc.l, sc.mi))

check("picker reaches all 78 orbitals with n<=7, l<=3", len(seen) == 78 == sum(2 * l + 1 for n in range(1, 8) for l in range(min(n, 4))))

if "--full" in sys.argv:
    bad = []
    for (n, l, mi) in sorted(seen):
        mm = I.OrbitalMesh(n, l, mi, A.slater_zeff(26, n, l)); ss = I.mesh_stats(mm.verts, mm.faces)
        if ss["closed"] < 0.999 or len(mm.faces) == 0: bad.append((n, l, mi))
    check("all 78 pickable meshes build and are closed (Z=26 Z_eff)", not bad, str(bad))

print(f"\n{len(FAILS)} failed, {time.time() - T0:.0f} s")
sys.exit(1 if FAILS else 0)
