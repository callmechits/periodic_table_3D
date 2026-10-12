"""Monte Carlo sampling of hydrogenic orbitals and of whole atoms.

|psi|^2 d^3r = [R^2 r^2 dr] * [|Y|^2 dOmega]  is separable, so:
  radial  : invert the numerical CDF F(r) = integral_0^r R^2 r'^2 dr'   (exact, no rejection)
  angular : rejection against the max-envelope of |Y|^2 on the sphere
  3D      : brute-force rejection in a cube, kept as an independent cross-check.
"""
import numpy as np
from functools import lru_cache
import hydrogenic as H
import atomdata as A


# --------------------------------------------------------------------------
# Separable sampler
# --------------------------------------------------------------------------
@lru_cache(maxsize=256)
def _radial_table(n, l, Z):
    """Cached (r grid, CDF) for the orbital; Z is rounded by the caller for cache hits."""
    return H.radial_cdf(n, l, Z)


@lru_cache(maxsize=64)
def _ang_max(l, mi):
    return H.angular_max(l, mi)


def sample_radial(n, l, Z, N, rng):
    """N radii drawn from R_nl^2 r^2 by inverse-CDF sampling."""
    r, f = _radial_table(n, l, round(Z, 6))
    return np.interp(rng.random(N), f, r)


def sample_angular(l, mi, N, rng):
    """N unit vectors distributed as |Y_lm|^2 (rejection against the max envelope)."""
    out, have = [], 0
    m_env = _ang_max(l, mi)
    accept = max((1.0 / (4.0 * np.pi)) / m_env, 1e-3)      # mean of |Y|^2 over the sphere is 1/4pi
    while have < N:
        batch = int((N - have) / accept * 1.25) + 64
        v = rng.normal(size=(batch, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        f = H.ylm(l, mi, v[:, 0], v[:, 1], v[:, 2]) ** 2
        keep = rng.random(batch) * m_env < f
        out.append(v[keep])
        have += int(keep.sum())
    return np.concatenate(out)[:N]


def sample_orbital(n, l, mi, Z, N, rng):
    """Points (N, 3) in Bohr radii distributed as |psi_nlm|^2, plus the sign of psi at each point."""
    r = sample_radial(n, l, Z, N, rng)
    u = sample_angular(l, mi, N, rng)
    pts = u * r[:, None]
    sign = np.sign(H.psi(n, l, mi, pts[:, 0], pts[:, 1], pts[:, 2], Z))
    return pts, sign


# --------------------------------------------------------------------------
# Brute-force 3D Monte Carlo rejector (independent cross-check)
# --------------------------------------------------------------------------
def mc_rejection_3d(n, l, mi, Z, N, rng, q=1 - 1e-4, batch=2_000_000):
    """Sample [x, y, z] uniformly in the cube [-rmax, rmax]^3 and accept when u <= rho(x, y, z)
    with u ~ U(0, rho_max). rho_max = max(R^2) * max(Y^2) is an exact bound because the
    radial and angular factors vary independently. rmax encloses a fraction q of the density
    (the discarded tail is 1 - q)."""
    rmax = H.radial_quantile(n, l, Z, q)
    r = np.linspace(0, rmax, 20001)
    rho_max = float((H.R(n, l, r, Z) ** 2).max()) * _ang_max(l, mi) * 1.001
    out, have = [], 0
    while have < N:
        p = rng.uniform(-rmax, rmax, size=(batch, 3))
        rho = H.psi(n, l, mi, p[:, 0], p[:, 1], p[:, 2], Z) ** 2
        keep = rng.random(batch) * rho_max <= rho
        out.append(p[keep])
        have += int(keep.sum())
    return np.concatenate(out)[:N]


# --------------------------------------------------------------------------
# Whole atom: rho(r) = sum over occupied orbitals of |psi_i|^2
# --------------------------------------------------------------------------
def atom_orbitals(z):
    """Occupied real orbitals of element z as a list of dicts:
    n, l, mi, occ (1 or 2 electrons), Z_eff (Slater)."""
    cfg = A.config(z)
    out = []
    for (n, l), e in sorted(cfg.items()):
        zeff = A.slater_zeff(z, n, l, cfg)
        for mi, occ in enumerate(A.hund_occupancy(l, e)):
            if occ:
                out.append(dict(n=n, l=l, mi=mi, occ=occ, zeff=zeff))
    return out


def atom_cloud(z, total_points=24000, min_per_orbital=800, seed=0):
    """Point cloud of the whole atom: every electron is sampled from its own Slater-screened
    hydrogenic orbital. Sampling is stratified: each occupied orbital gets the same number of
    samples k (so even a lone valence electron gives a smooth cloud) and every sample carries the
    weight occupancy / k, so the weighted density is exactly rho = sum_occupied occ_i |psi_i|^2.
    Returns dict(points (N,3) in a0, l (N,), weight (N,), fit_radius in a0)."""
    rng = np.random.default_rng(seed)
    orbs = atom_orbitals(z)
    k = max(min_per_orbital, int(np.ceil(total_points / len(orbs))))
    pts, ls, ws, fit = [], [], [], 0.0
    for orb in orbs:
        p, _ = sample_orbital(orb["n"], orb["l"], orb["mi"], orb["zeff"], k, rng)
        pts.append(p)
        ls.append(np.full(k, orb["l"]))
        ws.append(np.full(k, orb["occ"] / k))                 # weights sum to the electron count Z
        # Display scale: the outermost orbital's 95th-percentile radius must fit the screen.
        fit = max(fit, H.radial_quantile(orb["n"], orb["l"], orb["zeff"], 0.95))
    return dict(points=np.concatenate(pts), l=np.concatenate(ls),
                weight=np.concatenate(ws), fit_radius=fit)
