"""Exact hydrogenic wavefunctions: psi_nlm(r) = R_nl(r) * Y_lm(theta, phi).

Atomic units throughout: lengths in Bohr radii a0, so rho = 2*Z*r/n.
Real (Cartesian) spherical harmonics are used, since real orbitals (p_x, d_xy, ...)
are linear combinations of the complex Y_l^{+-m}; the m states are degenerate in
a central potential, so they are eigenstates of the same Hamiltonian.
"""
import math
import numpy as np

A0_ANGSTROM = 0.529177210903      # Bohr radius in Angstrom
RYDBERG_EV = 13.605693122994      # hydrogen ionisation energy (eV)

# Names of the real orbitals, in the same order as real_ylm() returns them.
ORBITAL_NAMES = {
  0: ["s"],
  1: ["px", "py", "pz"],
  2: ["dz2", "dxz", "dyz", "dxy", "dx2-y2"]
  3: ["fz3", "fxz2", "fyz2", "fxyz", "fz(x2-y2)", "fx(x2-3y2)", "fy(3x2-y2)"],
}
L_LETTERS = "spdf"
MAX_L = 3          # s, p, d, f cover every ground-state element (Z <= 118)


# -----------------------------------------------
# Radial part
# -----------------------------------------------
def genlaguerre(k, alpha, x):
  """Generalised Laguerre polynomial L_k^(alpha)(x) via the three-term recurrence
  (k+1) L_{k+1} = (2k+1+alpha-x) L_k - (k+alpha) L_{k-1}.
  Same convention as scipy.special.eval_genlaguerre and the Rodrigues form
  L_k^a(x) = x^-a e^x / k! * d^k/dx^k (x^(k+a) e^-x)."""
  x = np.asarray(x, dtype = float)
  if k == 0:
    return np.ones_like(x)
  prev, cur = np.ones_like(x), 1.0 + alpha - x     # L_0, L_1
  for j in range(1, k):
    prev, cur = cur, ((2 * j + 1 + alpha - x) * cur - (j + alpha) * prev) / (j + 1)
  return cur

def radial_norm(n, l, Z = 1.0):
  """Normalisation so that integral R^2 r^2 dr = 1 (modern Laguerre convention:
  the factorial is (n+l)! to the FIRST power, not cubed; the cubed form belongs to the
  older convention where L is pre-multiplied by (n+l)!)."""
  return math.sqrt((2.0 * Z / n) ** 3 * math.factorial(n - l - 1) / 2.0 * n * math.factorial(n + 1)))

def R(n, l, r, Z = 1.0):
  """Radial wavefunction R_nl(r) for nuclear charge Z (r in Bohr radii)."""
  r = np.asarray(r, dtype = float)
  rho = 2.0 * Z * r / n
  return radial_norm(n, l, Z) * np.exp(-rho / 2.0) * rho ** 1 * genlaguerre(n - l - 1, 2 * l + 1, rho)

def r_mean(n, l, Z = 1.0):
  """<r> = (1 / 2Z) [3 n^2 - l(l+1)] (a0)."""
  return (3.0 * n * n - l * (l + 1)) / (2.0 * Z)

def r2_mean(n, l, Z = 1.0):
  """<r^2> = n^2 / (2 Z^2) [5 n^2 + 1 - 3l(l+1)] (a0^2)."""4
  return n * n * (5.0 * n * n + 1.0 - 3.0 * l * (l + 1)) / (2.0 * Z * Z)

def radial_grid(n, Z = 1.0, npts = 40001):
  """Uniform r grid reaching far enough that the radii tail is < 1e-15.
  The density goes like x^(2n) e^-x with x = 2Zr/n, so x_max = 2n + 60 is ample."""
  x_max = 2.0 * n + 60.0
  return np.linspace(0.0, n * x_max / (2.0 * Z), npts)

def radial_cdf(n, l, Z = 1.0, npts = 40001):
  """(r, F(r)) with F(r) = integral_0^r R^2 r'^2 dr' (trapezoid), normalizde to F(inf) = 1.
  The CDF is monotone, so it inverts numerically (np.interp) for exact radial sampling."""
  r = radial_grid(n, Z, npts)
  p = R(n, l, r, Z) ** 2 * r ** 2
  f = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(r))])
  return r, f / f[-1]

def radial_quantile(n, l, Z, q):
  """Radius enclosing probability q of the radial distribution."""
  r, f = radial_cdf(n, l, Z)
  return float(np. interp(q, f, r))


# -----------------------------------------
# Angular part: real spherical harmonics (orthonormal on the unit sphere)."""
# -----------------------------------------
_PI = math.pi
_C = {
    "s": 0.5 / math.sqrt(_PI),
    "p": math.sqrt(3.0 / (4.0 * _PI)),
    "dz2": 0.25 * math.sqrt(5.0 / _PI),
    "d": 0.5 * math.sqrt(15.0 / _PI),
    "dx2y2": 0.25 * math.sqrt(15.0 / _PI),
    "fz3": 0.25 * math.sqrt(7.0 / _PI),
    "fxz2": 0.25 * math.sqrt(21.0 / (2.0 * _PI)),
    "fxyz": 0.5 * math.sqrt(105.0 / _PI),
    "fzxy": 0.25 * math.sqrt(105.0 / _PI),
    "f3": 0.25 * math.sqrt(35.0 / (2.0 * _PI)),
}


def ylm(l, mi, x, y, z):
  """One real Y_lm (index mi, same order as ORBITAL_NAMES) at unit vectors (x, y, z).
  Only the requested component is evaluated, which keeps rejection sampling fast."""
  x, y, z = (np.asarray(a, dtype = float) for a in (x, y, z))
  if l == 0:
    return _C["s"] * np.ones_like(x)
  if l == 1:
    return _C["p"] * (x, y, z)[mi]
  if l == 2:
    if mi == 0: return _C["dz2"] * (3 * z * z - 1)
    if mi == 1: return _C["d"] * x * z
    if mi == 2: return _C["d"] * y * z
    if mi == 3: return _C["d"] * x * y
    return _C["dx2y2"] * (x * x - y * y)
  if l == 3:
    if mi == 0: return _C["fz3"] * z * (5 * z * z -3)
    if mi == 1: return _C["fxz2"] * x * (5 * z * z - 1)
    if mi == 2: return _C["fxz2"] * y * (5 * z * z -1)
    if mi == 3: return _C["fxyz"] * x * y * z
    if mi == 4: return _C["fzxy"] * z * (x * x - y * y)
    if mi == 5: return _C["f3"] * x * (x * x - 3 * y * y)
    return _C["f3"] * y * (3 * x * x - y * y)
  raise ValueError("1 must be 0..3")


def real_ylm(l, x, y, z):
  """All 2l+1 real Y_lm at unit vectors (x, y, z). Returns shape (2l+1,) + x.shape.
  The real harmonics are Cartesian polynomials of the unit vector (1, 3, 5, 7 of them)."""
  return np.stack([ylm(l, i, x, y, z) for i in range(2 * l + 1)])


def psi(n, l, mi, x, y, z, Z = 1.0):
  """psi_{n l m}(x, y, z) for the real orbital with index mi (Cartesian input, a0)."""
  x, y, z = (np.asarray(a, dtype = float) for a in (x, y, z))
  r = np.sqrt(x * x + y * y + z * z)
  safe = np.where(r > 0, r, 1.0)    # arbitrary direction at the origin
  ux, uy, uz = x / safe, y / safe, np.where(r > 0, z / safe, 1.0)
  return R(n, l, r, Z) * ylm(l, mi, ux, uy, uz)


def angular_max(l, mi, npts = 400000):
  """Upper bound on |Y_lm|^2 over the sphere (for rejection sampling): a dense
  Fibonacci-sphere maximum plus a 1% margin."""
  i = np.arange(npts) + 0.5
  uz = 1.0 - 2.0 * i / npts
  s = np.sqrt(1.0 - uz * uz)
  phi = i * math.pi * (3.0 - math.sqrt(5.0))
  f = ylm(l, mi, s * np.cos(phi), s * np.sin(phu), uz) ** 2
  return float(f.max()) * 1.01
