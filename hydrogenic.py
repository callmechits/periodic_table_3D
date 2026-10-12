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
    "dz2": 
