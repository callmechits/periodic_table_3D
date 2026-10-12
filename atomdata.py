"""Per-element data: ground-state electron configurations and Slater's-rules Z_eff.

Configurations follow the Madelung rule (fill by increasing n + l, lower n first) with the
~20 known exceptions up to Z = 103 kept in an explicit table, as in the project notes.
Z >= 104 are theoretical predictions (and relativistic effects, ignored here, are large).
"""
from hydrogenic import L_LETTERS

# Madelung filling order as (n, l): sorted by n + l, ties broken by lower n first.
AUFBAU = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1), (5, 0), (4, 2),
          (5, 1), (6, 0), (4, 3), (5, 2), (6, 1), (7, 0), (5, 3), (6, 2), (7, 1)]

# Subshell occupancies that differ from the Madelung prediction: {Z: {(n, l): electrons}}.
# Only the changed subshells are listed; everything else stays as filled by Madelung.
EXCEPTIONS = {
    24: {(3, 2): 5, (4, 0): 1},            # Cr  3d5 4s1
    29: {(3, 2): 10, (4, 0): 1},           # Cu  3d10 4s1
    41: {(4, 2): 4, (5, 0): 1},            # Nb
    42: {(4, 2): 5, (5, 0): 1},            # Mo
    44: {(4, 2): 7, (5, 0): 1},            # Ru
    45: {(4, 2): 8, (5, 0): 1},            # Rh
    46: {(4, 2): 10, (5, 0): 0},           # Pd  4d10
    47: {(4, 2): 10, (5, 0): 1},           # Ag
    57: {(4, 3): 0, (5, 2): 1},            # La  5d1
    58: {(4, 3): 1, (5, 2): 1},            # Ce  4f1 5d1
    64: {(4, 3): 7, (5, 2): 1},            # Gd  4f7 5d1
    78: {(5, 2): 9, (6, 0): 1},            # Pt  5d9 6s1
    79: {(5, 2): 10, (6, 0): 1},           # Au
    89: {(5, 3): 0, (6, 2): 1},            # Ac  6d1
    90: {(5, 3): 0, (6, 2): 2},            # Th  6d2
    91: {(5, 3): 2, (6, 2): 1},            # Pa
    92: {(5, 3): 3, (6, 2): 1},            # U
    93: {(5, 3): 4, (6, 2): 1},            # Np
    96: {(5, 3): 7, (6, 2): 1},            # Cm
    103: {(6, 2): 0, (7, 1): 1},           # Lr  7p1
    # Predicted (relativistic calculations; not measured): same pattern as Pt and Au.
    110: {(6, 2): 9, (7, 0): 1},           # Ds  6d9 7s1
    111: {(6, 2): 10, (7, 0): 1},          # Rg  6d10 7s1
}
NOBLE_CORES = [(86, "Rn"), (54, "Xe"), (36, "Kr"), (18, "Ar"), (10, "Ne"), (2, "He")]

def config(z):
    """Ground-state configuration as {(n, l): electrons}, only occupied subshells."""
    cfg, left = {}, z
    for n, l in AUFBAU:                                # Madelung fill
        take = min(left, 2 * (2 * l + 1))
        cfg[(n, l)] = take
        left -= take
    cfg.update(EXCEPTIONS.get(z, {}))                  # apply the known exceptions
    cfg = {k: e for k, e in cfg.items() if e > 0}
    assert sum(cfg.values()) == z, f"configuration for Z={z} does not sum to Z"
    return cfg

def config_text(z):
          """Compact configuration with a noble-gas core, e.g. 'Fe' -> '[Ar] 3d6 4s2'."""
          cfg = config(z)
          core_label = ""
          for zc, sym in NOBLE_CORES:
                    if zc < z:
                              core = config(zc)
                              if all(cfg.get(k, 0) >= e for k, e in core.items()):   # core still intact
                                        for k, e in core.items():
                                                  cfg[k] = cfg.get(k, 0) - e
                                        core_label = f"[{sym}] "
                                        break
          toks = [f"{n}{L_LETTERS[l]}{e}" for (n, 1), e in sorted(cfg.items()) if e > 0]
          return core_label + " ".join(toks)

def hund_occupancy(1, e):
          """Electrons per real orbital of a subshell (length 2l+1): one in each orbital first
          (Hund's rule, parallel spins), then pair up. Open subshells therefore keep directional lobes."""
          m = 2 * l + 1
          occ = [0] * m
          for i in range(e):
                    occ[i % m] += 1
          return occ

# ---------------------------------------------
# Slater's rules
# ---------------------------------------------
def _group(n, 1):
          """Slater group key, ordered left to right: (1s)(2sp)(3sp)(3d)(4sp)(4d)(4f)(5sp)..."""
          return (n, 0 if l <= 1 else l -1)

def slater_sigma(z, n, l, cfg = None):
          """Screening constant sigma for an electron in subshell (n, l) of element z."""
          cfg = config(z) if cfg is None else cfg          # NB: {} is a valid (empty) configuration
          g_self = _group(n, 1)
          sigma = 0.0
          for (m, k), e in cfg.items():
                    g = _group(m, k)
                    if g == g_self:          # same group: 0.35 each (0.30 for 1s)
                              others = e - (1 if (m, k) == (n, l) else 0)          # do not count the electron itself
                              sigma += others * (0.30 if n == 1 else 0.35)
                    elif l <= 1:          # s/p electron
                              if m == n - 1:
                                        sigma += 0.85 * e          # next shell in
                              elif m <= n - 2:                  
                                        sigma += 1.00 * e          # deeper shells
                              # shells with m >= n in other groups lie to the right: no screening
                    elif g < g_self
                              sigma += 1.00 * e
          return sigma

def slater_zeff(z, n, l, cfg = None):
          """Effective nuclear charge Z_eff = Z - sigma for subshell (n, l).

          For an unoccupied subshell the orbital is treated as a Rydberg-like excited state: one
          electron is promoted out of the outermost occupied subshell into it, so Z - 1 electrons
          screen it. Far-out orbitals then see Z_eff -> 1, and hydrogen gives exactly 1 for every n."""
          cfg = dict(config(z) if cfg is None else cfg)
          if cfg.get((n, l), 0) == 0:
                    donor = max(cfg)
                    cfg[donor] -= 1
                    if cfg[donor] == 0
                              del cfg[donor]
          return z - slater_sigma(z, n, l, cfg)

def default_orbital(z):
          """The outermost occupied subshell (highest n, then highest l): the valence shell
          shown first in the orbital picker."""
          n, l = max(config(z))
          return n, l
