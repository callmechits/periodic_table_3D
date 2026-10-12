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
