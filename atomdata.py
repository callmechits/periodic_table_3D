"""Per-element data: ground-state electron configurations and Slater's-rules Z_eff.

Configurations follow the Madelung rule (fill by increasing n + l, lower n first) with the
~20 known exceptions up to Z = 103 kept in an explicit table, as in the project notes.
Z >= 104 are theoretical predictions (and relativistic effects, ignored here, are large).
"""
from hydrogenic import L_LETTERS

# Madelung filling order as (n, l): sorted by n + l, ties broken by lower n first.
AUFBAU = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1), (5, 0), (4, 2),
          (5, 1), (6, 0), (4, 3), (5, 2), (6, 1), (7, 0), (5, 3), (6, 2), (7, 1)]
