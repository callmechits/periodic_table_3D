"""Angular shapes for s/p/d/f clouds; only radial part"""
import numpy as np

AUFBAU = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1), (5, 0), (4, 2), (5, 1), (6, 0), (4, 3), (5, 2), (6, 1), (7, 0), (5, 3), (6, 2), (7, 1)]
LETTER = "spdf"
COL = {0: (255, 128, 0), 1: (0, 128, 255), 2: (60, 200, 60), 3: (200, 0, 200)}      # blue orange green magenta for spdf
PTS_PER_E = 28          # cloud points per electron

def subshell_config(z):
    """Gives out [(n, 1, electrons)] for the occupied subshells"""
    out, left = [], z
    for n, l in AUFBAU:
        take = min(left, 2 * (2 * l + 1))
        if take:
            out.append((n, l, take))
        left -= take
        if left == 0:
            break
    return out

def _angular(l, x, y, z):
    """Real angular functions which uses one array per orbital"""
    if l == 0: return[np.ones_like(x)]
    if l == 1: return[x, y, z]
    if l == 2: return[3 * z * z - 1, x * z, y * z, x * y, x * x - y * y]
    return [z * (5 * z * z - 3), x * (5 * z * z - 1), y * (5 * z * z - 1), x * y * z, z * (x * x - y * y), x * (x * x - 3 * y * y), y * (3 * x * x - y * y)]

def config_lines(z, per_line = 8):
    """Electron configuration to split display lines"""
    toks = [f"{n}{LETTER[1]}{e}" for n, l, e in sorted(subshell_config(z))]
    return[" ".join(toks[i:i + per_line]) for i in range(0, len(toks), per_line)]

def build_cloud(z, r_inner, r_step, seed = 0):
    """Return points and colors in the model's pixel units"""
    rng = np.random.default_rng(seed)
    v = rng.normal(size = (20000, 3))
    v /= np.linalg.norm(v, axis = 1, keepdims = True)
    pts, cols = [], []
    for n, l, e in subshell_config(z):
        funcs = _angular(l, v[:, 0], v[:, 1], v[:, 2])
        m = len(funcs)
        occ = [1] * min(e, m)
        for j in range(max(0, e - m)):
            occ[j] += 1
        r_n = r_inner + (n - 1) * r_step
        for f, k in zip(funcs, occ):
            w = f * f
            idx = rng.choice(len(w), size = PTS_PER_E * k, p = w / w.sum())
            r = np.clip(r_n * (1 + 0.15 * rng.normal(size = len(idx))), 0.4 * r_n, 1.6 * r_n)
            pts.append(v[idx] * r[:, None])
            cols.append(np.tile(COL[l], (len(idx), 1)))
    return np.concatenate(pts), np.concatenate(cols).astype(np.uint8)
