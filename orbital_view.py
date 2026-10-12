"""Orbital scene for the AR periodiic table (no dependency on bohr.py).

Two views, both driven by the same camera dolly / pinch zoom / rotation as the Bohr view:
  "atom"    : the whole atom. Every electron is sampled from its own Slater-screened hydrogenic
              orbital, so the point density is rho(r) = sum over occupied orbitals of |psi_i|^2.
              Open subshells fill distinct orbitals first (Hund), so they show directional lobes.
  "orbital" : one hydrogenic orbital (n, l, m) picked with three cycle buttons, drawn as
              dots (|psi|^2 samples), a shaded 90% - probability surface, or its wireframe.
              Colour = sign (phase) of psi.
"""
import numpy as np
import cv2
import hydrogenic as H
import atomdata as A
import sampler as S
import isosurface as I

FIT_PX = 185.0                    # screen radius (px at zoom 1) of the displayed object's fit radius (95th percentile)
CLOUD_SAMPLES = 40000             # samples for a single-orbital density image
C_OUTLINE = (20, 20, 20)
COL_L = {0: (255, 128, 0), 1: (0, 128, 255), 2: (60, 200, 60), 3: (200, 0, 200)}    # BGR: s p d f
STYLES = ["surface", "mesh", "cloud"]
STYLE_LABEL = {"surface": "SURF", "mesh": "MESH", "cloud": "CLOUD"}

# Short names for the picker buttons (full names are in H.ORBITAL_NAMES, shown in the HUD).
SHORT_NAMES = {0: ["s"], 1: ["px", "py", "pz"], 2: ["dz2", "dxz", "dyz", "dxy", "dx2y2"], 3: ["fz3", "fxz2", "fyz2", "fxyz", "fzxy", "fx3", "fy3"]}


_BUILDS = I.MeshCache()            # background worker shared by every scene (meshes, clouds, dots)
_dot_cache = {}


# ------------------------------------
# Builders (run on the worker thread)
# ------------------------------------
def _build_cloud(n, l, mi, zeff):
    rng = np.random.default_rng(1000 * n + 100 + l + mi)
    pts, sign = S.sample_orbital(n, l, mi, zeff, CLOUD_SAMPLES, rng)
    return dict(points = pts, sign = sign, fit = H.radial_quantile(n, l, zeff, 0.95))


# ------------------------------------
# Vectorised point drawing
# ------------------------------------
def draw_density(frame, base_px, colours, centre, Rg, spread, focal, proj, weights = None, sigma = 1.7, decades = 4.0):
    """Render weighted samples as a smoothed density image (a projected electron-density map).
    Samples are splatted into a half-resolution grid, Gaussian-blurred, mapped through a fixed
    logarithmic scale spanning 'decades' orders of magnitude below the peak (so a compact core and 
    a diffuse valence cloud are visible together, and the picture does not flicker), and
    alpha-composited onto the frame. base_px: (N, 3) screen-pixel units; colours: (N, 3) BGR."""

    pts = (base_px * spread) @ Rg.T
    xy, s, valid = proj(pts, centre, focal)
    hgt, wid = frame.shape[:2]
    ix, iy = xy[:, 0].astype(int), xy[:, 1].astype(int)
    ok = valid & (ix >= 0) & (ix < wid) & (iy >= 0) & (iy < hgt)

    if not ok.any():
        return

    w = np.ones(int(ok.sum())) if weights is None else weights[ok]        # same length as the kept samples
    ix, iy, col = ix[ok], iy[ok], colours[ok].astype(np.float32)
    x0, x1 = max(iy.min() - 10, 0), min(ix.max() + 11, wid)
    y0, y1 = max(iy.min() - 10, 0), min(iy.max() + 11, hgt)
    w_, h_ = x1 - x0, y1 - y0
    w2, h2 = (w_ + 1) // 2, (h_ + 1) // 2            # accumulate at half resolution (4x cheaper lol)
    idx = ((iy - y0) // 2) * w2 + (ix - x0) // 2

    acc = np.zeros((h2, w2, 4), np.float32)          # channels: weighted B, G, R and total weight

    for c in range(3):
        acc[..., c] = np.bincount(idx, weights = w * col[:, c], minlength = h2 * w2).reshape(h2, w2)

    acc[..., 3] = np.bincount(idx, weights = w * col[:, c], minlength = h2 * w2).reshape(h2, w2)
    acc = cv2.GaussianBlur(acc, (0, 0), sigma)        # soft splat of every sample
    d = acc[..., 3]
    d0 = d.max() * 10.0 ** (-decades)                 # density at the bottom of the log scale
    a = np.clip(np.loglp(d / d0) / np.loglp(d.max() / d0), 0.0, 1.0)
    alpha = (0.93 * a).astype(np.float32)
    colour = acc[..., :3] / np.maximum(d[..., None], 1e-12) * 0.85     # weighted mean colour, slightly darkened
    alpha = cv2.resize(alpha, (w_, h_), interpolation = cv2.INTER_LINEAR)[..., None]
    colour = cv2.resize(colour, (w_, h_), interpolation = cv2.INTER_LINEAR)
    roi = frame[y0:y1, x0:x1].astype(np.float32) 
    frame[y0:y1, x0:x1] = (roi * (1 - alpha) + colour * alpha).astype(np.uint8)


def _label(frame, text, pos, scale = 0.5):
    """Black text; OpenCV needs integer pixel coordinates (centre arrives as a float array)."""
    cv2.putText(frame, text, (int(pos[0]), int(pos[1])), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)



# -------------------------------------------
# Scene
# -------------------------------------------
class OrbitalScene:
    def __init__(self, z):
        self.z = z
        self.cfg = A.config(z)
        self.n, self.l = A.default_orbital(z)        # start on the valence subshell
        self.mi = 0
        self.style = "surface"

    # picker state
    def _clamp(self):
        self.l = min(self.l, self.n - 1, H.MAX_L)
        self.mi = min(self.mi, 2 * self.l)

    def cycle_n(self):
        self.n = self.n % 7 + 1            # 1...7
        self._clamp()

    def cycle_l(self):
        self.l = (self.l + 1) % (min(self.n - 1, H.MAX_L) + 1)
        self.mi = 0

    def cycle_m(self):
        self.mi = (self.mi + 1) % (2 * self.l + 1)

    def cycle_style(self):
        self.style = STYLES[(STYLES.index(self.style) + 1) % len(STYLES)]

    @property
    def zeff(self):
        """Slater Z_eff felt by this subshell (also for unoccupied subshells)."""
        return A.slater_zeff(self.z, self.n, self.l, self.cfg)

    def picker_labels(self):
        return [f"n = {self.n}", f"l - {H.L_LETTERS[self.l]}", f"m = {SHORT_NAMES[self.l][self.mi]}"]

    def style_label(self):
        return STYLE_LABEL[self.style]

    # HUD text
    def hud_lines(self, mode):
        if mode == "atom":
            n, l = max(self.cfg)            # outermost occupied subshell
            ze = A.slater_zeff(self.z, n, l, self.cfg)        
            rA = H.r_mean(n, l, ze) * H.A0_ANGSTROM
            return [f"{self.z} e-: sum of |psi|^2 over occupied orbitals (Slater Zeff)", A.config_text(self.z), f"valence {n}{H.L_LETTERS[l]}: Zeff = {ze:.2f}  <r> = {rA:.2f} A"]
        n, l, mi, ze = self.n, self.l, self.mi, self.zeff
        occ = self.cfg.get((n, l), 0)
        occ_txt = f"occupied ({occ} e-)" if occ else "not occupied in the ground state"
        what = "density ~ |psi|^2" if self.style == "cloud" else "90% probability surface"
        return [f"'{n}{H.ORBITAL_NAMES[l][mi]}   Zeff= {ze:.2f}   {occ_txt}", f"radial nodes {n - l -1}, angular nodes {l}   <r>= {H.r_mean(n, l, ze) * H.A0_ANGSTROM:.2f} A", f"{what}; colour = sign of psi"]

    # drawing
    def draw(self, frame, mode, centre, Rg, spread, focal, proj):
        """Draw the current view. Heavy data is built on a worker thread; until it is ready a 
        cheaper stand-in (or a status line) is shown so the camera loop never stalls."""
        if mode == "atom":
            data = _BUILDS.get(("atom", self.z), lambda z = self.z: S.atom_cloud(z))
            if data is None:
                _label(frame, "sampling electron density...", (centre[0] - 110, centre[1]))
                return
            px = FIT_PX / data["fit_radius"]
            col = np.array([COL_L[int(l)] for l in range(4)], dtype = np.uint8)[data["l"]]
            draw_density(frame, data["points"] * px, col, centre, Rg, spread, focal, proj, weights = data["weight"])
            return

        n, l, mi, ze = self.n, self.l, self.mi, round(self.zeff, 4)
        key = (n, l, mi, ze)
        dots = _BUILDS.get(("cloud",) + key, lambda: _build_cloud(n, l, mi, ze))
        if dots is None:
            _label(frame, "sampling orbital...", (centre[0] - 80, centre[1]))
            return
        px = FIT_PX / dots["fit"]
        mesh = None
        if self.style != "cloud":
            mesh = _BUILDS.get(("mesh",) + key, lambda: I.OrbitalMesh(n, l, mi, ze))
        if mesh is None:
            col = np.where(dots["sign"][:, None] >= 0, np.array(I.C_POS, np.uint8), np.array(I.C_NEG, np.uint8))
            draw_density(frame, dots["points"] * px, col, centre, Rg, spread, focal, proj)
            if self.style != "cloud":
                _label(frame, "building surface...", (centre[0] - 80, centre[1] + 215), 0.45)
        elif self.style == "surface":
            I.draw_surface(frame, mesh, px, centre, Rg, spread, focal, proj)
        else:
            I.draw_wireframe(frame, mesh, px, centre, Rg, spread, focal, proj)


# --------------------------------------
# UI helpers (left-hand picker column, legends)
# --------------------------------------
def picker_rects(w, h):
    """Three stacked cycle buttons under the PREV button on the left edge."""
    y0 = h // 2 + 30
    return [(8, y0 + 24 * i, 96, y0 + 24 * i + 21) for i in range(3)]


def draw_picker_buttons(frame, rects, labels):
    for (x0, y0, x1, y1), text in zip(rects, labels):
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), 1)
        _label(frame, text + " >", (x0 + 7, y0 + 15), 0.45)


def draw_legend(frame, h, mode):
    """Colour key, bottom-left."""
    if mode == "atom":
        items = [("s orbitals", COL_L[0]), ("p orbitals", COL_L[1]), ("d orbitals", COL_L[2]), ("f orbitals", COL_L[3])]
    else:
        items = [("psi > 0", I.C_POS), ("psi < 0", I.C_NEG)]
    y_start = h - 12 - 16 * len(items)
    for i, (label, col) in enumerate(items):
        y = y_start + i * 16
        cv2.circle(frame, (18, y), 6, C_OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(frame, (18, y), 5, col, -1, cv2.LINE_AA)
        _label(frame, label, (30, y + 5), 0.45)
