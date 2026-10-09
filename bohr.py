"""Module 4: simplified 3-D Bohr-model renderer (NumPy projection + OpenCV drawing).
Standalone test (no camera):  python bohr.py 26"""
import sys
import numpy as np
import cv2
from elements import ELEMENTS
from ionization import IE1

# ---- Tunable parameters -------------------------------------------------
NUC_CAP = 30                  # max nucleons drawn; heavier nuclei are scaled down
NUC_R = 4.2                   # nucleus radius = NUC_R * N^(1/3) px
R_INNER, R_STEP = 30, 24      # shell radii: n=1 -> 30 px ... n=7 -> 174 px
FOCAL = 400.0                 # perspective focal length (px)
PITCH = 0.45                  # fixed camera tilt (rad)
YAW_SPEED = 0.7               # whole-model spin (rad/s)
RING_PTS = 72                 # polyline resolution of each orbit ring
NEAR = 15.0
MAX_R = 120
RING_W = 2.5           # ring tube thickness in world units; drawn width = RING_W * perspective scale

# BGR colours; every particle gets a dark outline so it reads on bright backgrounds.
C_PROTON, C_NEUTRON, C_ELECTRON = (60, 60, 255), (170, 170, 170), (255, 140, 40)
C_OUTLINE = (20, 20, 20)
C_GREEN = (60, 220, 60)

# Aufbau (Madelung) filling order as (n, l); capacity of a subshell is 2(2l+1).
_ORDER = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1), (5, 0), (4, 2), (5, 1), (6, 0), (4, 3), (5, 2), (6, 1), (7, 0), (5, 3), (6, 2), (7, 1)]


def shell_occupancy(z):
    """Electrons per principal shell n = 1..7 via Aufbau filling.
    Ignores the known exceptions (Cr, Cu, Pd, ...)."""
    occ, left = [0] * 7, z
    for n, l in _ORDER:
        take = min(left, 2 * (2 * l + 1))
        occ[n - 1] += take
        left -= take
        if left == 0:
            break
    return occ


# ---- Rotation matrices (applied to row vectors: p' = p @ R.T) ------------
def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class BohrModel:
    def __init__(self, z):
        self.paused = False         # To freeze the orbital electron motion
        self.orbit_t = 0.0          # Electron clock (just to measure where each electron is)
        sym, name, a = ELEMENTS[z]
        self.z, self.a = z, a
        occ = shell_occupancy(z)
        self.info = (f"{sym} - {name}   Z={z}  A={a}  n={a - z}   "
                     f"shells {[c for c in occ if c]}")
        rng = np.random.default_rng(z)            # seeded: same element, same look

        # --- Nucleus: simplified for heavy elements, keeping the p:n ratio ---
        if a > NUC_CAP:
            total = NUC_CAP
            p = max(1, round(z * NUC_CAP / a))
        else:
            total, p = a, z
        self.is_proton = np.zeros(total, bool)
        self.is_proton[:p] = True
        rng.shuffle(self.is_proton)               # mix protons and neutrons
        d = rng.normal(size=(total, 3))           # random directions on the sphere
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        # cbrt(uniform) radius -> uniform density inside a ball
        self.nuc = d * (NUC_R * total ** (1 / 3)) * np.cbrt(rng.random((total, 1)))

        # --- Electron shells: each in its own tilted plane, own speed ---
        self.shells = []
        for n, count in enumerate(occ, start=1):
            if count == 0:
                continue
            self.shells.append(dict(
                r=R_INNER + (n - 1) * R_STEP,
                count=count,
                R=_rx(0.9 * n) @ _rz(0.6 * n),    # orbital-plane orientation
                omega=(2.2 / n) * (1 if n % 2 else -1),  # outer = slower, alternate sense
                phase=rng.random() * 2 * np.pi,
            ))
        self._ring = np.linspace(0, 2 * np.pi, RING_PTS)

        # --- Removing electrons ---
        self.detached = False
        self.hold = False
        self.green_xy, self.green_r = None, 0
        self.green_t, self._last_t = 0.0, 0.0

    def ion_text(self):
        """Info showing that electron is detached"""
        if not self.detached:
            return None
        sym, ie = ELEMENTS[self.z][0], IE1[self.z]
        tag = " (predicted)" if self.z >= 104 else ""
        return(f"{sym} -> {sym}+ + e-  IE1 = {ie:.2f} eV{tag}  " f"photon <= {1239.84 / ie:.0f} nm")

    @staticmethod
    def _project(pts, centre, focal):
        """Perspective-project (N,3) points -> (N,2) screen px, plus scale s."""
        denom = focal + pts[:, 2]            # larger z = farther = smaller
        valid = denom > NEAR                 # drop points at/behind the camera plane
        s = focal / np.where(valid, denom, 1.0)
        return centre + pts[:, :2] * s[:, None], s, valid
    
    def render(self, frame, centre, t, zoom = 1.00, yaw = None, pitch = PITCH, roll = 0.0):
        """Draw the model onto frame in place. t: seconds since the view opened."""
        centre = np.array(centre, dtype=float)
        Rg = _rz(roll) @_rx(pitch) @ _ry(YAW_SPEED * t if yaw is None else yaw)     # global tilt + spin
        spread = max(0.2, zoom)
        focal = FOCAL * spread

        dt = max(0.0, t - self._last_t)
        self._last_t = t
        if not self.paused:
            self.orbit_t += dt
            if not self.hold:
                self.green_t += dt
        self.green_xy = None

        #meow meow

        # 1) Orbit rings: constant stroke width; only the part in front of the camera.
        for sh in self.shells:
            r = sh["r"] * spread
            ring = np.stack([r * np.cos(self._ring), r * np.sin(self._ring), np.zeros(RING_PTS)], axis = 1)            
            xy, s, valid = self._project(ring @ sh["R"].T @ Rg.T, centre, focal)
            pts_i = [(int(x), int(y)) for x, y in xy]
            segs = []
            for k in range(RING_PTS):
                k2 = (k + 1) % RING_PTS
                if valid[k] and valid[k2]:
                    wd = min(60, max(1, int(round(RING_W * 0.5 * (s[k] + s[k2])))))
                    segs.append((pts_i[k], pts_i[k2], wd))

            for a, b, wd in segs:
                cv2.line(frame, a, b, C_OUTLINE, wd + 2, cv2.LINE_AA)
            for a, b, wd in segs:
                cv2.line(frame, a, b, (255, 255, 255), wd, cv2.LINE_AA)
                        #poly = xy[run].astype(np.int32).reshape(-1, 1, 2)
                        #cv2.polylines(frame, [poly], False, C_OUTLINE, RING_PX + 2, cv2.LINE_AA)
                        #cv2.polylines(frame, [poly], False, (255, 255, 255), RING_PX, cv2.LINE_AA)

        # 2) Collect all particles: positions, colours, base radii.
        pts = [(self.nuc * spread) @ Rg.T]
        cols = [[C_PROTON if p else C_NEUTRON for p in self.is_proton]]
        rads = [np.full(len(self.nuc), 3.2)]
        green_idx = None
        for k, sh in enumerate(self.shells):
            r = sh["r"] * spread
            ang = sh["phase"] + sh["omega"] * self.orbit_t + 2 * np.pi * np.arange(sh["count"]) / sh["count"]
            outer = (k == len(self.shells) - 1)
            if outer:
                ang[0] = sh["phase"] + sh["omega"] * self.green_t
            local = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros(sh["count"])], axis = 1)
            col = [C_ELECTRON] * sh["count"]
            rad = np.full(sh["count"], 4.0)
            if outer:
                if self.detached:
                    local, col, rad = local[1:], col[1:], rad[1:]
                else:
                    col[0] = C_GREEN
                    green_idx = sum(len(p) for p in pts)
            pts.append(local @ sh["R"].T @ Rg.T)
            cols.append(col)
            rads.append(rad)
        pts, rads = np.concatenate(pts), np.concatenate(rads)
        cols = [c for group in cols for c in group]

        # 3) Project, depth-sort (painter's algorithm: far first), draw.
        xy, s, valid = self._project(pts, centre, focal)
        for i in np.argsort(-pts[:, 2]):
            if not valid[i]:
                continue
            x, y = int(xy[i, 0]), int(xy[i, 1])
            r = min(MAX_R, max(1, int(round(rads[i] * s[i] * spread))))   # particles scale
            cv2.circle(frame, (x, y), r + 1, C_OUTLINE, -1, cv2.LINE_AA)   # outline
            cv2.circle(frame, (x, y), r, cols[i], -1, cv2.LINE_AA)
            if i == green_idx:
                cv2.circle(frame, (x, y), r + 5, C_OUTLINE, 3, cv2.LINE_AA)
                cv2.circle(frame, (x, y), r + 5, C_GREEN, 1, cv2.LINE_AA)
                self.green_xy, self.green_r = (x, y), r

# ---- UI helpers -----------------------------------------------------------
def put_label(frame, text, pos, scale=0.55):
    """Plain black text, no halo."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 1, cv2.LINE_AA)


def back_rect(w, h):
    """BACK button rectangle (x0, y0, x1, y1), bottom-right corner."""
    return (w - 96, h - 52, w - 12, h - 12)

def mode_rect(w, h):
    return(w - 130, 30, w - 12, 62)

def side_rect(w, slot):
    """Top-right button to show SPIN/ORBIT/ZOOM"""
    y = 30 + 38 * slot
    return(w - 130, y, w - 12, y + 32)

def draw_toggle(frame, rect, text):
    """Toggle button"""
    x0, y0, x1, y1 = rect
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), 1)
    put_label(frame, text, (x0 + 8, y0 + 21), 0.5)
     
def draw_mode_button(frame, rect, manual):
    x0, y0, x1, y1 = rect
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), 1)
    put_label(frame, "SPIN: " + ("MANUAL" if manual else "AUTO"), (x0 + 8, y0 + 21), 0.5)

def draw_rotate_marker(frame, pt):
    c = (int(pt[0])), int(pt[1])
    cv2.circle(frame, c, 16, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.circle(frame, c, 16, (255, 0, 200), 2, cv2.LINE_AA)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        a, b = (c[0] + dx * 19, c[1] + dy * 19), (c[0] + dx * 32, c[1] + dy * 32)
        cv2.arrowedLine(frame, a, b, (20, 20, 20), 4, cv2.LINE_AA, tipLength = 0.5)
        cv2.arrowedLine(frame, a, b, (255, 0, 200), 2, cv2.LINE_AA, tipLength = 0.5)
        
def draw_back_button(frame, rect):
    x0, y0, x1, y1 = rect
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), 1)
    put_label(frame, "BACK", (x0 + 18, y0 + 26))

def rect_around(c, half):
    """Square hit-zone for the electron of half its width"""
    return(int(c[0] - half), int(c[1] - half), int(c[0] + half), int(c[1] + half))

def park_pos(w, h):
    """Holding the detached electron"""
    return(w // 2, 112)
def draw_parked_electron(frame, pos):
    c = (int(pos[0]), int(pos[1]))
    cv2.circle(frame, c, 10, C_OUTLINE, -1, cv2.LINE_AA)
    cv2.circle(frame, c, 8, C_GREEN, -1, cv2.LINE_AA)
    put_label(frame, "e- (hold to return)", (c[0] + 16, c[1] + 5), 0.45)

def nav_rects(w, h):
    cy = h // 2
    return(8, cy - 18, 66, cy + 18), (w - 66, cy - 18, w - 8, cy + 18)

def draw_nav_button(frame, rect, label):
    x0, y0, x1, y1 = rect
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), 1)
    put_label(frame, label, (x0 + 7, y0 + 24), 0.5)

def draw_legend(frame, h):
    """Colour key, bottom-left."""
    items = [("proton", C_PROTON), ("neutron", C_NEUTRON), ("electron", C_ELECTRON), ("outer e-  (hold to remove)", C_GREEN)]
    for i, (label, col) in enumerate(items):
        y = h - 68 + i * 16
        cv2.circle(frame, (18, y), 6, C_OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(frame, (18, y), 5, col, -1, cv2.LINE_AA)
        put_label(frame, label, (30, y + 5), 0.45)

if __name__ == "__main__":
    # Camera-free preview: python bohr.py <Z>
    import time
    z = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    model, t0 = BohrModel(z), time.time()
    while True:
        canvas = np.full((480, 640, 3), 235, np.uint8)   # light background
        model.render(canvas, (320, 240), time.time() - t0)
        put_label(canvas, model.info, (10, 24))
        draw_legend(canvas, 480)
        cv2.imshow("bohr", canvas)
        if cv2.waitKey(16) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
