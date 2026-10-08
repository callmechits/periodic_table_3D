"""Module 3: select an element by holding a thumb tip over its cell."""
import cv2

THUMB_TIP = 4      # MediaPipe landmark id of the thumb tip
DWELL = 0.8        # seconds the thumb must stay on a cell to select it
GRACE = 0.15       # seconds of tracking loss tolerated before the timer resets
MARGIN = 5         # px: hysteresis; thumb may drift this far outside the current cell


def _inside(pt, rect, margin=0):
    """True if pt lies in rect (x0, y0, x1, y1) inflated by margin pixels."""
    x, y = pt
    x0, y0, x1, y1 = rect
    return x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin

class PointSmoother:
    """Exponential moving average for a small set of points.
    Points are ordered by x so slot i stays attached to the same hand; the
    state resets whenever the number of points changes."""

    def __init__(self, alpha=0.6):               # higher alpha = less lag, more jitter
        self.alpha, self.state = alpha, None

    def update(self, pts):
        pts = sorted(pts)                        # sort by x
        if self.state is None or len(self.state) != len(pts):
            self.state = [tuple(p) for p in pts]
        else:
            a = self.alpha
            self.state = [(a * p[0] + (1 - a) * s[0], a * p[1] + (1 - a) * s[1]) for p, s in zip(pts, self.state)]
        return self.state
    
class Selector:
    """Dwell-based selection. update() returns (hover_z, progress, selected_z)."""

    def __init__(self):
        self.hover = None      # element currently under the thumb
        self.t_start = 0.0     # when the current dwell began
        self.t_lost = None     # when the thumb last left every cell
        self.locked = False    # True after firing; must leave the cell to re-fire
        self.progress = 0.0

    def _hit(self, pt, cells):
        """Return Z of the cell containing pt, else None."""
        for z, rect in cells.items():
            if _inside(pt, rect):
                return z
        return None

    def update(self, thumbs, cells, now):
        """thumbs: list of (x, y) thumb tips (one per detected hand)."""
        # Pick the first thumb that lies on a cell.
        z = None
        for p in thumbs:
            if self._hit(p, cells) is not None:
                # Hysteresis: stay on the current cell while within MARGIN of it.
                if self.hover is not None and _inside(p, cells[self.hover], MARGIN):
                    z = self.hover
                else:
                    z = self._hit(p, cells)
                break

        if z is None:
            # No thumb on the table: tolerate brief loss, then reset everything.
            if self.t_lost is None:
                self.t_lost = now
            if now - self.t_lost > GRACE:
                self.hover, self.locked, self.progress = None, False, 0.0
            return self.hover, self.progress, None   # progress frozen during grace

        self.t_lost = None
        if z != self.hover:                          # moved to a different cell
            self.hover, self.t_start, self.locked = z, now, False
        self.progress = min(1.0, (now - self.t_start) / DWELL)

        if self.progress >= 1.0 and not self.locked:
            self.locked = True                       # fire exactly once per dwell
            return self.hover, self.progress, self.hover
        return self.hover, self.progress, None


def draw_progress(frame, rect, progress):
    """Circular progress ring centred on the hovered cell."""
    x0, y0, x1, y1 = rect
    centre = ((x0 + x1) // 2, (y0 + y1) // 2)
    # Arc starts at 12 o'clock (angle = -90) and sweeps clockwise.
    cv2.ellipse(frame, centre, (16, 16), -90, 0, int(360 * progress), (0, 0, 0), 4, cv2.LINE_AA)           # dark under-stroke
    cv2.ellipse(frame, centre, (16, 16), -90, 0, int(360 * progress), (255, 255, 255), 2, cv2.LINE_AA)
