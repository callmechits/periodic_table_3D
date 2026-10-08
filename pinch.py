"""Module 6: pinch-to-zoom from thumb-index distance (scale-invariant, bounded)."""
import numpy as np

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9


class PinchZoom:
    def __init__(self, zmin=0.5, zmax=2.5, gain=1.5, deadband=0.012, alpha=0.5):
        self.zmin, self.zmax = zmin, zmax    # zoom bounds
        self.gain = gain                     # >1 amplifies: more zoom per pinch
        self.deadband = deadband             # min relative change that counts
        self.alpha = alpha                   # EMA weight on the newest distance
        self.reset()

    def reset(self):
        """Call when a view opens or the hand is lost."""
        self.zoom, self.smooth, self.ref = 1.0, None, None

    def update(self, lm, w, h):
        """lm: one hand's landmark list, or None. Returns the current zoom."""
        if lm is None:                       # hand lost: keep zoom, drop references
            self.smooth = self.ref = None
            return self.zoom
        P = lambda i: np.array([lm[i].x * w, lm[i].y * h])
        size = np.linalg.norm(P(WRIST) - P(MIDDLE_MCP))    # hand-size normaliser
        if size < 1:
            return self.zoom
        d = np.linalg.norm(P(THUMB_TIP) - P(INDEX_TIP)) / size

        if self.smooth is None:              # first frame of a new tracking run
            self.smooth = self.ref = d
            return self.zoom
        self.smooth = self.alpha * d + (1 - self.alpha) * self.smooth

        ratio = self.smooth / max(self.ref, 1e-6)   # >1 when the fingers open
        if abs(ratio - 1) > self.deadband:
            self.zoom = float(np.clip(self.zoom * ratio ** self.gain, self.zmin, self.zmax))
            self.ref = self.smooth           # advance the reference only on change
        return self.zoom
