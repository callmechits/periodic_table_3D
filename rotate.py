"""Module 7: manual rotation by dragging two fingers (index + middle together)."""
import numpy as np
from bohr import PITCH

# MediaPipe landmark ids.
WRIST, MIDDLE_MCP = 0, 9
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12


class RotateControl:
    def __init__(self, sens=0.012, join=0.32, release=0.45, pitch_lim=1.4, alpha=0.6):
        self.sens = sens            # rad per px of finger travel
        self.join = join            # tip gap / hand size below which the gesture engages
        self.release = release      # looser threshold to disengage (hysteresis)
        self.pitch_lim = pitch_lim  # pitch clamp (rad) so the model can't flip over
        self.alpha = alpha          # EMA weight on the newest fingertip midpoint
        self.reset()

    def reset(self):
        """New model view: default orientation, nothing tracked."""
        self.yaw, self.pitch = 0.0, PITCH
        self.release_hand()

    def release_hand(self):
        """Forget the tracked hand. Returns None so callers can use it as a no-op value."""
        self.engaged, self.hand, self.prev, self.mid = False, None, None, None
        return None

    def spin(self, dt, speed):
        """Auto-spin: advance yaw by speed (rad/s) over dt seconds."""
        self.yaw += speed * dt

    def _midpoint(self, lm, w, h, engaged):
        """Midpoint of index/middle tips if the gesture holds on this hand, else None."""
        P = lambda i: np.array([lm[i].x * w, lm[i].y * h])
        size = np.linalg.norm(P(WRIST) - P(MIDDLE_MCP))      # hand-size normaliser
        if size < 1:
            return None
        # Both fingers extended: tip farther from the wrist than its PIP joint (excludes a fist).
        extended = (np.linalg.norm(P(INDEX_TIP) - P(WRIST)) > np.linalg.norm(P(INDEX_PIP) - P(WRIST))
                    and np.linalg.norm(P(MIDDLE_TIP) - P(WRIST)) > np.linalg.norm(P(MIDDLE_PIP) - P(WRIST)))
        gap = np.linalg.norm(P(INDEX_TIP) - P(MIDDLE_TIP)) / size
        if extended and gap < (self.release if engaged else self.join):
            return (P(INDEX_TIP) + P(MIDDLE_TIP)) / 2
        return None

    def update(self, hands_lm, w, h):
        """hands_lm: list of per-hand landmark lists. Returns the (x, y) marker if engaged, else None."""
        order = list(range(len(hands_lm)))
        if self.hand in order:                                # prefer the hand already tracked
            order.remove(self.hand)
            order.insert(0, self.hand)
        found = None
        for i in order:
            m = self._midpoint(hands_lm[i], w, h, self.engaged and i == self.hand)
            if m is not None:
                found = (i, m)
                break
        if found is None:
            return self.release_hand()

        i, m = found
        if i != self.hand:                                    # different hand: avoid a jump
            self.prev = self.mid = None
        self.hand, self.engaged = i, True
        self.mid = m if self.mid is None else self.alpha * m + (1 - self.alpha) * self.mid
        if self.prev is not None:
            dx, dy = self.mid - self.prev
            self.yaw -= dx * self.sens                        # surface follows the fingers
            self.pitch = float(np.clip(self.pitch + dy * self.sens,
                                       -self.pitch_lim, self.pitch_lim))
        self.prev = self.mid.copy()
        return tuple(self.mid)
