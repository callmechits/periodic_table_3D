"""Module 1: detect the 'draw a rectangle with both index fingers' open gesture."""
import time
import cv2
import numpy as np
import mediapipe as mp

# ---- Tunable parameters -------------------------------------------------
TOGETHER = 0.06      # tip separation (fraction of frame width) counted as "together"
APART = 0.15         # separation required at some point during the drawing
MIN_T, MAX_T = 1.0, 5.0   # allowed gesture duration in seconds
RECT_FILL = 0.80     # min (hull area / min-area-rect area): rectangularity
BORDER_TOL = 0.08    # max mean distance to rect border (fraction of diagonal)
INDEX_TIP = 8        # MediaPipe landmark id of the index fingertip


class RectGesture:
    """State machine: IDLE -> DRAWING -> (success | reset)."""

    def __init__(self):
        self.drawing = False
        self.pts = []            # trail of both tips, pixel coords
        self.t0 = 0.0
        self.was_apart = False

    def _reset(self):
        self.drawing, self.pts, self.was_apart = False, [], False

    def _is_rectangle(self):
        """True if the trail is close to the border of a rectangle."""
        pts = np.array(self.pts, dtype=np.float32)
        rect = cv2.minAreaRect(pts)               # ((cx, cy), (w, h), angle)
        (w, h) = rect[1]
        if w < 40 or h < 40:                      # too small to be intentional
            return False, None
        hull = cv2.convexHull(pts)
        if cv2.contourArea(hull) / (w * h) < RECT_FILL:
            return False, None                    # trail is not rectangular
        box = cv2.boxPoints(rect).astype(np.float32)   # 4 corners
        # Distance of every trail point to the nearest rectangle edge.
        dists = np.empty((len(pts), 4))
        for i in range(4):
            a, b = box[i], box[(i + 1) % 4]
            ab = b - a
            t = np.clip(((pts - a) @ ab) / (ab @ ab), 0, 1)
            proj = a + t[:, None] * ab
            dists[:, i] = np.linalg.norm(pts - proj, axis=1)
        diag = np.hypot(w, h)
        if dists.min(axis=1).mean() / diag > BORDER_TOL:
            return False, None                    # points wander inside the shape
        # Every side must be covered by some trail points (rules out a line).
        side_hits = (dists.argmin(axis=1)[:, None] == np.arange(4)).sum(axis=0)
        if (side_hits < 3).any():
            return False, None
        return True, box

    def update(self, tips, frame_w):
        """tips: two (x, y) pixel points or None. Returns box corners on success."""
        if tips is None:                          # a hand was lost: abort
            self._reset()
            return None
        sep = np.hypot(*(np.array(tips[0]) - np.array(tips[1]))) / frame_w
        now = time.time()

        if not self.drawing:
            if sep < TOGETHER:                    # fingers touching: start
                self.drawing, self.t0 = True, now
                self.pts, self.was_apart = [], False
            return None

        self.pts += [tips[0], tips[1]]
        if now - self.t0 > MAX_T:                 # too slow
            self._reset()
            return None
        if sep > APART:
            self.was_apart = True
        # Fingers reunited after being apart: evaluate the shape.
        if self.was_apart and sep < TOGETHER and now - self.t0 > MIN_T:
            ok, box = self._is_rectangle()
            self._reset()
            return box if ok else None
        return None


def main():
    hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)      # DSHOW: faster startup on Windows
    gesture, shown_until, box = RectGesture(), 0.0, None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                # mirror for natural interaction
        h, w = frame.shape[:2]
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        tips = None
        if res.multi_hand_landmarks and len(res.multi_hand_landmarks) == 2:
            tips = [(lm.landmark[INDEX_TIP].x * w, lm.landmark[INDEX_TIP].y * h) for lm in res.multi_hand_landmarks]
            for p in tips:
                cv2.circle(frame, (int(p[0]), int(p[1])), 6, (0, 255, 255), -1)

        result = gesture.update(tips, w)
        if result is not None:                    # success: show it for 2 s
            box, shown_until = result.astype(int), time.time() + 2
        if time.time() < shown_until:
            cv2.polylines(frame, [box], True, (0, 255, 0), 3)
            cv2.putText(frame, "OPEN", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        # Draw the live trail so you can see what is being recorded.
        for p in gesture.pts[-120:]:
            cv2.circle(frame, (int(p[0]), int(p[1])), 2, (255, 0, 255), -1)

        cv2.imshow("gesture", frame)
        if cv2.waitKey(1) & 0xFF == 27:           # Esc quits
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
