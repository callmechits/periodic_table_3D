"""Module 2: periodic-table overlay, fixed at screen centre.
Test: python table_overlay.py  (open gesture or press 't' toggles the table)."""
import time
import cv2
import numpy as np
import mediapipe as mp
from elements import ELEMENTS, grid_pos, category
from gesture import RectGesture, INDEX_TIP
from selection import Selector, THUMB_TIP, draw_progress
from bohr import (BohrModel, put_label, back_rect, draw_back_button, draw_legend, mode_rect, draw_mode_button, draw_rotate_marker, YAW_SPEED, nav_rects, draw_nav_button, rect_around, park_pos, draw_parked_electron, side_rect, draw_toggle, level_rects, draw_level_buttons, draw_spectrum, draw_orbital_legend)
from orbitals import config_lines
from camera import CameraThread
from selection import Selector, THUMB_TIP, draw_progress, PointSmoother
from pinch import PinchZoom
from rotate import RotateControl

CELL = 34      # cell side in pixels (18 * 34 = 612 fits a 640 px frame)
GAP = 10       # vertical gap between the main table and the f-block rows
PAD = 6        # padding inside the backing panel

# BGR colours per category: bright, saturated, distinguishable on a dark panel.
COLORS = {
    "alkali": (80, 80, 255),     "alkaline": (60, 160, 255),
    "transition": (60, 220, 255), "post": (150, 230, 120),
    "metalloid": (200, 220, 60), "nonmetal": (255, 200, 80),
    "halogen": (255, 120, 180),  "noble": (255, 90, 220),
    "lanthanide": (220, 200, 255), "actinide": (180, 160, 255),
}

def build_layout(frame_w, frame_h):
    """Return (cells, panel). cells: {Z: (x0, y0, x1, y1)} in frame pixels;
    panel: backing rectangle. Computed once; also used later for hit-testing."""
    total_w = 18 * CELL
    total_h = 9 * CELL + GAP                      # 7 rows + gap + 2 f-block rows
    left = (frame_w - total_w) // 2               # centre horizontally
    top = (frame_h - total_h) // 2                # centre vertically

    def row_y(r):                                 # row 7 is the gap; f-block follows
        return top + (r * CELL if r < 7 else (r - 1) * CELL + GAP)

    cells = {}
    for z in range(1, 119):
        r, c = grid_pos(z)
        x0, y0 = left + c * CELL, row_y(r)
        cells[z] = (x0, y0, x0 + CELL - 2, y0 + CELL - 2)   # 2 px spacing
    panel = (left - PAD, top - PAD, left + total_w + PAD, top + total_h + PAD)
    return cells, panel


CELL_ALPHA = 0.35   # fill opacity: lower = more translucent (0 = invisible, 1 = opaque)
BORDER_PX = 1       # opaque outline thickness that keeps cells readable


def draw_table(frame, cells, panel, highlight=None):
    """Draw the table onto frame in place. highlight: Z to outline, or None."""
    x0, y0, x1, y1 = panel
    # Work on the panel region only: cheaper than copying the full frame.
    roi = frame[y0:y1, x0:x1]
    fill = roi.copy()

    # 1) Draw filled cells on the copy (coordinates shifted into ROI space).
    for z, (cx0, cy0, cx1, cy1) in cells.items():
        cv2.rectangle(fill, (cx0 - x0, cy0 - y0), (cx1 - x0, cy1 - y0), COLORS[category(z)], -1)

    # 2) Single blend: translucent fills over the live camera image.
    frame[y0:y1, x0:x1] = cv2.addWeighted(fill, CELL_ALPHA, roi, 1 - CELL_ALPHA, 0)

    # 3) Opaque outlines and text on top, so they stay sharp.
    font = cv2.FONT_HERSHEY_SIMPLEX
    for z, (cx0, cy0, cx1, cy1) in cells.items():
        cv2.rectangle(frame, (cx0, cy0), (cx1, cy1), COLORS[category(z)], BORDER_PX)
        sym = ELEMENTS[z][0]
        (tw, th), _ = cv2.getTextSize(sym, font, 0.5, 1)
        pos = (cx0 + (CELL - 2 - tw) // 2, cy0 + (CELL + th) // 2 + 3)
        #cv2.putText(frame, sym, pos, font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  #Uncomment for white halo on element name
        cv2.putText(frame, sym, pos, font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)        # text
        num_pos = (cx0 + 2, cy0 + 9)
        #cv2.putText(frame, str(z), num_pos, font, 0.28, (255, 255, 255), 2, cv2.LINE_AA) #Uncomment for white halo on atomic no.
        cv2.putText(frame, str(z), num_pos, font, 0.28, (0, 0, 0), 1, cv2.LINE_AA)

    if highlight in cells:
        a, b, c, d = cells[highlight]
        cv2.rectangle(frame, (a - 1, b - 1), (c + 1, d + 1), (255, 255, 255), 2)

def draw_cursor(frame, pts):
    """Thumb pointer yay"""
    for x, y in pts:
        c = (int(x), int(y))
        #cv2.circle(frame, c, 9 , (20, 20, 20), 2, cv2.LINE_AA)    #Black outer pointer
        cv2.circle(frame, c, 5, (0, 230, 255), -1, cv2.LINE_AA)    #Yellow dot + 4 lines
        #cv2.circle(frame, c, 5, (20, 20, 20), 1, cv2.LINE_AA)     #Outer black circle + 4 lines
        for dx, dy in ((14, 0), (-14, 0), (0, 14), (0, -14)):
            cv2.line(frame, (c[0] + dx // 2, c[1] + dy // 2), (c[0] + dx, c[1] + dy), (20, 20, 20), 1, cv2.LINE_AA)

def main():
    pinch = PinchZoom(zmin = 0.3, zmax = 10.0)
    rot, manual, t_last = RotateControl(), False, 0.0
    paused, zoom_locked = False, False
    view_orbital, spec_on = False, False

    hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity = 0, 
                                     min_detection_confidence=0.6,
                                     min_tracking_confidence=0.6)
    cam = CameraThread()
    smoother, last_idx, fps, t_prev = PointSmoother(), 0, 0.0, time.time()
    gesture, selector = RectGesture(), Selector()
    visible, layout = False, None
    model, t0 = None, 0.0          # model is not None  <=>  an element view is open

    while True:
        frame, idx = cam.read()
        if frame is None or idx == last_idx:
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        last_idx = idx
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if layout is None:
            layout = build_layout(w, h)
        cells, panel = layout
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        hands_lm = res.multi_hand_landmarks or []
        tips = None
        if len(hands_lm) == 2:                       # index tips for the open gesture
            tips = [(l.landmark[INDEX_TIP].x * w, l.landmark[INDEX_TIP].y * h) for l in hands_lm]
        thumbs = smoother.update([(l.landmark[THUMB_TIP].x * w, l.landmark[THUMB_TIP].y * h) for l in hands_lm])

        if gesture.update(tips, w) is not None:      # open gesture toggles everything
            visible = not visible
            model, selector = None, Selector()       # always reopen on the table

        if visible:
            now = time.time()
            mid = None                               # rotation marker position, if engaged
            if model is None:
                # ---- Table mode: thumb dwell picks an element ----
                hover, progress, chosen = selector.update(thumbs, cells, now)
                draw_table(frame, cells, panel, highlight=hover)
                if hover is not None:
                    draw_progress(frame, cells[hover], progress)
                if chosen is not None:
                    model, t0, selector = BohrModel(chosen), now, Selector()
                    pinch.reset()
                    rot.reset()
                    manual, paused, zoom_locked, t_last = False, False, False, now    # Fresh view state now
            else:
                # ---- View mode: model + BACK + SPIN toggle ----
                nav_prev, nav_next = nav_rects(w, h)
                prev_z = (model.z - 2) % 118 + 1
                next_z = model.z % 118 + 1
                zones = {"back": back_rect(w, h), "mode": mode_rect(w, h), "orbit": side_rect(w, 1), "zoom": side_rect(w, 2), "view": side_rect(w, 3), "prev": nav_prev, "next": nav_next}
                if not view_orbital:
                    zones["spec"] = side_rect(w, 4)
                lv_rects = []
                if spec_on and not view_orbital:
                    lv_rects = level_rects(w, h, len(model.levels))
                    for i, rc in enumerate(lv_rects):
                        zones[f"lv{i}"] = rc
                if not (spec_on or view_orbital):
                    if model.detached:
                        zones["electron"] = rect_around(park_pos(w, h), 26)
                    elif model.green_xy is not None:
                        zones["electron"] = rect_around(model.green_xy, max(model.green_r + 14, 24))
                
                dt, t_last = now - t_last, now
                
                mid = rot.update([l.landmark for l in hands_lm], w, h) if manual else rot.release_hand()
                if not manual:
                    rot.spin(dt, YAW_SPEED)          # auto-spin advances the shared yaw
                # While rotating, pause the thumb dwell and the pinch zoom.
                hover, progress, chosen = selector.update([] if mid else thumbs, zones, now)
                model.hold = (hover == "electron")
                zoom = pinch.update(None if (mid or zoom_locked) else (hands_lm[0].landmark if hands_lm else None), w, h)
                model.paused = paused
                model.view = "orbitals" if view_orbital else "bohr"
                model.set_spectrum(spec_on and not view_orbital)
                model.paused = paused
                model.render(frame, (w // 2, h // 2), now - t0, zoom, rot.yaw, rot.pitch)

                if not view_orbital:
                    if model.detached:
                        draw_parked_electron(frame, park_pos(w, h))
                    ion = model.ion_text()
                    draw_legend(frame, h)
                    if spec_on:
                        draw_level_buttons(frame, lv_rects, model.levels, model.level)
                        draw_spectrum(frame, w, h, model)

                else:
                    for i, line in enumerate(config_lines(model.z)):
                        put_label(frame, line, (10, 84 + 16 * i), 0.42)
                    draw_orbital_legend(frame, h)

                
                put_label(frame, model.info, (10, 24))
                put_label(frame, f"zoom {zoom:.2f}x", (10, 44))

                if spec_on and model.event:
                    put_label(frame, model.event, (10, 64), 0.38)
                if manual:
                    put_label(frame, "index + middle together: drag to rotate", (10, 84), 0.45)
                draw_back_button(frame, zones["back"])
                draw_mode_button(frame, zones["mode"], manual)
                draw_toggle(frame, zones["orbit"], "ORBIT: " + ("HOLD" if paused else "RUN"))
                draw_toggle(frame, zones["zoom"], "ZOOM: " + ("LOCK" if zoom_locked else "FREE"))
                draw_nav_button(frame, nav_prev, "< " + ELEMENTS[prev_z][0])
                draw_nav_button(frame, nav_next, ELEMENTS[next_z][0] + " >")
                draw_toggle(frame, zones["view"], "VIEW: " + ("ORBITAL" if view_orbital else "BOHR"), 0.42)
                if not view_orbital:
                    draw_toggle(frame, zones["spec"], "SPEC: " + ("ON" if spec_on else "OFF"))
                    if model.detached:
                        draw_parked_electron(frame, park_pos(w, h))
                        ion = model.ion_text()
                        if ion:
                            put_label(frame, ion, (10, 104), 0.38)
                if hover in zones:
                    draw_progress(frame, zones[hover], progress)
                if mid:
                    draw_rotate_marker(frame, mid)
                if chosen == "back":                 # back to the table
                    model, selector = None, Selector()
                elif chosen == "mode":               # toggle auto / manual spin
                    manual = not manual
                elif chosen in("prev", "next"):
                    model = BohrModel(prev_z if chosen == "prev" else next_z)
                elif chosen == "electron":
                    model.detached = not model.detached
                elif chosen == "orbit":
                    paused = not paused
                elif chosen == "zoom":
                    zoom_locked = not zoom_locked
                elif chosen == "view":
                    view_orbital = not view_orbital
                elif chosen == "spec":
                    spec_on = not spec_on
                elif chosen and chosen.startswith("lv"):
                    model.set_level(model.levels[int(chosen[2:])], now - t0)
                
            draw_cursor(frame, [] if mid else thumbs)   # thumb pointer, hidden while rotates


        now_t = time.time()
        fps = 0.9 * fps + 0.1 / max(now_t - t_prev, 1e-6)
        t_prev = now_t
        cv2.putText(frame, f"{fps:.0f} FPS", (w - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("table", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:                                # Esc quits
            break
        if key == ord("t"):                          # manual toggle for testing
            visible = not visible
            model, selector = None, Selector()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
