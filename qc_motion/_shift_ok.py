# -*- coding: utf-8 -*-
"""Check that the scene shifted by about the commanded millimetres.

Exit 0 when the measured move is at least half the command. Exit 2 when the
frame barely moved (stall). X+ shifts the scene up, Y+ shifts it left.
"""
import sys

import cv2
import numpy as np

PX_PER_MM = 44.2
SCALE = 4


def load(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"unreadable {path}")
    small = cv2.resize(image, (image.shape[1] // SCALE, image.shape[0] // SCALE), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def main():
    before_path, after_path = sys.argv[1], sys.argv[2]
    x0, y0, x1, y1 = (int(v) for v in sys.argv[3:7])
    command_x = x1 - x0
    command_y = y1 - y0
    command = (command_x ** 2 + command_y ** 2) ** 0.5
    before = load(before_path)
    after = load(after_path)
    small_per_mm = PX_PER_MM / SCALE
    # Where a feature in `before` should land in `after`.
    expect_row = -command_x * small_per_mm
    expect_col = -command_y * small_per_mm
    side = 140
    center_row = before.shape[0] // 2 - int(round(expect_row * 0.45))
    center_col = before.shape[1] // 2 - int(round(expect_col * 0.45))
    r0 = min(max(center_row - side // 2, 0), before.shape[0] - side)
    c0 = min(max(center_col - side // 2, 0), before.shape[1] - side)
    template = before[r0:r0 + side, c0:c0 + side]
    result = cv2.matchTemplate(after, template, cv2.TM_CCOEFF_NORMED)

    def window_peak(row_shift, col_shift, radius_mm):
        radius = max(2, int(round(radius_mm * small_per_mm)))
        row = int(round(r0 + row_shift))
        col = int(round(c0 + col_shift))
        r_lo = max(row - radius, 0)
        r_hi = min(row + radius + 1, result.shape[0])
        c_lo = max(col - radius, 0)
        c_hi = min(col + radius + 1, result.shape[1])
        patch = result[r_lo:r_hi, c_lo:c_hi]
        if patch.size == 0:
            return -1.0, 0.0, 0.0
        _, local, _, where = cv2.minMaxLoc(patch)
        return float(local), (r_lo + where[1] - r0) / small_per_mm, (c_lo + where[0] - c0) / small_per_mm

    expect_score, row_mm, col_mm = window_peak(expect_row, expect_col, 8)
    zero_score, _, _ = window_peak(0, 0, 4)
    moved_x = -row_mm
    moved_y = -col_mm
    print(
        f"shift cmd {command_x:+d},{command_y:+d} mm  measured x {moved_x:.1f} y {moved_y:.1f}"
        f"  at-target {expect_score:.3f}  at-zero {zero_score:.3f}",
        flush=True,
    )
    # A repeating chamber can score well 40 mm away even when the camera did not move.
    # The real displacement has to beat the unshifted match.
    # 0.70 is above the false peak from a repeated chamber when the camera did not move
    # (that peak was 0.64) and below a real 40 mm move (0.88).
    if expect_score < 0.70 or expect_score < zero_score + 0.15:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
