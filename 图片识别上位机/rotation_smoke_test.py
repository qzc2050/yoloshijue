# -*- coding: utf-8 -*-
"""Rotation + whole-board SR compensation smoke tests."""

from __future__ import annotations

import cv2
import numpy as np

from ring_detector import detect_rings


def _draw_dual_ring(
    image: np.ndarray,
    center: tuple[int, int],
    outer: tuple[int, int],
    inner: tuple[int, int],
    angle: float,
    thickness: int = 6,
    color: int = 220,
) -> None:
    cv2.ellipse(image, center, outer, angle, 0, 360, color, thickness)
    cv2.ellipse(image, center, inner, angle, 0, 360, color, thickness)


def test_each_angle_detected() -> None:
    counts = []
    for angle in (0, 30, 60, 90, 135):
        image = np.full((480, 480), 40, dtype=np.uint8)
        _draw_dual_ring(image, (240, 240), (150, 58), (118, 34), angle, thickness=6)
        result = detect_rings(image, source_dpi=1200, threshold_width_px=3, sr_scale=2, allow_small=True)
        counts.append((angle, len(result.results)))
        assert len(result.results) >= 1, f"angle {angle}: expected >=1 pattern, got {len(result.results)}"
        assert result.results[0].min_line_width_px > 0
    print(f"per_angle_counts={counts}")


def test_mixed_orientations_and_narrow_report() -> None:
    image = np.full((900, 1100), 36, dtype=np.uint8)
    specs = [
        ((220, 220), 0, 6),
        ((780, 220), 45, 6),
        ((220, 680), 90, 6),
        ((780, 680), 135, 2),  # thinner copper -> below 3.5 px threshold after scale
    ]
    for center, angle, thickness in specs:
        _draw_dual_ring(image, center, (130, 50), (102, 30), angle, thickness=thickness)

    output = detect_rings(image, source_dpi=1200, threshold_width_px=3.5, sr_scale=2, allow_small=True)
    print(f"mixed_count={len(output.results)}")
    print(f"angles={[round(item.angle_deg) for item in output.results]}")
    print(f"min_widths={[round(item.min_line_width_px, 2) for item in output.results]}")
    print(f"below={[item.index for item in output.results if item.is_below_threshold]}")
    print(f"centers={[ (round(item.cx), round(item.cy)) for item in output.results ]}")

    assert len(output.results) >= 4, f"expected 4 dual-rings, got {len(output.results)}"
    narrow = [item for item in output.results if item.is_below_threshold]
    assert len(narrow) >= 1, "expected at least one below-threshold copper trace"
    # The thin 135° ring should be near (780, 680).
    assert any(abs(item.cx - 780) < 80 and abs(item.cy - 680) < 80 for item in narrow), (
        f"thin ring not reported near (780,680): {[(item.cx, item.cy) for item in narrow]}"
    )


def main() -> None:
    test_each_angle_detected()
    test_mixed_orientations_and_narrow_report()
    print("PASS")


if __name__ == "__main__":
    main()
