import numpy as np
import cv2

from ring_detector import detect_rings


def main() -> None:
    img = np.full((600, 900), 40, dtype=np.uint8)
    # Use ellipses instead of rectangles — more representative of real PCB rings.
    shapes = [
        (100, 100, 60, 15),
        (260, 105, 55, 14),
        (420, 100, 65, 16),
        (580, 95, 50, 13),
        (740, 90, 45, 12),
    ]
    for x, y, w, h in shapes:
        cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, 220, thickness=3)

    output = detect_rings(img, source_dpi=1200, threshold_width_px=2)
    dpis = [round(item.effective_dpi, 1) for item in output.results]

    print(f"baseline_minor_px={output.baseline_minor_px}")
    print(f"results={len(output.results)}")
    print(f"components={len(output.components)}")
    print(f"below_threshold={sum(item.is_below_threshold for item in output.results)}")
    print(f"dpis={dpis}")
    print(f"annotated_shape={output.annotated_bgr.shape}")

    assert len(output.results) >= 1
    assert len(output.components) >= 1
    assert output.annotated_bgr.shape[:2] == img.shape[:2]
    print("PASS")


if __name__ == "__main__":
    main()
