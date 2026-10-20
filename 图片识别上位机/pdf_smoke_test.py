from __future__ import annotations

from pathlib import Path

from pdf_image_loader import is_pdf, load_pdf_page
from ring_detector import detect_rings


def main() -> None:
    pdf_path = Path("23121.pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing test file: {pdf_path}")

    if not is_pdf(str(pdf_path)):
        raise ValueError(f"Not a PDF file: {pdf_path}")

    image = load_pdf_page(str(pdf_path), page_index=0, render_dpi=600)
    output = detect_rings(image, source_dpi=1200, threshold_width_px=4)

    print(f"pdf={pdf_path.name}")
    print(f"render_shape={image.shape}")
    print(f"baseline_minor_px={output.baseline_minor_px}")
    print(f"results={len(output.results)}")
    print(f"patterns={len(output.components)}")
    print(f"below_threshold={sum(item.is_below_threshold for item in output.results)}")
    print(f"open_rings={sum(item.is_broken for item in output.results)}")
    print(f"inner_widths={[round(item.inner_line_width_px, 1) for item in output.results[:15]]}")
    print(f"outer_widths={[round(item.outer_line_width_px, 1) for item in output.results[:15]]}")
    print(f"min_widths={[round(item.min_line_width_px, 1) for item in output.results[:15]]}")
    same = sum(1 for item in output.results if abs(item.inner_line_width_px - item.outer_line_width_px) < 1e-6)
    print(f"identical_inner_outer={same}")

    annotated_path = Path("pdf_smoke_test_annotated.png")
    import cv2

    cv2.imwrite(str(annotated_path), output.annotated_bgr)
    print(f"annotated={annotated_path.name}")
    if len(output.results) != 131:
        raise SystemExit(f"EXPECTED 131 patterns, got {len(output.results)}")
    print("PASS count=131")


if __name__ == "__main__":
    main()
