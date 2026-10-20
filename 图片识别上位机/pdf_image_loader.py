from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
from PIL import Image


def load_image_file(path: str) -> np.ndarray:
    image = Image.open(path)
    return np.array(image.convert("RGB"))


def load_pdf_page(path: str, page_index: int, render_dpi: float) -> np.ndarray:
    pdf = fitz.open(path)
    if pdf.page_count == 0:
        raise ValueError("PDF has no pages")
    page_index = max(0, min(page_index, pdf.page_count - 1))
    page = pdf.load_page(page_index)
    matrix = fitz.Matrix(render_dpi / 72.0, render_dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    pdf.close()
    return np.array(image)


def is_pdf(path: str) -> bool:
    return Path(path).suffix.lower() == ".pdf"
