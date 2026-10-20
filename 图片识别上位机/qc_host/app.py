# -*- coding: utf-8 -*-
"""Entry: python -m qc_host"""

from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qc_host.ui.preview import PreviewApp  # noqa: E402


def main() -> None:
    root = tk.Tk()
    PreviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
