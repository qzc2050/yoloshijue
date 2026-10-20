from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from enhance import px_to_um, unsharp_gray


@dataclass
class RingResult:
    index: int
    x: int
    y: int
    w: int
    h: int
    area: int
    minor_px: float
    major_px: float
    inner_line_width_px: float
    outer_line_width_px: float
    min_line_width_px: float
    inner_effective_dpi: float
    outer_effective_dpi: float
    effective_dpi: float
    is_below_threshold: bool
    is_broken: bool  # True = black cover fully occludes a track (overflow proxy)
    component_index: int | None = None
    irregularity_min_px: float = 0.0
    irregularity_max_px: float = 0.0
    irregularity_std_px: float = 0.0
    irregularity_ratio: float = 1.0
    is_irregular_edges: bool = False
    angle_deg: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    sr_scale: int = 1
    inner_width_um: float = 0.0
    outer_width_um: float = 0.0
    min_width_um: float = 0.0


@dataclass
class ComponentResult:
    index: int
    ring_indices: Tuple[int, ...]
    x: int
    y: int
    w: int
    h: int


@dataclass
class DetectionOutput:
    annotated_bgr: np.ndarray
    results: List[RingResult]
    components: List[ComponentResult]
    baseline_minor_px: float
    sr_scale: int = 1


# Candidate: (x, y, w, h, area, fill_ratio, inner_width, outer_width, minor, aspect)
CandidateTuple = Tuple[int, int, int, int, int, float, float, float, float, float]


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("Unsupported image shape")


def _binary_copper_mask(gray: np.ndarray) -> np.ndarray:
    """Local-contrast mask. Useful on mixed boards; noisy on flat low-contrast scans."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )
    return cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )


def _global_copper_mask(gray: np.ndarray) -> np.ndarray:
    """High-contrast bright-trace mask — primary mask for stadium rings."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold = max(50, int(np.percentile(blurred, 88)))
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )


def _tophat_copper_mask(gray: np.ndarray) -> np.ndarray:
    """Emphasize thin bright tracks against a flat dark board."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )


def _primary_copper_mask(gray: np.ndarray) -> np.ndarray:
    """Prefer global/tophat union when adaptive floods the board."""
    adaptive = _binary_copper_mask(gray)
    global_mask = _global_copper_mask(gray)
    tophat = _tophat_copper_mask(gray)
    adaptive_ratio = float((adaptive > 0).mean())
    if adaptive_ratio > 0.45:
        combined = cv2.bitwise_or(global_mask, tophat)
    else:
        combined = cv2.bitwise_or(adaptive, cv2.bitwise_or(global_mask, tophat))
    return cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )


def _run_lengths(values: np.ndarray) -> List[Tuple[int, int, bool]]:
    if len(values) == 0:
        return []
    runs: List[Tuple[int, int, bool]] = []
    start = 0
    current = bool(values[0])
    for index in range(1, len(values)):
        value = bool(values[index])
        if value != current:
            runs.append((start, index, current))
            start = index
            current = value
    runs.append((start, len(values), current))
    return runs


def _measure_single_ring_width(binary: np.ndarray, contour: np.ndarray) -> Tuple[float, float, float]:
    """Measure one ring's thickness via distance transform. Returns (median, min, max)."""
    if contour is None or len(contour) < 5:
        return 0.0, 0.0, 0.0
    mask = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    # Keep only the ring band (filled outer minus hole already handled by caller if needed).
    # If contour is just the outer path of a thin ring drawn as stroke, fill may be wrong.
    # Prefer pixels of binary that lie near this contour.
    ring = cv2.bitwise_and(binary, mask)
    # Also include a thin dilation of the contour path itself.
    stroke = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(stroke, [contour], -1, 255, thickness=3)
    ring = cv2.bitwise_or(ring, cv2.bitwise_and(binary, stroke))
    if int(ring.max()) == 0:
        return 0.0, 0.0, 0.0

    dist = cv2.distanceTransform(ring, cv2.DIST_L2, 5)
    vals = dist[ring > 0]
    if vals.size < 8:
        return 0.0, 0.0, 0.0
    ridge = vals[vals >= np.percentile(vals, 60)]
    if ridge.size < 5:
        ridge = vals
    widths = ridge * 2.0
    return float(np.median(widths)), float(np.percentile(widths, 5)), float(np.percentile(widths, 95))


def _measure_trace_widths(binary: np.ndarray, contour: np.ndarray) -> Tuple[float, float]:
    """Legacy helper: sample band thickness; returns (inner, outer) when dual walls exist."""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    edges = [box[(index + 1) % 4] - box[index] for index in range(4)]
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    longest_edge = edges[int(np.argmax(lengths))]
    major_length = max(lengths)
    minor_length = min(lengths)
    if major_length < 6 or minor_length < 4:
        return 0.0, 0.0

    major_axis = longest_edge / major_length
    if abs(major_axis[0]) < 0.02 or abs(major_axis[1]) < 0.02:
        if rect[1][0] >= rect[1][1]:
            major_axis = np.array([1.0, 0.0], dtype=np.float32)
        else:
            major_axis = np.array([0.0, 1.0], dtype=np.float32)
    minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=np.float32)
    center_point = np.array(rect[0], dtype=np.float32)
    inner_samples: List[float] = []
    outer_samples: List[float] = []

    for major_offset in np.linspace(-major_length * 0.25, major_length * 0.25, num=15):
        line_points = []
        for minor_offset in np.arange(-minor_length * 0.95, minor_length * 0.95 + 1, 1.0):
            point = center_point + major_axis * major_offset + minor_axis * minor_offset
            point_x = int(round(point[0]))
            point_y = int(round(point[1]))
            if 0 <= point_x < binary.shape[1] and 0 <= point_y < binary.shape[0]:
                line_points.append(binary[point_y, point_x])
            else:
                line_points.append(0)

        line = np.asarray(line_points, dtype=np.uint8)
        runs = _run_lengths(line > 0)
        center_index = len(line) // 2
        hole_index = next(
            (
                index
                for index, (start, end, is_bright) in enumerate(runs)
                if start <= center_index < end and not is_bright
            ),
            None,
        )
        if hole_index is None:
            continue

        # Runs below hole index are one side; above are the other.
        # On each side: nearest bright = inner ring, next = outer ring.
        left_bright = [(start, end) for start, end, is_bright in runs[:hole_index] if is_bright]
        right_bright = [(start, end) for start, end, is_bright in runs[hole_index + 1 :] if is_bright]
        for side in (left_bright, right_bright):
            if not side:
                continue
            # left side: last is nearest to hole (inner); previous is outer
            # right side: first is nearest to hole (inner); next is outer
            if side is left_bright:
                inner_run = side[-1]
                outer_run = side[-2] if len(side) >= 2 else None
            else:
                inner_run = side[0]
                outer_run = side[1] if len(side) >= 2 else None
            inner_samples.append(float(inner_run[1] - inner_run[0]))
            if outer_run is not None:
                outer_samples.append(float(outer_run[1] - outer_run[0]))
            elif len(side) == 1:
                # Single wall: treat as both until dual edges resolve elsewhere.
                outer_samples.append(float(inner_run[1] - inner_run[0]))

    inner_width = float(np.median(inner_samples)) if inner_samples else 0.0
    outer_width = float(np.median(outer_samples)) if outer_samples else 0.0
    return inner_width, outer_width



def _fwhm_width(profile: np.ndarray, peak_idx: int, min_contrast: float = 3.0) -> float:
    """Half-max width around a peak on a 1-D intensity profile.

    ``min_contrast`` is peak-baseline. Keep it low so faint hairline / almost-broken
    traces still measure instead of being dropped (which would hide the true min).
    """
    if peak_idx < 0 or peak_idx >= len(profile):
        return 0.0
    peak = float(profile[peak_idx])
    baseline = float(np.percentile(profile, 25))
    if peak - baseline < min_contrast:
        return 0.0
    half = baseline + 0.5 * (peak - baseline)
    left = peak_idx
    while left > 0 and float(profile[left]) >= half:
        left -= 1
    right = peak_idx
    while right < len(profile) - 1 and float(profile[right]) >= half:
        right += 1

    def _edge(i0: int, i1: int) -> float:
        y0, y1 = float(profile[i0]), float(profile[i1])
        if abs(y1 - y0) < 1e-6:
            return float(i1)
        return float(i0) + (half - y0) / (y1 - y0)

    width = _edge(max(right - 1, 0), min(right, len(profile) - 1)) - _edge(
        left, min(left + 1, len(profile) - 1)
    )
    return float(max(0.0, width))


def _core_width(profile: np.ndarray, peak_idx: int, frac: float = 0.65, min_contrast: float = 3.0) -> float:
    """High-threshold core width — better reflects almost-broken hairlines."""
    if peak_idx < 0 or peak_idx >= len(profile):
        return 0.0
    peak = float(profile[peak_idx])
    baseline = float(np.percentile(profile, 25))
    if peak - baseline < min_contrast:
        return 0.0
    thr = baseline + frac * (peak - baseline)
    left = peak_idx
    while left > 0 and float(profile[left - 1]) >= thr:
        left -= 1
    right = peak_idx
    while right < len(profile) - 1 and float(profile[right + 1]) >= thr:
        right += 1
    return float(max(0.0, right - left + 1))


def _dual_peaks_on_half(profile: np.ndarray, toward_end: bool) -> Tuple[Optional[int], Optional[int]]:
    """Find (outer_peak, inner_peak) on one half of a cross-section.

    ``toward_end=False``: half nearer index 0 (top/left).
    ``toward_end=True``: half nearer the last index (bottom/right).
    Outer is farther from the hole; inner is nearer the hole.
    """
    if profile.size < 8:
        return None, None
    kernel = np.asarray([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= float(kernel.sum())
    smooth = np.convolve(profile.astype(np.float32), kernel, mode="same")
    baseline = float(np.percentile(smooth, 30))
    peaks = [
        index
        for index in range(2, len(smooth) - 2)
        if smooth[index] >= smooth[index - 1]
        and smooth[index] >= smooth[index + 1]
        and smooth[index] > baseline + 12
    ]
    if len(peaks) < 2:
        return None, None
    if toward_end:
        inner_idx = peaks[0]
        outer_idx = peaks[-1]
    else:
        outer_idx = peaks[0]
        inner_idx = peaks[-1]
    if abs(inner_idx - outer_idx) < 3:
        return None, None
    return int(outer_idx), int(inner_idx)


def _dual_peaks_radial(profile: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    """Find (outer, inner) peaks on a center→outward radial profile.

    Softer than straight-section detection so faint arc traces still register.
    """
    if profile.size < 10:
        return None, None
    kernel = np.asarray([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= float(kernel.sum())
    smooth = np.convolve(profile.astype(np.float32), kernel, mode="same")
    baseline = float(np.percentile(smooth, 25))
    peak_val = float(np.max(smooth))
    if peak_val - baseline < 3.0:
        return None, None
    prom = max(3.0, 0.18 * (peak_val - baseline))
    peaks = [
        index
        for index in range(2, len(smooth) - 2)
        if smooth[index] >= smooth[index - 1]
        and smooth[index] >= smooth[index + 1]
        and smooth[index] > baseline + prom
    ]
    n = len(smooth)
    lo, hi = int(n * 0.32), int(n * 0.98)
    band = [index for index in peaks if lo <= index <= hi]
    if len(band) >= 2:
        # Outer is farther from center.
        return int(band[-1]), int(band[0])
    if len(peaks) >= 2:
        return int(peaks[-1]), int(peaks[0])

    # Run-length fallback on soft threshold — still measures continuous pixels.
    thr = baseline + 0.35 * (peak_val - baseline)
    bright = smooth >= thr
    runs = _run_lengths(bright)
    bands = [(start, end) for start, end, on in runs if on and end - start >= 1]
    bands = [(start, end) for start, end in bands if end > lo and start < hi]
    if len(bands) < 2:
        return None, None
    # Nearest-to-center = inner, farthest = outer.
    inner_run = min(bands, key=lambda item: 0.5 * (item[0] + item[1]))
    outer_run = max(bands, key=lambda item: 0.5 * (item[0] + item[1]))
    if outer_run == inner_run:
        return None, None
    return int(0.5 * (outer_run[0] + outer_run[1])), int(0.5 * (inner_run[0] + inner_run[1]))


def _sample_radial_dual_widths(
    roi: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    a0: float,
    a1: float,
    n_angles: int,
    max_plausible: float,
    inner_samples: List[float],
    outer_samples: List[float],
) -> None:
    """Sample FWHM of inner/outer rings along stadium-cap radii."""
    if radius < 8 or n_angles < 3:
        return
    rh, rw = roi.shape[:2]
    r_max = min(radius * 1.05, float(min(rh, rw)))
    n_bins = max(16, int(round(r_max)))
    for angle in np.linspace(a0, a1, n_angles):
        ca, sa = float(np.cos(angle)), float(np.sin(angle))
        profile = np.zeros(n_bins, dtype=np.float32)
        valid = 0
        for index in range(n_bins):
            rad = index * (r_max / max(1, n_bins - 1))
            px = int(round(cx + rad * ca))
            py = int(round(cy + rad * sa))
            if 0 <= py < rh and 0 <= px < rw:
                profile[index] = roi[py, px]
                valid += 1
        if valid < 10:
            continue
        outer_i, inner_i = _dual_peaks_radial(profile)
        if outer_i is None or inner_i is None:
            continue
        if abs(outer_i - inner_i) < 3:
            continue
        scale = r_max / max(1, n_bins - 1)
        ow_f = _fwhm_width(profile, outer_i) * scale
        iw_f = _fwhm_width(profile, inner_i) * scale
        baseline = float(np.percentile(profile, 25))
        thr = baseline + 0.35 * (float(np.max(profile)) - baseline)

        def _run_w(center: int) -> float:
            left = center
            while left > 0 and float(profile[left]) >= thr:
                left -= 1
            right = center
            while right < len(profile) - 1 and float(profile[right]) >= thr:
                right += 1
            return max(0.0, (right - left) * scale)

        ow_r, iw_r = _run_w(outer_i), _run_w(inner_i)
        ow_cands = [value for value in (ow_f, ow_r) if 0.5 <= value <= max_plausible]
        iw_cands = [value for value in (iw_f, iw_r) if 0.5 <= value <= max_plausible]
        if iw_cands:
            inner_samples.append(float(min(iw_cands)))
        if ow_cands:
            outer_samples.append(float(min(ow_cands)))


def _measure_dual_ring_widths(
    gray: np.ndarray,
    fallback_binary: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    sr_scale: int = 1,
    already_aligned: bool = False,
) -> Tuple[float, float, float, float, float, float, float]:
    """Measure inner/outer stadium line widths for defect analysis.

    Samples both straight sides and curved caps, then reports the narrowest
    stable local FWHM on each ring. Rotated patterns are deskewed first;
    ``sr_scale`` locally upsamples the ROI so whole-board shots can still
    resolve hairline traces.

    Returns
    -------
    inner_min_px, outer_min_px, inner_med_px, outer_med_px, irr_min, irr_max, irr_ratio
    in original image pixels (SR scale already divided out).
    """
    if not already_aligned:
        aligned_g, aligned_b, _orient = _aligned_pair(gray, fallback_binary, x, y, w, h, sr_scale)
        ah, aw = aligned_g.shape[:2]
        measured = _measure_dual_ring_widths(
            aligned_g, aligned_b, 0, 0, aw, ah, sr_scale=1, already_aligned=True
        )
        scale = float(max(1, sr_scale))
        inner_min, outer_min, inner_med, outer_med, irr_min, irr_max, irr_ratio = measured
        return (
            inner_min / scale,
            outer_min / scale,
            inner_med / scale,
            outer_med / scale,
            irr_min / scale,
            irr_max / scale,
            irr_ratio,
        )

    pad = 10
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    if roi.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0

    rh, rw = roi.shape[:2]
    horizontal = w >= h
    inner_samples: List[float] = []
    outer_samples: List[float] = []
    max_plausible = max(4.0, min(w, h) * 0.35)

    def _append_pair(profile: np.ndarray, outer_i: int, inner_i: int, offset: int = 0) -> None:
        oi, ii = offset + outer_i, offset + inner_i
        ow = _fwhm_width(profile, oi)
        iw = _fwhm_width(profile, ii)
        if 0.5 <= iw <= max_plausible:
            inner_samples.append(float(iw))
        if 0.5 <= ow <= max_plausible:
            outer_samples.append(float(ow))

    if horizontal:
        cx0, cx1 = int(rw * 0.15), int(rw * 0.85)
        cy = rh // 2
        # Straight long sides.
        for col in range(cx0, cx1, 2):
            profile = roi[:, col]
            for half, toward_end, offset in (
                (profile[:cy], False, 0),
                (profile[cy:], True, cy),
            ):
                outer_i, inner_i = _dual_peaks_on_half(half, toward_end=toward_end)
                if outer_i is None or inner_i is None:
                    continue
                _append_pair(profile, outer_i, inner_i, offset)
        # Curved caps (left / right semicircles), in ROI box coordinates.
        ox, oy = float(x - x0), float(y - y0)
        radius = 0.5 * float(h)
        left_cx = ox + radius
        right_cx = ox + float(w) - radius
        cy_f = oy + radius
        n_arc = max(14, int(round(radius * 0.55)))
        _sample_radial_dual_widths(
            roi, left_cx, cy_f, radius, 0.5 * np.pi, 1.5 * np.pi, n_arc,
            max_plausible, inner_samples, outer_samples,
        )
        _sample_radial_dual_widths(
            roi, right_cx, cy_f, radius, -0.5 * np.pi, 0.5 * np.pi, n_arc,
            max_plausible, inner_samples, outer_samples,
        )
    else:
        cy0, cy1 = int(rh * 0.15), int(rh * 0.85)
        cx = rw // 2
        for row in range(cy0, cy1, 2):
            profile = roi[row, :]
            for half, toward_end, offset in (
                (profile[:cx], False, 0),
                (profile[cx:], True, cx),
            ):
                outer_i, inner_i = _dual_peaks_on_half(half, toward_end=toward_end)
                if outer_i is None or inner_i is None:
                    continue
                _append_pair(profile, outer_i, inner_i, offset)
        ox, oy = float(x - x0), float(y - y0)
        radius = 0.5 * float(w)
        cx_f = ox + radius
        top_cy = oy + radius
        bot_cy = oy + float(h) - radius
        n_arc = max(14, int(round(radius * 0.55)))
        _sample_radial_dual_widths(
            roi, cx_f, top_cy, radius, np.pi, 2.0 * np.pi, n_arc,
            max_plausible, inner_samples, outer_samples,
        )
        _sample_radial_dual_widths(
            roi, cx_f, bot_cy, radius, 0.0, np.pi, n_arc,
            max_plausible, inner_samples, outer_samples,
        )

    def _stats(samples: List[float]) -> Tuple[float, float, float]:
        if not samples:
            return 0.0, 0.0, 0.0
        arr = np.asarray(samples, dtype=np.float32)
        median = float(np.median(arr))
        # Keep real thin arcs/notches; only drop absurd sub-noise spikes.
        filtered = arr[arr >= max(0.5, median * 0.15)]
        if filtered.size < max(8, len(arr) // 5):
            filtered = arr[arr >= 0.5]
        if filtered.size == 0:
            filtered = arr
        ordered = np.sort(filtered)
        # Defect min: near-thinnest (p5), still includes almost-broken hairlines.
        narrow = float(np.percentile(ordered, 5))
        wide = float(np.percentile(filtered, 95))
        return narrow, float(np.median(filtered)), wide

    inner_min, inner_med, inner_max = _stats(inner_samples)
    outer_min, outer_med, outer_max = _stats(outer_samples)

    if inner_min < 0.8 or outer_min < 0.8:
        blurred = cv2.GaussianBlur(roi.astype(np.uint8), (3, 3), 0)
        thr = max(90, int(np.percentile(blurred, 88)))
        _, ridges = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY)
        ridges = cv2.morphologyEx(
            ridges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        )
        bin_inner: List[float] = []
        bin_outer: List[float] = []
        if horizontal:
            for col in range(int(rw * 0.15), int(rw * 0.85)):
                runs = _run_lengths(ridges[:, col] > 0)
                hole = next((i for i, (s, e, b) in enumerate(runs) if s <= rh // 2 < e and not b), None)
                if hole is None:
                    continue
                above = [(s, e) for s, e, b in runs[:hole] if b]
                below = [(s, e) for s, e, b in runs[hole + 1 :] if b]
                for side, nearest_last in ((above, True), (below, False)):
                    if len(side) < 2:
                        continue
                    if nearest_last:
                        bin_inner.append(float(side[-1][1] - side[-1][0]))
                        bin_outer.append(float(side[-2][1] - side[-2][0]))
                    else:
                        bin_inner.append(float(side[0][1] - side[0][0]))
                        bin_outer.append(float(side[1][1] - side[1][0]))
        else:
            for row in range(int(rh * 0.15), int(rh * 0.85)):
                runs = _run_lengths(ridges[row, :] > 0)
                hole = next((i for i, (s, e, b) in enumerate(runs) if s <= rw // 2 < e and not b), None)
                if hole is None:
                    continue
                left = [(s, e) for s, e, b in runs[:hole] if b]
                right = [(s, e) for s, e, b in runs[hole + 1 :] if b]
                for side, nearest_last in ((left, True), (right, False)):
                    if len(side) < 2:
                        continue
                    if nearest_last:
                        bin_inner.append(float(side[-1][1] - side[-1][0]))
                        bin_outer.append(float(side[-2][1] - side[-2][0]))
                    else:
                        bin_inner.append(float(side[0][1] - side[0][0]))
                        bin_outer.append(float(side[1][1] - side[1][0]))
        if inner_min < 0.8 and bin_inner:
            inner_min, inner_med, inner_max = _stats(bin_inner)
        if outer_min < 0.8 and bin_outer:
            outer_min, outer_med, outer_max = _stats(bin_outer)

    if inner_min < 0.8 and outer_min < 0.8:
        fb = fallback_binary[y0:y1, x0:x1]
        contours, hierarchy = cv2.findContours(fb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None and contours:
            best_idx = -1
            best_area = 0.0
            for index, contour in enumerate(contours):
                if hierarchy[0][index][3] != -1 or hierarchy[0][index][2] == -1:
                    continue
                area = float(cv2.contourArea(contour))
                if area > best_area:
                    best_area = area
                    best_idx = index
            if best_idx >= 0:
                contour = contours[best_idx] + np.array([[[x0, y0]]])
                fb_inner, fb_outer = _measure_trace_widths(fallback_binary, contour)
                if fb_inner >= 1.0:
                    inner_min = inner_med = inner_max = fb_inner
                if fb_outer >= 1.0:
                    outer_min = outer_med = outer_max = fb_outer

    all_mins = [value for value in (inner_min, outer_min) if value > 0]
    all_maxs = [value for value in (inner_max, outer_max) if value > 0]
    irr_min = float(min(all_mins)) if all_mins else 0.0
    irr_max = float(max(all_maxs)) if all_maxs else 0.0
    irr_ratio = irr_max / max(1.0, irr_min) if irr_min > 0 else 1.0
    return inner_min, outer_min, inner_med, outer_med, irr_min, irr_max, irr_ratio


def _measure_band_width_in_roi(binary: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[float, float, float, float]:
    """Compatibility wrapper returning a single band width estimate."""
    # Approximate with fallback solid-mask sampling.
    pad = 4
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(binary.shape[1], x + w + pad)
    y1 = min(binary.shape[0], y + h + pad)
    roi = binary[y0:y1, x0:x1]
    if roi.size == 0 or int(roi.max()) == 0:
        return 0.0, 0.0, 0.0, 1.0
    contours, hierarchy = cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or not contours:
        return 0.0, 0.0, 0.0, 1.0
    best_idx = -1
    best_area = 0.0
    for index, contour in enumerate(contours):
        if hierarchy[0][index][3] != -1 or hierarchy[0][index][2] == -1:
            continue
        area = float(cv2.contourArea(contour))
        if area > best_area:
            best_area = area
            best_idx = index
    if best_idx < 0:
        return 0.0, 0.0, 0.0, 1.0
    contour = contours[best_idx] + np.array([[[x0, y0]]])
    inner_w, outer_w = _measure_trace_widths(binary, contour)
    width = float(np.mean([v for v in (inner_w, outer_w) if v > 0])) if (inner_w > 0 or outer_w > 0) else 0.0
    return width, min(inner_w, outer_w) if inner_w > 0 and outer_w > 0 else width, max(inner_w, outer_w), (
        max(inner_w, outer_w) / max(1.0, min(inner_w, outer_w)) if inner_w > 0 and outer_w > 0 else 1.0
    )


def _raw_candidates(binary: np.ndarray, quick: bool = False) -> List[CandidateTuple]:
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[CandidateTuple] = []
    if hierarchy is None:
        return candidates

    for contour_index, contour in enumerate(contours):
        if hierarchy[0][contour_index][3] != -1:
            continue
        child_index = hierarchy[0][contour_index][2]
        if child_index == -1:
            continue
        area = float(cv2.contourArea(contour))
        if area < 12:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        _, (rect_width, rect_height), _ = cv2.minAreaRect(contour)
        minor = min(rect_width, rect_height)
        major = max(rect_width, rect_height)
        aspect = major / max(1.0, minor)
        fill_ratio = area / max(1.0, float(w * h))

        _, (inner_rect_width, inner_rect_height), _ = cv2.minAreaRect(contours[child_index])
        inner_minor = float(min(inner_rect_width, inner_rect_height))
        if minor < 3 or minor > 800:
            continue
        if inner_minor < 2 or inner_minor >= minor:
            continue
        if not 1.05 <= aspect <= 50.0:
            continue
        if not 0.02 <= fill_ratio <= 0.92:
            continue

        if quick:
            # Seed stage only needs geometry; skip expensive width metrology.
            inner_line_width, outer_line_width = 2.0, 2.0
        else:
            inner_line_width, outer_line_width = _measure_trace_widths(binary, contour)
            if inner_line_width < 1.0:
                continue
        candidates.append(
            (x, y, w, h, int(area), fill_ratio, inner_line_width, outer_line_width, float(minor), float(aspect))
        )

    return candidates


def _candidate_overlap(left: CandidateTuple, right: CandidateTuple) -> float:
    left_x, left_y, left_w, left_h = left[:4]
    right_x, right_y, right_w, right_h = right[:4]
    width = max(0, min(left_x + left_w, right_x + right_w) - max(left_x, right_x))
    height = max(0, min(left_y + left_h, right_y + right_h) - max(left_y, right_y))
    return width * height / max(1, min(left_w * left_h, right_w * right_h))


def _centers_close(left: CandidateTuple, right: CandidateTuple, frac: float = 0.15) -> bool:
    left_center = np.array((left[0] + left[2] / 2, left[1] + left[3] / 2))
    right_center = np.array((right[0] + right[2] / 2, right[1] + right[3] / 2))
    scale = max(np.hypot(left[2], left[3]), np.hypot(right[2], right[3]))
    return float(np.linalg.norm(left_center - right_center)) <= scale * frac


def _same_component(left: CandidateTuple, right: CandidateTuple) -> bool:
    """True when two candidates are the same pattern (including nested double-edge)."""
    if _centers_close(left, right, 0.14):
        size_ratio = min(left[4], right[4]) / max(1, max(left[4], right[4]))
        if size_ratio >= 0.35:
            return True
    return (
        _candidate_overlap(left, right) >= 0.62
        and _centers_close(left, right, 0.12)
    )


def _deduplicate_candidates(candidates: List[CandidateTuple]) -> List[CandidateTuple]:
    unique: List[CandidateTuple] = []
    for candidate in sorted(candidates, key=lambda item: item[4], reverse=True):
        if not any(_same_component(candidate, existing) for existing in unique):
            unique.append(candidate)
    return unique


def _is_stadium_shape(
    candidate: CandidateTuple,
    min_minor: float = 8.0,
    min_area: float = 200.0,
) -> bool:
    """Reject squares/vias; keep elongated hollow stadiums of consistent aspect."""
    _, _, _, _, area, fill_ratio, _, _, minor, aspect = candidate
    if aspect < 1.8 or aspect > 5.5:
        return False
    if minor < min_minor:
        return False
    if fill_ratio < 0.05 or fill_ratio > 0.85:
        return False
    if area < min_area:
        return False
    return True


def _dominant_size_keep(candidates: List[CandidateTuple]) -> List[CandidateTuple]:
    if len(candidates) < 8:
        return candidates
    minors = sorted(c[8] for c in candidates)
    best_lo, best_hi, best_count = 0.0, 0.0, 0
    for center in minors:
        lo_c = center * 0.70
        hi_c = center * 1.30
        count = sum(1 for minor in minors if lo_c <= minor <= hi_c)
        if count > best_count:
            best_count = count
            best_lo, best_hi = lo_c, hi_c
    lo = best_lo / 1.25
    hi = best_hi * 1.25
    return [c for c in candidates if lo <= c[8] <= hi]


def _contour_stadium_seeds(
    gray: np.ndarray,
    min_minor: float = 8.0,
    min_area: float = 200.0,
) -> List[CandidateTuple]:
    # Global + tophat are enough for seeds; skip adaptive/primary (slow & often flooded).
    masks = (_global_copper_mask(gray), _tophat_copper_mask(gray))
    merged: List[CandidateTuple] = []
    for mask in masks:
        for candidate in _raw_candidates(mask, quick=True):
            if not _is_stadium_shape(candidate, min_minor=min_minor, min_area=min_area):
                continue
            if not any(_same_component(candidate, existing) for existing in merged):
                merged.append(candidate)
    merged = _deduplicate_candidates(merged)
    return _dominant_size_keep(merged)


def _normalize_long_axis(rect: Tuple) -> Tuple[float, float, float, float, float]:
    (cx, cy), (rw, rh), angle = rect
    rw, rh, angle = float(rw), float(rh), float(angle)
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    angle = (angle + 90.0) % 180.0 - 90.0
    return float(cx), float(cy), rw, rh, angle


def _estimate_orientation(
    gray: np.ndarray,
    binary: Optional[np.ndarray],
    x: int,
    y: int,
    w: int,
    h: int,
) -> Tuple[float, float, float, float, float]:
    """Return (cx, cy, length, width, long-axis angle in degrees)."""
    pad = 8
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    fallback_angle = 0.0 if w >= h else 90.0
    fallback = (x + w / 2.0, y + h / 2.0, float(max(w, h)), float(min(w, h)), fallback_angle)
    roi_g = gray[y0:y1, x0:x1]
    if roi_g.size == 0:
        return fallback

    if binary is not None:
        roi_b = binary[y0:y1, x0:x1]
        if roi_b.dtype != np.uint8:
            roi_b = ((roi_b > 0).astype(np.uint8)) * 255
    else:
        src = roi_g if roi_g.dtype == np.uint8 else np.clip(roi_g, 0, 255).astype(np.uint8)
        roi_b = _global_copper_mask(src)

    contours, _ = cv2.findContours(roi_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        src = roi_g if roi_g.dtype == np.uint8 else np.clip(roi_g, 0, 255).astype(np.uint8)
        threshold = max(50, int(np.percentile(src, 88)))
        _, bw = cv2.threshold(src, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:
        return fallback
    cx, cy, length, width, angle = _normalize_long_axis(cv2.minAreaRect(contour))
    if length < 4 or width < 3:
        return fallback
    return cx + x0, cy + y0, length, width, angle


def _warp_aligned(
    image: np.ndarray,
    cx: float,
    cy: float,
    length: float,
    width: float,
    angle_deg: float,
    scale: float,
    pad: float,
    interpolation: int,
) -> np.ndarray:
    scale = max(1.0, float(scale))
    out_w = max(8, int(round((length + 2.0 * pad) * scale)))
    out_h = max(8, int(round((width + 2.0 * pad) * scale)))
    matrix = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle_deg), scale)
    matrix[0, 2] += out_w * 0.5 - cx
    matrix[1, 2] += out_h * 0.5 - cy
    return cv2.warpAffine(
        image,
        matrix,
        (out_w, out_h),
        flags=interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _aligned_pair(
    gray: np.ndarray,
    binary: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    sr_scale: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float, float]]:
    orient = _estimate_orientation(gray, binary, x, y, w, h)
    cx, cy, length, width, angle = orient
    scale = float(max(1, sr_scale))
    pad = max(6.0, 0.08 * length)
    src_g = gray if gray.dtype == np.uint8 else np.clip(gray, 0, 255).astype(np.uint8)
    aligned_g = _warp_aligned(src_g, cx, cy, length, width, angle, scale, pad, cv2.INTER_CUBIC)
    src_b = binary if binary.dtype == np.uint8 else ((binary > 0).astype(np.uint8) * 255)
    aligned_b = _warp_aligned(src_b, cx, cy, length, width, angle, scale, pad, cv2.INTER_NEAREST)
    if scale >= 2:
        aligned_g = unsharp_gray(aligned_g)
    return aligned_g, aligned_b, orient


def _nms_boxes(
    boxes: List[Tuple[int, int, int, int, float]],
    iou_or_center_frac: float = 0.45,
) -> List[Tuple[int, int, int, int, float]]:
    ordered = sorted(boxes, key=lambda item: -item[4])
    kept: List[Tuple[int, int, int, int, float]] = []
    for x, y, w, h, score in ordered:
        cx, cy = x + w / 2, y + h / 2
        overlap = False
        for kx, ky, kw, kh, _ in kept:
            kcx, kcy = kx + kw / 2, ky + kh / 2
            if abs(cx - kcx) < max(w, kw) * iou_or_center_frac and abs(cy - kcy) < max(h, kh) * iou_or_center_frac:
                overlap = True
                break
        if not overlap:
            kept.append((x, y, w, h, score))
    return kept


def _crop_seed_template(gray: np.ndarray, candidate: CandidateTuple) -> Optional[np.ndarray]:
    x, y, w, h = candidate[:4]
    pad = 3
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    template = gray[y0:y1, x0:x1]
    if template.size == 0 or min(template.shape[:2]) < 8:
        return None
    return template.copy()


def _pick_template(gray: np.ndarray, seeds: List[CandidateTuple], horizontal: bool) -> Optional[np.ndarray]:
    oriented = []
    for candidate in seeds:
        x, y, w, h = candidate[:4]
        if horizontal and w >= h * 1.2:
            oriented.append(candidate)
        elif not horizontal and h >= w * 1.2:
            oriented.append(candidate)
    if not oriented:
        return None

    areas = [c[4] for c in oriented]
    median_area = float(np.median(areas))
    best = min(oriented, key=lambda c: abs(c[4] - median_area))
    return _crop_seed_template(gray, best)


def _angle_bin(angle_deg: float) -> int:
    key = int(round(float(angle_deg) / 15.0) * 15)
    return int((key + 90) % 180) - 90


def _match_rotated_seed_templates(
    gray: np.ndarray,
    seeds: List[CandidateTuple],
    threshold: float,
    scale: float,
) -> List[Tuple[int, int, int, int, float]]:
    """Match AABB templates for seed orientations other than H/V."""
    groups: dict[int, List[CandidateTuple]] = {}
    for seed in seeds:
        *_, angle = _estimate_orientation(gray, None, seed[0], seed[1], seed[2], seed[3])
        key = _angle_bin(angle)
        if abs(key) <= 8 or abs(abs(key) - 90) <= 8:
            continue
        groups.setdefault(key, []).append(seed)

    boxes: List[Tuple[int, int, int, int, float]] = []
    for group in groups.values():
        areas = [item[4] for item in group]
        median_area = float(np.median(areas))
        best = min(group, key=lambda item: abs(item[4] - median_area))
        template = _crop_seed_template(gray, best)
        if template is None:
            continue
        boxes.extend(_match_template_all(gray, template, threshold, scale=scale))
    return boxes


def _match_template_all(
    gray: np.ndarray,
    template: np.ndarray,
    threshold: float,
    scale: float = 1.0,
) -> List[Tuple[int, int, int, int, float]]:
    if template is None or min(template.shape[:2]) < 8:
        return []
    work = gray
    tmpl = template
    if scale < 0.999:
        work = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tmpl = cv2.resize(template, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if min(tmpl.shape[:2]) < 8:
            return []
    if work.shape[0] < tmpl.shape[0] or work.shape[1] < tmpl.shape[1]:
        return []
    response = cv2.matchTemplate(work, tmpl, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(response >= threshold)
    th, tw = tmpl.shape[:2]
    inv = 1.0 / scale
    boxes = [
        (int(round(x * inv)), int(round(y * inv)), int(round(tw * inv)), int(round(th * inv)), float(response[y, x]))
        for x, y in zip(xs, ys)
    ]
    return _nms_boxes(boxes, iou_or_center_frac=0.45)


def _expand_with_template_match(gray: np.ndarray, seeds: List[CandidateTuple]) -> List[Tuple[int, int, int, int]]:
    """Grow a complete set of same-size stadium patterns via template matching."""
    if len(seeds) < 2:
        return [(c[0], c[1], c[2], c[3]) for c in seeds]

    tmpl_h = _pick_template(gray, seeds, horizontal=True)
    tmpl_v = _pick_template(gray, seeds, horizontal=False)

    # Match on a half-resolution pyramid for speed; coordinates are scaled back.
    match_scale = 0.5 if min(gray.shape[:2]) >= 1600 else 1.0

    def _run(threshold: float) -> List[Tuple[int, int, int, int, float]]:
        boxes: List[Tuple[int, int, int, int, float]] = []
        if tmpl_h is not None:
            boxes.extend(_match_template_all(gray, tmpl_h, threshold, scale=match_scale))
        if tmpl_v is not None:
            boxes.extend(_match_template_all(gray, tmpl_v, threshold, scale=match_scale))
        boxes.extend(_match_rotated_seed_templates(gray, seeds, threshold, match_scale))
        return _nms_boxes(boxes, iou_or_center_frac=0.45)

    # Prefer 0.50; only try neighbors if the count looks unstable/underfilled.
    primary = _run(0.50)
    best_boxes = primary
    if len(seeds) >= 40 and (len(primary) + 8 < len(seeds) or len(primary) > len(seeds) + 20):
        alt_hi = _run(0.55)
        alt_lo = _run(0.45)
        scored = []
        for merged in (primary, alt_hi, alt_lo):
            count = len(merged)
            seed_agreement = -abs(count - len(seeds))
            scored.append((seed_agreement + min(count, len(seeds) + 15) * 0.01, merged))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_boxes = scored[0][1]

    if not best_boxes:
        return [(c[0], c[1], c[2], c[3]) for c in seeds]

    if len(best_boxes) + 5 < len(seeds):
        seed_boxes = [(c[0], c[1], c[2], c[3], 1.0) for c in seeds]
        best_boxes = _nms_boxes(best_boxes + seed_boxes, iou_or_center_frac=0.45)

    return [(x, y, w, h) for x, y, w, h, _ in best_boxes]


def _circular_max_false_run(flags: List[bool]) -> int:
    """Longest consecutive False run on a circular sequence."""
    if not flags:
        return 0
    n = len(flags)
    doubled = flags + flags
    run = 0
    best = 0
    for value in doubled:
        if not value:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return min(best, n)


def _ring_open_from_flags(flags: List[bool], min_gap: int = 3) -> bool:
    """True when an otherwise-present ring has a contiguous fully-covered gap."""
    if not flags:
        return True
    present_n = sum(1 for value in flags if value)
    if present_n < max(8, len(flags) // 4):
        return True
    return _circular_max_false_run(flags) >= min_gap


def _track_not_covered(
    profile: np.ndarray,
    center: float,
    tol: float,
    cover_level: float,
    ring_level: float,
) -> bool:
    """True if track locus is still bright (not fully covered by black mask)."""
    if profile.size == 0 or ring_level <= cover_level + 2:
        return False
    lo = max(0, int(np.floor(center - tol)))
    hi = min(len(profile), int(np.ceil(center + tol)) + 1)
    if lo >= hi:
        return False
    peak = float(np.max(profile[lo:hi]))
    gate = cover_level + 0.25 * (ring_level - cover_level)
    return peak >= gate


def _cover_overflow(
    gray: np.ndarray,
    binary: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    sr_scale: int = 1,
    already_aligned: bool = False,
) -> bool:
    """True when black cover fully occludes inner or outer track.

    Product rule: if black covering completely covers a runway ring, treat it as
    overflow into the region between the two rings. Thinning alone is NOT enough;
    only a contiguous fully-dark gap on the track counts.
    """
    if not already_aligned:
        aligned_g, aligned_b, _orient = _aligned_pair(gray, binary, x, y, w, h, sr_scale)
        ah, aw = aligned_g.shape[:2]
        return _cover_overflow(
            aligned_g, aligned_b, 0, 0, aw, ah, sr_scale=1, already_aligned=True
        )

    padding = 4
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(gray.shape[1], x + w + padding)
    y1 = min(gray.shape[0], y + h + padding)
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    fb = binary[y0:y1, x0:x1]
    if roi.size == 0:
        return True

    cover_level = float(np.percentile(roi, 25))
    ring_level = float(np.percentile(roi, 90))
    if ring_level - cover_level < 8:
        return True

    thr = cover_level + 0.45 * (ring_level - cover_level)
    copper = (roi >= thr) | (fb > 0)
    if float(copper.mean()) < 0.006:
        return True

    rh, rw = roi.shape
    horizontal = w >= h
    outer_dists: List[float] = []
    inner_dists: List[float] = []
    peak_vals: List[float] = []

    def _collect_half(half: np.ndarray, toward_end: bool) -> None:
        outer_i, inner_i = _dual_peaks_on_half(half, toward_end=toward_end)
        if outer_i is None or inner_i is None:
            return
        half_len = float(len(half))
        peak_vals.append(float(half[outer_i]))
        peak_vals.append(float(half[inner_i]))
        if toward_end:
            outer_dists.append(float(outer_i))
            inner_dists.append(float(inner_i))
        else:
            outer_dists.append(half_len - 1.0 - float(outer_i))
            inner_dists.append(half_len - 1.0 - float(inner_i))

    if horizontal:
        cy = rh // 2
        for col in range(int(rw * 0.2), int(rw * 0.8), 2):
            profile = roi[:, col]
            _collect_half(profile[:cy], toward_end=False)
            _collect_half(profile[cy:], toward_end=True)
    else:
        cx = rw // 2
        for row in range(int(rh * 0.2), int(rh * 0.8), 2):
            profile = roi[row, :]
            _collect_half(profile[:cx], toward_end=False)
            _collect_half(profile[cx:], toward_end=True)

    if peak_vals:
        ring_level = max(ring_level, float(np.median(peak_vals)))

    if len(outer_dists) < 6 or len(inner_dists) < 6:
        present: List[bool] = []
        gate = cover_level + 0.25 * (ring_level - cover_level)
        if horizontal:
            cy = rh // 2
            for col in range(int(rw * 0.1), int(rw * 0.9)):
                alive = float(np.max(roi[1:cy, col])) >= gate and float(np.max(roi[cy:rh - 1, col])) >= gate
                present.append(alive)
        else:
            cx = rw // 2
            for row in range(int(rh * 0.1), int(rh * 0.9)):
                alive = float(np.max(roi[row, 1:cx])) >= gate and float(np.max(roi[row, cx:rw - 1])) >= gate
                present.append(alive)
        return _ring_open_from_flags(present, min_gap=4)

    d_outer = float(np.median(outer_dists))
    d_inner = float(np.median(inner_dists))
    if d_outer < d_inner:
        d_outer, d_inner = d_inner, d_outer
    sep = max(3.0, d_outer - d_inner)
    tol = max(3.0, min(7.0, 0.45 * sep))

    inner_flags: List[bool] = []
    outer_flags: List[bool] = []

    def _append(outer_alive: bool, inner_alive: bool) -> None:
        outer_flags.append(outer_alive)
        inner_flags.append(inner_alive)

    # Mid-band cross-sections (avoid rounded-end geometry mismatch / ROI padding).
    if horizontal:
        cy = rh // 2
        x_lo, x_hi = int(rw * 0.18), int(rw * 0.82)
        for col in range(x_lo, x_hi):
            top = roi[:cy, col]
            _append(
                _track_not_covered(top, (cy - 1) - d_outer, tol, cover_level, ring_level),
                _track_not_covered(top, (cy - 1) - d_inner, tol, cover_level, ring_level),
            )
        for col in range(x_hi - 1, x_lo - 1, -1):
            bot = roi[cy:, col]
            _append(
                _track_not_covered(bot, d_outer, tol, cover_level, ring_level),
                _track_not_covered(bot, d_inner, tol, cover_level, ring_level),
            )
    else:
        cx = rw // 2
        y_lo, y_hi = int(rh * 0.18), int(rh * 0.82)
        for row in range(y_lo, y_hi):
            left = roi[row, :cx]
            _append(
                _track_not_covered(left, (cx - 1) - d_outer, tol, cover_level, ring_level),
                _track_not_covered(left, (cx - 1) - d_inner, tol, cover_level, ring_level),
            )
        for row in range(y_hi - 1, y_lo - 1, -1):
            right = roi[row, cx:]
            _append(
                _track_not_covered(right, d_outer, tol, cover_level, ring_level),
                _track_not_covered(right, d_inner, tol, cover_level, ring_level),
            )

    return _ring_open_from_flags(inner_flags, min_gap=5) or _ring_open_from_flags(outer_flags, min_gap=5)


# Back-compat alias.
_broken_ring = _cover_overflow


def detect_rings(
    image_bgr_or_gray: np.ndarray,
    source_dpi: float,
    threshold_width_px: float,
    sr_scale: int = 1,
    allow_small: bool = False,
) -> DetectionOutput:
    """Detect dual stadium runway rings and inspect black-cover overflow.

    Defect model
    ------------
    Bright dual runway rings sit in black covering. If black cover fully
    occludes a ring track, that is treated as overflow into the inter-ring
    region (``is_broken``). Remaining ring width is also measured; values
    below ``threshold_width_px`` flag partial covering / thinning.

    ``sr_scale`` locally upsamples each ring (deskew + cubic/unsharp) so a
    single whole-board frame can still measure hairline copper width.
    ``allow_small`` relaxes size gates for distant whole-board shots.
    """
    from concurrent.futures import ThreadPoolExecutor

    sr_scale = int(max(1, min(4, sr_scale)))
    min_minor = 6.0 if allow_small else 8.0
    min_area = 80.0 if allow_small else 200.0

    gray = ensure_grayscale(image_bgr_or_gray)
    measure_mask = _global_copper_mask(gray)

    seeds = _contour_stadium_seeds(gray, min_minor=min_minor, min_area=min_area)
    boxes = _expand_with_template_match(gray, seeds) if seeds else []
    if not boxes:
        boxes = [(c[0], c[1], c[2], c[3]) for c in seeds]

    if float((measure_mask > 0).mean()) < 0.002:
        measure_mask = _primary_copper_mask(gray)

    def _measure_one(box: Tuple[int, int, int, int]):
        x, y, w, h = box
        inner_min, outer_min, inner_med, outer_med, irr_min, irr_max, irr_ratio = _measure_dual_ring_widths(
            gray, measure_mask, x, y, w, h, sr_scale=sr_scale
        )
        is_broken = _cover_overflow(gray, measure_mask, x, y, w, h, sr_scale=sr_scale)
        orient = _estimate_orientation(gray, measure_mask, x, y, w, h)
        return (
            x, y, w, h, inner_min, outer_min, inner_med, outer_med,
            irr_min, irr_max, irr_ratio, is_broken, orient,
        )

    workers = min(8, max(2, (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        measured = list(pool.map(_measure_one, boxes))

    widths: List[float] = []
    for item in measured:
        for value in item[4:8]:
            if value >= 0.5:
                widths.append(value)

    if widths:
        median_w = float(np.median(widths))
        inliers = [value for value in widths if 0.45 * median_w <= value <= 1.8 * median_w]
        baseline_trace_px = float(np.median(inliers)) if inliers else median_w
    else:
        baseline_trace_px = 1.0
    baseline_minor_px = float(np.median([min(w, h) for _, _, w, h, *_ in measured])) if measured else 0.0

    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    results: List[RingResult] = []
    components: List[ComponentResult] = []

    for index, (
        x, y, w, h, inner_min, outer_min, inner_med, outer_med,
        irr_min, irr_max, irr_ratio, is_broken, orient,
    ) in enumerate(measured, start=1):
        cx, cy, length, width, angle = orient
        minor_px = float(min(w, h))
        major_px = float(max(w, h))
        positive = [value for value in (inner_min, outer_min) if value >= 0.5]
        min_width = float(min(positive)) if positive else 0.0
        inner_dpi = source_dpi * (inner_min / max(1.0, baseline_trace_px)) if inner_min >= 0.5 else 0.0
        outer_dpi = source_dpi * (outer_min / max(1.0, baseline_trace_px)) if outer_min >= 0.5 else 0.0
        effective_dpi = source_dpi * (min_width / max(1.0, baseline_trace_px)) if min_width >= 0.5 else 0.0
        typical = [value for value in (inner_med, outer_med) if value >= 0.5]
        typical_med = float(np.median(typical)) if typical else 0.0
        narrow_vs_typical = (typical_med / max(min_width, 1e-6)) if min_width >= 0.5 and typical_med > 0 else 1.0
        is_irregular = narrow_vs_typical > 1.8 or (irr_ratio > 3.5 and irr_max > 0)
        is_below = min_width < threshold_width_px if min_width >= 0.5 else True

        result = RingResult(
            index=index,
            x=x,
            y=y,
            w=w,
            h=h,
            area=int(w * h),
            minor_px=minor_px,
            major_px=major_px,
            inner_line_width_px=inner_min,
            outer_line_width_px=outer_min,
            min_line_width_px=min_width,
            inner_effective_dpi=inner_dpi,
            outer_effective_dpi=outer_dpi,
            effective_dpi=effective_dpi,
            is_below_threshold=is_below,
            is_broken=is_broken,
            component_index=index,
            irregularity_min_px=irr_min if irr_min > 0 else min_width,
            irregularity_max_px=irr_max if irr_max > 0 else max(inner_med, outer_med),
            irregularity_std_px=0.0,
            irregularity_ratio=max(irr_ratio, narrow_vs_typical),
            is_irregular_edges=is_irregular,
            angle_deg=float(angle),
            cx=float(cx),
            cy=float(cy),
            sr_scale=sr_scale,
            inner_width_um=px_to_um(inner_min, source_dpi),
            outer_width_um=px_to_um(outer_min, source_dpi),
            min_width_um=px_to_um(min_width, source_dpi),
        )
        results.append(result)
        components.append(ComponentResult(index=index, ring_indices=(index,), x=x, y=y, w=w, h=h))

        if result.is_broken:
            color, label = (0, 149, 255), "溢出"
        elif result.is_below_threshold:
            color, label = (20, 20, 230), f"细{result.min_line_width_px:.1f}"
        elif result.is_irregular_edges:
            color, label = (200, 50, 200), "IRREG"
        else:
            color, label = (0, 185, 185), f"{result.min_line_width_px:.1f}"
        box_pts = cv2.boxPoints(((cx, cy), (length, width), angle))
        cv2.drawContours(annotated, [np.intp(box_pts)], 0, color, 2)
        if label:
            cv2.putText(
                annotated,
                label,
                (x, max(14, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            annotated,
            f"#{index}",
            (x, min(annotated.shape[0] - 4, y + h + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (240, 140, 0),
            1,
            cv2.LINE_AA,
        )

    return DetectionOutput(annotated, results, components, baseline_minor_px, sr_scale=sr_scale)
