"""
YOLO & OpenCV Shelf Planogram Compliance Checker.

Detects products on shelf camera images, maps each detection against a JSON
planogram layout, renders green/red compliance overlays, and exports a CSV
mismatch table plus an interactive HTML inspection dashboard.
"""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Protocol

import cv2
import numpy as np


@dataclass
class Detection:
    """Single detected product bounding box in pixel coordinates."""

    label: str
    confidence: float
    bbox_px: tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class PlanogramZone:
    """Expected SKU placement zone normalized to [0, 1] image coordinates."""

    zone_id: str
    shelf_row: int
    slot: int
    expected_sku: str
    bbox_norm: tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass
class ZoneCheckResult:
    """Compliance evaluation outcome for one planogram zone."""

    zone_id: str
    shelf_row: int
    slot: int
    expected_sku: str
    detected_sku: Optional[str]
    confidence: float
    status: str  # "MATCH", "MISPLACED", "MISSING"
    iou: float
    bbox_px: tuple[int, int, int, int]


class ProductDetector(Protocol):
    """Protocol for pluggable shelf-product detectors."""

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        ...


class UltralyticsYOLODetector:
    """Ultralytics YOLOv8/v11 object detector wrapper."""

    def __init__(self, weights_path: str = "yolov8n.pt", conf_threshold: float = 0.25) -> None:
        from ultralytics import YOLO  # lazy import so OpenCV-only mode stays lightweight

        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(image_bgr, conf=self.conf_threshold, verbose=False)
        detections: List[Detection] = []
        for res in results:
            names = res.names
            for box in res.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        label=str(names.get(cls_id, cls_id)),
                        confidence=round(conf, 4),
                        bbox_px=(x1, y1, x2, y2),
                    )
                )
        return detections


class HSVContourSKUDetector:
    """
    Deterministic OpenCV HSV color-band SKU detector for offline demos & unit tests.
    Maps distinct package color signatures to SKU labels without external model weights.
    """

    DEFAULT_SIGNATURES = {
        "SKU_COLA": ((0, 140, 90), (10, 255, 255)),
        "SKU_WATER": ((100, 130, 90), (125, 255, 255)),
        "SKU_JUICE": ((15, 140, 110), (32, 255, 255)),
        "SKU_TEA": ((45, 110, 70), (80, 255, 255)),
    }

    def __init__(self, signatures: Optional[dict] = None, min_area: int = 1800) -> None:
        self.signatures = signatures or self.DEFAULT_SIGNATURES
        self.min_area = min_area

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        detections: List[Detection] = []
        kernel = np.ones((5, 5), np.uint8)

        for sku, (low, high) in self.signatures.items():
            mask = cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append(
                    Detection(
                        label=sku,
                        confidence=0.96,
                        bbox_px=(x, y, x + w, y + h),
                    )
                )
        return detections


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union (IoU) between two pixel bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return round(inter / union, 4) if union > 0 else 0.0


def load_planogram(path: Path) -> List[PlanogramZone]:
    """Load planogram zones from JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        PlanogramZone(
            zone_id=item["zone_id"],
            shelf_row=int(item["shelf_row"]),
            slot=int(item["slot"]),
            expected_sku=item["expected_sku"],
            bbox_norm=tuple(float(v) for v in item["bbox_norm"]),  # type: ignore[arg-type]
        )
        for item in raw["zones"]
    ]


def evaluate_planogram(
    image_shape: tuple[int, int, int],
    zones: Iterable[PlanogramZone],
    detections: List[Detection],
    min_iou: float = 0.15,
) -> List[ZoneCheckResult]:
    """Match detected SKUs to normalized planogram zones and classify compliance."""
    h, w = image_shape[:2]
    results: List[ZoneCheckResult] = []

    for zone in zones:
        nx1, ny1, nx2, ny2 = zone.bbox_norm
        zone_px = (int(nx1 * w), int(ny1 * h), int(nx2 * w), int(ny2 * h))

        best_det: Optional[Detection] = None
        best_iou = 0.0
        for det in detections:
            score = compute_iou(zone_px, det.bbox_px)
            if score > best_iou:
                best_iou = score
                best_det = det

        if best_det is None or best_iou < min_iou:
            results.append(
                ZoneCheckResult(
                    zone_id=zone.zone_id,
                    shelf_row=zone.shelf_row,
                    slot=zone.slot,
                    expected_sku=zone.expected_sku,
                    detected_sku=None,
                    confidence=0.0,
                    status="MISSING",
                    iou=0.0,
                    bbox_px=zone_px,
                )
            )
            continue

        status = "MATCH" if best_det.label == zone.expected_sku else "MISPLACED"
        results.append(
            ZoneCheckResult(
                zone_id=zone.zone_id,
                shelf_row=zone.shelf_row,
                slot=zone.slot,
                expected_sku=zone.expected_sku,
                detected_sku=best_det.label,
                confidence=best_det.confidence,
                status=status,
                iou=best_iou,
                bbox_px=zone_px,
            )
        )
    return results


def render_overlay(image_bgr: np.ndarray, results: List[ZoneCheckResult]) -> np.ndarray:
    """Draw green outlines for compliant zones and red outlines for mismatches/missing."""
    canvas = image_bgr.copy()
    for r in results:
        x1, y1, x2, y2 = r.bbox_px
        is_ok = r.status == "MATCH"
        color = (46, 175, 62) if is_ok else (36, 48, 228)  # BGR: green vs red
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 4)

        exp_short = r.expected_sku.replace("SKU_", "")
        det_short = (r.detected_sku or "NONE").replace("SKU_", "")
        lines = (
            [f"{r.zone_id}: OK ({exp_short})"]
            if is_ok
            else [f"{r.zone_id}: {r.status}", f"exp {exp_short} | got {det_short}"]
        )
        font_scale = 0.44
        line_h = 16
        max_tw = max(cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] for ln in lines)
        box_h = len(lines) * line_h + 6
        top_y = max(0, y1 - box_h)
        cv2.rectangle(canvas, (x1, top_y), (min(canvas.shape[1] - 1, max(x2, x1 + max_tw + 10)), y1), color, -1)
        for idx, ln in enumerate(lines):
            cv2.putText(
                canvas,
                ln,
                (x1 + 5, top_y + 14 + idx * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return canvas


def export_csv(results: List[ZoneCheckResult], out_csv: Path) -> None:
    """Write structured compliance & mismatch report to CSV."""
    fieldnames = [
        "zone_id", "shelf_row", "slot", "expected_sku", "detected_sku",
        "status", "confidence", "iou", "bbox_px",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["detected_sku"] = row["detected_sku"] or "NONE"
            writer.writerow(row)


def export_html_dashboard(
    results: List[ZoneCheckResult],
    overlay_image_name: str,
    out_html: Path,
) -> None:
    """Generate a self-contained HTML visual inspection report."""
    total = len(results)
    matches = sum(1 for r in results if r.status == "MATCH")
    mismatches = total - matches
    compliance_pct = round((matches / total) * 100, 1) if total else 0.0

    rows_html = []
    for r in results:
        badge_cls = "ok" if r.status == "MATCH" else "bad"
        rows_html.append(
            f"<tr>"
            f"<td>{html.escape(r.zone_id)}</td>"
            f"<td>Row {r.shelf_row} / Slot {r.slot}</td>"
            f"<td>{html.escape(r.expected_sku)}</td>"
            f"<td>{html.escape(r.detected_sku or 'NONE')}</td>"
            f"<td><span class='badge {badge_cls}'>{html.escape(r.status)}</span></td>"
            f"<td>{r.confidence:.2f}</td>"
            f"<td>{r.iou:.2f}</td>"
            f"</tr>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Shelf Planogram Compliance Dashboard</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem; background: #f8fafc; color: #0f172a; }}
  .cards {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; }}
  .card {{ background: #fff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 1rem 1.4rem; min-width: 150px; }}
  .card .val {{ font-size: 1.8rem; font-weight: 700; color: #1e3a8a; }}
  img {{ max-width: 100%; border: 1px solid #cbd5e1; border-radius: 8px; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 0.6rem 0.8rem; text-align: left; }}
  th {{ background: #1e293b; color: #fff; }}
  .badge {{ padding: 0.2rem 0.55rem; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }}
  .badge.ok {{ background: #dcfce7; color: #166534; }}
  .badge.bad {{ background: #fee2e2; color: #991b1b; }}
</style>
</head>
<body>
  <h1>Shelf Planogram Compliance Dashboard (POC)</h1>
  <div class="cards">
    <div class="card"><div>Total Zones</div><div class="val">{total}</div></div>
    <div class="card"><div>Compliant (Green)</div><div class="val">{matches}</div></div>
    <div class="card"><div>Mismatches (Red)</div><div class="val">{mismatches}</div></div>
    <div class="card"><div>Compliance Rate</div><div class="val">{compliance_pct}%</div></div>
  </div>
  <img src="{html.escape(overlay_image_name)}" alt="Shelf Compliance Overlay" />
  <table>
    <thead>
      <tr><th>Zone</th><th>Position</th><th>Expected SKU</th><th>Detected SKU</th><th>Status</th><th>Conf</th><th>IoU</th></tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>"""
    out_html.write_text(doc, encoding="utf-8")


def run_pipeline(
    image_path: Path,
    planogram_path: Path,
    out_dir: Path,
    detector: Optional[ProductDetector] = None,
) -> List[ZoneCheckResult]:
    """Execute end-to-end shelf compliance check and save overlay, CSV, and HTML dashboard."""
    out_dir.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    zones = load_planogram(planogram_path)
    active_detector = detector or HSVContourSKUDetector()
    detections = active_detector.detect(image_bgr)
    results = evaluate_planogram(image_bgr.shape, zones, detections)

    overlay = render_overlay(image_bgr, results)
    overlay_name = "sample_overlay_result.png"
    cv2.imwrite(str(out_dir / overlay_name), overlay)
    export_csv(results, out_dir / "planogram_mismatches.csv")
    export_html_dashboard(results, overlay_name, out_dir / "dashboard.html")
    return results


if __name__ == "__main__":
    root = Path(__file__).parent
    res = run_pipeline(root / "sample_shelf.png", root / "sample_planogram.json", root)
    for item in res:
        print(f"{item.zone_id}: {item.status} (expected={item.expected_sku}, got={item.detected_sku}, iou={item.iou})")
