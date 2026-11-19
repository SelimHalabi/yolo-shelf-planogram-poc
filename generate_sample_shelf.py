"""
Generate a realistic synthetic retail shelf test image and matching planogram JSON.
Two slots intentionally contain misplaced/swapped SKUs and one slot is empty
to demonstrate green (MATCH) vs red (MISPLACED / MISSING) detection overlays.
"""

import json
from pathlib import Path
import cv2
import numpy as np


SKU_COLORS_BGR = {
    "SKU_COLA": (35, 35, 220),     # Vivid red
    "SKU_WATER": (220, 120, 25),   # Vivid blue
    "SKU_JUICE": (25, 195, 235),   # Vivid yellow-orange
    "SKU_TEA": (55, 180, 55),      # Vivid green
}


def build_sample_assets(out_dir: Path) -> None:
    width, height = 1000, 560
    img = np.full((height, width, 3), 235, dtype=np.uint8)

    # Draw shelf backing and two shelf planks
    cv2.rectangle(img, (40, 30), (960, 520), (210, 214, 220), -1)
    cv2.rectangle(img, (40, 250), (960, 270), (90, 75, 65), -1)
    cv2.rectangle(img, (40, 490), (960, 515), (90, 75, 65), -1)

    expected_layout = [
        ("R1-S1", 1, 1, "SKU_COLA", (80, 75, 250, 248)),
        ("R1-S2", 1, 2, "SKU_WATER", (300, 75, 470, 248)),
        ("R1-S3", 1, 3, "SKU_JUICE", (520, 75, 690, 248)),
        ("R1-S4", 1, 4, "SKU_TEA", (740, 75, 910, 248)),
        ("R2-S1", 2, 1, "SKU_COLA", (80, 315, 250, 488)),
        ("R2-S2", 2, 2, "SKU_WATER", (300, 315, 470, 488)),
        ("R2-S3", 2, 3, "SKU_JUICE", (520, 315, 690, 488)),
        ("R2-S4", 2, 4, "SKU_TEA", (740, 315, 910, 488)),
    ]

    # Actual shelf items: R1-S3 has COLA instead of JUICE (MISPLACED), R2-S2 is empty (MISSING),
    # R2-S4 has JUICE instead of TEA (MISPLACED).
    actual_items = {
        "R1-S1": "SKU_COLA",
        "R1-S2": "SKU_WATER",
        "R1-S3": "SKU_COLA",
        "R1-S4": "SKU_TEA",
        "R2-S1": "SKU_COLA",
        "R2-S2": None,
        "R2-S3": "SKU_JUICE",
        "R2-S4": "SKU_JUICE",
    }

    zones_json = []
    for zid, row, slot, exp_sku, (x1, y1, x2, y2) in expected_layout:
        zones_json.append(
            {
                "zone_id": zid,
                "shelf_row": row,
                "slot": slot,
                "expected_sku": exp_sku,
                "bbox_norm": [
                    round((x1 - 10) / width, 4),
                    round((y1 - 10) / height, 4),
                    round((x2 + 10) / width, 4),
                    round((y2 + 10) / height, 4),
                ],
            }
        )
        placed = actual_items[zid]
        if placed:
            color = SKU_COLORS_BGR[placed]
            cv2.rectangle(img, (x1 + 15, y1 + 10), (x2 - 15, y2 - 2), color, -1)
            cv2.rectangle(img, (x1 + 15, y1 + 10), (x2 - 15, y2 - 2), (40, 40, 40), 2)
            cv2.putText(
                img,
                placed.replace("SKU_", ""),
                (x1 + 38, (y1 + y2) // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(out_dir / "sample_shelf.png"), img)
    (out_dir / "sample_planogram.json").write_text(
        json.dumps({"shelf_id": "SHELF-A01", "zones": zones_json}, indent=2),
        encoding="utf-8",
    )
    print("Generated sample_shelf.png and sample_planogram.json")


if __name__ == "__main__":
    build_sample_assets(Path(__file__).parent)
