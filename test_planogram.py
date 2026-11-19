import csv
import unittest
from pathlib import Path

from generate_sample_shelf import build_sample_assets
from planogram_checker import compute_iou, run_pipeline


class TestShelfPlanogramPOC(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).parent
        build_sample_assets(self.root)

    def test_iou_exact_and_disjoint(self) -> None:
        self.assertEqual(compute_iou((0, 0, 100, 100), (0, 0, 100, 100)), 1.0)
        self.assertEqual(compute_iou((0, 0, 50, 50), (100, 100, 150, 150)), 0.0)

    def test_end_to_end_pipeline_flags_mismatches_and_missing(self) -> None:
        results = run_pipeline(
            self.root / "sample_shelf.png",
            self.root / "sample_planogram.json",
            self.root,
        )
        by_id = {r.zone_id: r for r in results}
        self.assertEqual(len(results), 8)
        self.assertEqual(by_id["R1-S1"].status, "MATCH")
        self.assertEqual(by_id["R1-S3"].status, "MISPLACED")
        self.assertEqual(by_id["R1-S3"].detected_sku, "SKU_COLA")
        self.assertEqual(by_id["R2-S2"].status, "MISSING")
        self.assertEqual(by_id["R2-S4"].status, "MISPLACED")

        self.assertTrue((self.root / "sample_overlay_result.png").exists())
        self.assertTrue((self.root / "dashboard.html").exists())
        with (self.root / "planogram_mismatches.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 8)


if __name__ == "__main__":
    unittest.main()
