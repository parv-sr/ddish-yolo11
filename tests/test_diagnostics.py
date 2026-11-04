"""Synthetic tests for Fusion auditing and nucleus-wise counting."""

import unittest

import torch

from src.evaluation.audit_fusion import audit_image
from src.evaluation.counts import CountMetric, compare_image
from src.evaluation.validation import greedy_matches
from src.inference.postprocess import Detections, PatchPredictions


class DiagnosticsTests(unittest.TestCase):
    def test_greedy_match_is_unique(self) -> None:
        overlaps = torch.tensor([[0.90, 0.10], [0.85, 0.20], [0.10, 0.91]])
        scores = torch.tensor([0.90, 0.80, 0.70])
        self.assertEqual(greedy_matches(overlaps, scores), {0: 0, 2: 1})

    def test_fusion_audit_stages_and_false_positives(self) -> None:
        centers = [1.0, 5.0, 9.0, 13.0, 17.0, 25.0]
        boxes = torch.tensor([[x, 1.0, 2.0, 2.0] for x in centers]).T
        scores = torch.tensor([
            [0.90, 0.20, 0.90, 0.05, 0.0001, 0.70],
            [0.01, 0.90, 0.01, 0.01, 0.0001, 0.01],
            [0.01, 0.05, 0.01, 0.01, 0.0001, 0.01],
        ])
        raw = torch.cat((boxes, scores), dim=0)
        gt_boxes = torch.tensor([[x - 1, 0, x + 1, 2] for x in centers[:5] + [21.0, 25.0]])
        gt_classes = torch.tensor([0, 0, 0, 0, 0, 0, 1])
        final = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.90, 0.0], [24.0, 0.0, 26.0, 2.0, 0.70, 0.0]])

        stages, spurious = audit_image(raw, final, torch.tensor([0, 5]), gt_boxes, gt_classes, 0.25, 0.001, 0.50)
        self.assertEqual([row["status"] for row in stages], [
            "matched", "other_class_wins", "suppressed_or_capped", "below_threshold", "below_floor", "no_localized_candidate"
        ])
        self.assertEqual(len(spurious), 1)
        self.assertEqual(spurious[0]["category"], "overlaps_other_signal_gt")
        self.assertEqual(spurious[0]["best_gt_class"], "Chr17")

    def test_counting_reports_missed_and_extra_nuclei(self) -> None:
        gt_masks = torch.zeros((2, 8, 8), dtype=torch.bool)
        gt_masks[0, :4, :4] = True
        gt_masks[1, 4:, 4:] = True
        pred_masks = torch.zeros((2, 8, 8), dtype=torch.bool)
        pred_masks[0] = gt_masks[0]
        pred_masks[1, 4:, :2] = True
        prediction = PatchPredictions(
            nuclei=Detections(
                boxes=torch.tensor([[0, 0, 4, 4], [0, 4, 2, 8]], dtype=torch.float32),
                scores=torch.tensor([0.9, 0.8]),
                classes=torch.zeros(2, dtype=torch.long),
                masks=pred_masks,
            ),
            signals=Detections(
                boxes=torch.tensor([[1, 1, 2, 2], [0, 5, 1, 6]], dtype=torch.float32),
                scores=torch.tensor([0.9, 0.9]),
                classes=torch.tensor([0, 2]),
            ),
        )
        target = {
            "path": "sample.png", "source_group": "field",
            "nucleus_masks": gt_masks,
            "nucleus_boxes": torch.tensor([[0.25, 0.25, 0.5, 0.5], [0.75, 0.75, 0.5, 0.5]]),
            "signal_boxes": torch.tensor([[0.1875, 0.1875, 0.125, 0.125], [0.6875, 0.6875, 0.125, 0.125], [0.6875, 0.6875, 0.125, 0.125]]),
            "signal_classes": torch.tensor([0, 1, 2]),
        }

        report = compare_image(prediction, target, 0.25, 0.50, 0.50)
        self.assertEqual(report["coverage"], {"gt_nuclei": 2, "matched_nuclei": 1, "missed_nuclei": 1, "extra_predicted_nuclei": 1})
        self.assertEqual(report["rows"][0]["gt_fusion"], 1)
        self.assertEqual(report["rows"][1]["pred_chr17"], 0)
        self.assertEqual(report["extra_counts"].tolist(), [0, 0, 1, 1, 0])
        self.assertEqual(report["errors"]["predicted_masks_patch_totals"].tolist(), [[0, -1, 0, 0, -1]])
        self.assertEqual(report["association"]["ground_truth"]["assigned"], 3)

    def test_empty_count_metric(self) -> None:
        metric = CountMetric()
        metric.update(torch.empty((0, 5)))
        self.assertEqual(metric.summary()["samples"], 0)
        self.assertIsNone(metric.summary()["classes"]["fusion"]["mae"])


if __name__ == "__main__":
    unittest.main()
