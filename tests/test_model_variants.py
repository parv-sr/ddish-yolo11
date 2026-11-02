"""Configuration and architecture tests."""

import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

import torch
from ultralytics.nn.modules.head import Detect, Segment

from src.model_variants.classifier_transfer import transfer_classifier_trunks
from src.model_variants.p2_segmentation import prepend_p2_segment_head
from src.model_variants.registry import MODEL_REGISTRY, ModelConfig, build_model


class ModelConfigTests(unittest.TestCase):
    def test_legacy_checkpoint(self) -> None:
        self.assertEqual(ModelConfig.from_checkpoint({"variant": "n", "p2_detection": True}), ModelConfig(p2_detection=True))
        self.assertEqual(ModelConfig.from_checkpoint({}), ModelConfig())

    def test_new_checkpoint(self) -> None:
        config: ModelConfig = ModelConfig(name="classifier_transfer", size="s", p2_detection=True)
        self.assertEqual(ModelConfig.from_checkpoint({"model_config": config.to_dict()}), config)

    def test_invalid_config(self) -> None:
        for kwargs in ({"name": "unknown"}, {"size": "x"}, {"name": "p2_segmentation"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelConfig(**kwargs)

    def test_factory(self) -> None:
        constructor: Mock = Mock(return_value=object())
        with patch.dict(MODEL_REGISTRY, {"baseline": constructor}):
            build_model(ModelConfig(name="baseline", size="s", p2_detection=True))

        constructor.assert_called_once_with(variant="s", p2_detection=True)


class HeadTests(unittest.TestCase):
    def test_classifier_trunk_and_new_p2(self) -> None:
        source: Detect = Detect(nc=80, ch=(64, 128, 256))
        target: Detect = Detect(nc=3, ch=(64, 64, 128, 256))
        source.stride = torch.tensor([8.0, 16.0, 32.0])
        target.stride = torch.tensor([4.0, 8.0, 16.0, 32.0])
        p2_state = deepcopy(target.cv3[0].state_dict())

        self.assertGreater(transfer_classifier_trunks(target, source), 0)
        self.assertEqual(target.cv3[1][-1].out_channels, 3)
        self.assertEqual(target.cv3[1][-1].in_channels, source.cv3[0][-1].in_channels)

        for key, value in source.cv3[0][:-1].state_dict().items():
            self.assertTrue(torch.equal(value, target.cv3[1][:-1].state_dict()[key]))

        for key, value in p2_state.items():
            self.assertTrue(torch.equal(value, target.cv3[0].state_dict()[key]))

    def test_p2_segmentation_transplant(self) -> None:
        source: Segment = Segment(nc=1, nm=32, npr=64, ch=(64, 128, 256))
        source.stride = torch.tensor([8.0, 16.0, 32.0])
        target: Segment = prepend_p2_segment_head(source)
        self.assertEqual(target.nl, 4)
        self.assertEqual(target.stride.tolist(), [4.0, 8.0, 16.0, 32.0])

        for name in ("cv2", "cv3", "cv4"):
            for index in range(3):
                old_state = getattr(source, name)[index].state_dict()
                new_state = getattr(target, name)[index + 1].state_dict()
                self.assertTrue(all(torch.equal(value, new_state[key]) for key, value in old_state.items()))


if __name__ == "__main__":
    unittest.main()
