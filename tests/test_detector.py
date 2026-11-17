"""Detector adapter behavior tested without loading YOLO weights."""
import numpy as np

from fewshot_vi.detector.yolo_detector import YoloDetector


class _Array:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Boxes:
    def __init__(self):
        self.xyxy = _Array(np.array([[1, 2, 20, 30], [5, 6, 40, 50]], np.float32))
        self.conf = _Array(np.array([0.7, 0.9], np.float32))
        self.cls = _Array(np.array([39, 41], np.float32))

    def __len__(self):
        return 2


class _Model:
    def __init__(self):
        self.kwargs = None
        self.names = {39: "bottle", 41: "cup"}
        self.boxes = _Boxes()

    def predict(self, frame, **kwargs):
        self.kwargs = kwargs
        return [type("Result", (), {"boxes": self.boxes})()]


def test_detector_uses_configured_resolution_and_keeps_all_classes():
    detector = YoloDetector(conf=0.2, max_objects=2, imgsz=320)
    model = _Model()
    detector._model = model
    boxes = detector.detect(np.zeros((64, 64, 3), np.uint8))
    assert model.kwargs["imgsz"] == 320
    assert model.kwargs["agnostic_nms"] is True
    assert len(boxes) == 2 and boxes[0].score > boxes[1].score
    assert boxes[0].class_name == "cup" and boxes[0].class_id == 41
    assert boxes[1].class_name == "bottle" and boxes[1].class_id == 39


def test_detector_keeps_legacy_box_providers_without_class_ids():
    detector = YoloDetector(conf=0.2, max_objects=2)
    model = _Model()
    del model.boxes.cls
    detector._model = model
    boxes = detector.detect(np.zeros((64, 64, 3), np.uint8))
    assert len(boxes) == 2
    assert all(box.class_name is None and box.class_id is None for box in boxes)
