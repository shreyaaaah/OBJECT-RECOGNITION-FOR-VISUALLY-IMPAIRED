"""Ultralytics YOLO detector returning boxes and pretrained category labels."""
from __future__ import annotations

import logging

import numpy as np

from ..types import Detection
from .base import Detector

log = logging.getLogger(__name__)


class YoloDetector(Detector):
    """Loads Ultralytics YOLO lazily and preserves its category predictions.

    The default ``yolov8n.pt`` weights are trained on COCO categories. Personal
    gallery matching remains available for items outside those categories.
    """

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25,
                 max_objects: int = 10, device: str | None = None, imgsz: int = 640) -> None:
        self.weights, self.conf, self.max_objects, self.device = weights, conf, max_objects, device
        self.imgsz = imgsz
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("ultralytics is not installed; run `pip install ultralytics` "
                                   "(first run also downloads the weights file)") from exc
            log.info("loading YOLO weights %s", self.weights)
            self._model = YOLO(self.weights)
        return self._model

    def warmup(self, shape: tuple[int, int] = (480, 640)) -> None:
        self.detect(np.zeros((*shape, 3), dtype=np.uint8))

    @property
    def class_names(self) -> list[str]:
        if self._model is None:
            return []
        names = self._model.names
        return [str(names[i]) for i in range(len(names))] if not isinstance(names, dict) \
            else [str(names[i]) for i in sorted(names)]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        model = self._load()
        res = model.predict(frame, imgsz=self.imgsz, conf=self.conf, agnostic_nms=True, max_det=self.max_objects,
                            device=self.device, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        raw_class_ids = getattr(boxes, "cls", None)
        class_ids = None if raw_class_ids is None else raw_class_ids.cpu().numpy().astype(int)
        order = np.argsort(-conf)[: self.max_objects]
        names = getattr(model, "names", {})
        detections = []
        for i in order:
            if class_ids is None:
                detections.append(Detection(*map(float, xyxy[i]), float(conf[i])))
                continue
            class_id = int(class_ids[i])
            name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            detections.append(Detection(*map(float, xyxy[i]), float(conf[i]), str(name), class_id))
        return detections

    def close(self) -> None:
        self._model = None
