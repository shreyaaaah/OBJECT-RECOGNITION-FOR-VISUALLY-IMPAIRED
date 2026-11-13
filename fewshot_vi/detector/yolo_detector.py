"""Ultralytics YOLO used as a class-agnostic box proposer."""
from __future__ import annotations

import logging

import numpy as np

from ..types import Detection
from .base import Detector

log = logging.getLogger(__name__)


class YoloDetector(Detector):
    """Loads Ultralytics YOLO lazily. Class labels are ignored on purpose:
    personal objects (keys, pill boxes) are generally not COCO classes, so every
    box above ``conf`` is kept (class-agnostic NMS), capped at ``max_objects``.
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

    def detect(self, frame: np.ndarray) -> list[Detection]:
        model = self._load()
        res = model.predict(frame, imgsz=self.imgsz, conf=self.conf, agnostic_nms=True, max_det=self.max_objects,
                            device=self.device, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        order = np.argsort(-conf)[: self.max_objects]
        return [Detection(*map(float, xyxy[i]), float(conf[i])) for i in order]

    def close(self) -> None:
        self._model = None
