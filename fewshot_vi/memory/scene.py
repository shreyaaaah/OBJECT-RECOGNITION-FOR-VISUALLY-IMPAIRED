"""CLIP zero-shot scene classifier behind an interface.

``NullSceneClassifier`` is the graceful-disable path: it reports ``available =
False`` and returns ``None`` -- it NEVER invents a label. ``ClipSceneClassifier``
loads ``open_clip`` lazily; if the dependency or weights are missing it logs one
clear warning and disables itself (same ``None`` behaviour).
"""
from __future__ import annotations

import abc
import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_LABELS = ["kitchen", "desk", "bed", "dining table", "sofa", "shelf", "bathroom", "floor", "outdoors"]


class SceneClassifier(abc.ABC):
    available: bool = False

    @abc.abstractmethod
    def classify(self, frame: np.ndarray) -> Optional[tuple[str, np.ndarray]]:
        """Return ``(label, scene_embedding)`` or ``None`` when no label can be given."""


class NullSceneClassifier(SceneClassifier):
    available = False

    def __init__(self, reason: str = "scene classification disabled") -> None:
        self.reason = reason

    def classify(self, frame: np.ndarray) -> Optional[tuple[str, np.ndarray]]:
        return None


class FixedSceneClassifier(SceneClassifier):
    """Deterministic classifier for tests/dev (explicitly labelled as fake by its name)."""
    available = True

    def __init__(self, label: str, dim: int = 4) -> None:
        self.label, self._emb = label, np.ones(dim, np.float32) / np.sqrt(dim)

    def classify(self, frame: np.ndarray) -> Optional[tuple[str, np.ndarray]]:
        return self.label, self._emb


class ClipSceneClassifier(SceneClassifier):
    def __init__(self, labels: Optional[list[str]] = None, model: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k", device: Optional[str] = None) -> None:
        self.labels = labels or DEFAULT_LABELS
        self.model_name, self.pretrained, self.device = model, pretrained, device
        self.available = False
        self.reason: Optional[str] = None
        self._loaded = False
        self._model = self._preprocess = self._text = self._torch = None

    def _load(self) -> bool:
        if self._loaded:
            return self.available
        self._loaded = True
        try:
            import open_clip
            import torch
            from PIL import Image  # noqa: F401  (needed by the preprocess transform)
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
            tok = open_clip.get_tokenizer(self.model_name)
            model = model.eval().to(dev)
            with torch.inference_mode():
                text = model.encode_text(tok([f"a photo of a {l}" for l in self.labels]).to(dev))
                text = text / text.norm(dim=-1, keepdim=True)
            self._model, self._preprocess, self._text, self._torch, self.device = model, preprocess, text, torch, dev
            self.available = True
        except Exception as exc:  # noqa: BLE001 - missing dep, no network, bad weights...
            self.reason = f"{type(exc).__name__}: {exc}"
            log.warning("scene classifier disabled (%s); no scene labels will be produced", self.reason)
        return self.available

    def classify(self, frame: np.ndarray) -> Optional[tuple[str, np.ndarray]]:
        if not self._load():
            return None
        import cv2
        from PIL import Image
        torch = self._torch
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            x = self._preprocess(img).unsqueeze(0).to(self.device)
            f = self._model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = (f @ self._text.T)[0]
            idx = int(logits.argmax().item())
        return self.labels[idx], f[0].float().cpu().numpy()
