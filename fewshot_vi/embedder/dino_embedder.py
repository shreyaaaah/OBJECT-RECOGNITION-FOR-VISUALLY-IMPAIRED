"""DINOv2 ViT-S/14 embedder returning L2-normalised CLS vectors and patch tokens.

Pre-processing (square padding with the mean colour, resize, ImageNet
normalisation) is pure NumPy/OpenCV so it is testable without PyTorch. The model
itself is an injectable ``backbone`` callable ``(N,3,S,S) float32 -> (cls, patches)``;
the real backbone is loaded lazily from ``torch.hub`` on first use.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# backbone: float32 array (N,3,S,S) -> (cls (N,D), patches (N,P,D)) as numpy arrays
Backbone = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Pad a BGR crop to a square using its mean colour (centered)."""
    h, w = img.shape[:2]
    if h == w:
        return img
    s = max(h, w)
    mean = img.reshape(-1, img.shape[2]).mean(axis=0) if img.size else np.zeros(3)
    out = np.empty((s, s, img.shape[2]), dtype=img.dtype)
    out[:] = mean.astype(img.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def preprocess(crops: Sequence[np.ndarray], size: int) -> np.ndarray:
    """BGR uint8 crops -> normalised float32 batch (N,3,size,size)."""
    batch = np.empty((len(crops), 3, size, size), dtype=np.float32)
    for i, c in enumerate(crops):
        if c is None or c.size == 0:
            raise ValueError("empty crop passed to embedder")
        sq = cv2.resize(pad_to_square(c), (size, size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch[i] = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    return batch


def l2_normalise(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def select_salient(patches: np.ndarray, k: int) -> np.ndarray:
    """Keep the ``k`` highest-norm patch tokens (a cheap saliency proxy)."""
    if patches.shape[0] <= k:
        return patches
    idx = np.argsort(-np.linalg.norm(patches, axis=1))[:k]
    return patches[np.sort(idx)]


def pick_device() -> str:
    """CUDA > MPS > CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class DinoEmbedder:
    def __init__(self, model: str = "dinov2_vits14", input_size: int = 224, fp16: bool = True,
                 backbone: Optional[Backbone] = None, device: Optional[str] = None) -> None:
        self.model_name, self.input_size, self.fp16 = model, input_size, fp16
        self._backbone = backbone
        self.device = device
        self.dim: Optional[int] = None

    def _load(self) -> Backbone:
        if self._backbone is not None:
            return self._backbone
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for the DINOv2 embedder; see README for install") from exc
        self.device = self.device or pick_device()
        use_fp16 = self.fp16 and self.device == "cuda"  # FP16 only on CUDA
        log.info("loading %s on %s (fp16=%s)", self.model_name, self.device, use_fp16)
        net = torch.hub.load("facebookresearch/dinov2", self.model_name)
        net = net.eval().to(self.device)
        if use_fp16:
            net = net.half()
        dtype = torch.float16 if use_fp16 else torch.float32
        dev = self.device

        def run(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            with torch.inference_mode():
                x = torch.from_numpy(batch).to(dev, dtype=dtype)
                out = net.forward_features(x)
                cls = out["x_norm_clstoken"].float().cpu().numpy()
                patches = out["x_norm_patchtokens"].float().cpu().numpy()
            return cls, patches

        self._backbone = run
        return run

    def warmup(self) -> None:
        self.embed([np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)])

    def embed(self, crops: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Embed all crops in ONE forward pass.

        Returns ``(cls (N,D) L2-normalised, patches (N,P,D) L2-normalised)``.
        """
        if len(crops) == 0:
            return np.zeros((0, self.dim or 384), np.float32), np.zeros((0, 0, self.dim or 384), np.float32)
        backbone = self._load()
        cls, patches = backbone(preprocess(crops, self.input_size))
        cls = l2_normalise(np.asarray(cls, dtype=np.float32))
        patches = l2_normalise(np.asarray(patches, dtype=np.float32))
        self.dim = cls.shape[1]
        return cls, patches

    def close(self) -> None:
        self._backbone = None
