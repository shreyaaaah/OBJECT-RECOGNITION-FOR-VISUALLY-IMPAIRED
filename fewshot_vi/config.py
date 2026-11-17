"""Configuration loading and validation.

All behavioural thresholds live in ``config.yaml``. ``load_config`` merges the
YAML over built-in defaults, validates ranges/types, and returns a ``Config``
object with attribute access per section (e.g. ``cfg.matching.accept_sim``).
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

DEFAULTS: dict[str, dict[str, Any]] = {
    "camera": {"index": 0, "width": 640, "height": 480, "target_fps": 30},
    "detector": {"backend": "yolo", "weights": "yolov8n.pt", "conf": 0.25,
                 "category_min_conf": 0.45, "detect_every_n_frames": 2, "max_objects": 10, "imgsz": 640},
    "embedder": {"model": "dinov2_vits14", "input_size": 224, "fp16": True},
    "matching": {"accept_sim": 0.70, "ambiguous_sim": 0.55, "min_margin": 0.08, "knn_k": 3},
    "temporal": {"window": 7, "min_votes": 4},
    "gallery": {"max_embeddings_per_object": 20, "update_min_sim": 0.85,
                "update_min_margin": 0.15, "update_cooldown_s": 5},
    "quality": {"blur_var_min": 60.0, "edge_margin_px": 8, "min_box_area_frac": 0.02},
    "guidance": {"hint_cooldown_s": 4, "announce_cooldown_s": 3,
                 "known_object_width_cm_default": 8, "camera_hfov_deg": 65},
    "memory": {"min_conf_to_log": 0.80, "log_cooldown_s": 10, "scene_every_n_s": 5},
    "features": {"self_improve": True, "active_hints": True, "memory": True, "category_fallback": False,
                 "patch_rerank": True, "hand_guidance": False, "depth_model": False},
    "tracker": {"reembed_iou": 0.6, "reembed_every_frames": 30, "match_iou": 0.3,
                "max_missed_frames": 10},
    "rerank": {"deadline_s": 0.03, "top_patches": 32, "min_gain": 0.0,
               "accept_patch_sim": 0.50, "min_patch_margin": 0.02},
    "gallery_policy": {"duplicate_sim": 0.97, "drift_min_sim": 0.60},
    "enrol": {"min_views": 3, "max_views": 5, "timeout_s": 30, "augmentations_per_view": 8,
              "seed": 1234, "stable_frames": 5, "min_box_area_frac": 0.05, "stable_iou": 0.8,
              "no_detection_hint_s": 3.0},
    "guidance_policy": {"find_memory_wait_s": 10, "near_zone_frac": 0.10,
                        "slight_zone_frac": 0.20, "stale_speech_s": 2.0,
                        "beep_min_interval_s": 0.1, "beep_max_interval_s": 1.0,
                        "beep_far_m": 3.0, "beep_near_m": 0.3},
    "voice": {"fuzzy_cutoff": 0.6, "confirm_timeout_s": 8},
    "logging": {"level": "INFO"},
}

# (section, key) -> (min, max) inclusive; None = unbounded.
_RANGES: dict[tuple[str, str], tuple[float | None, float | None]] = {
    ("camera", "width"): (16, None), ("camera", "height"): (16, None),
    ("camera", "target_fps"): (1, 240),
    ("detector", "conf"): (0.0, 1.0), ("detector", "category_min_conf"): (0.0, 1.0),
    ("detector", "detect_every_n_frames"): (1, None),
    ("detector", "max_objects"): (1, None),
    ("detector", "imgsz"): (32, None),
    ("embedder", "input_size"): (14, None),
    ("matching", "accept_sim"): (-1.0, 1.0), ("matching", "ambiguous_sim"): (-1.0, 1.0),
    ("matching", "min_margin"): (0.0, 2.0), ("matching", "knn_k"): (1, None),
    ("temporal", "window"): (1, None), ("temporal", "min_votes"): (1, None),
    ("gallery", "max_embeddings_per_object"): (1, None),
    ("gallery", "update_min_sim"): (-1.0, 1.0), ("gallery", "update_min_margin"): (0.0, 2.0),
    ("gallery", "update_cooldown_s"): (0, None),
    ("quality", "blur_var_min"): (0, None), ("quality", "edge_margin_px"): (0, None),
    ("quality", "min_box_area_frac"): (0.0, 1.0),
    ("guidance", "hint_cooldown_s"): (0, None), ("guidance", "announce_cooldown_s"): (0, None),
    ("guidance", "known_object_width_cm_default"): (0.1, None),
    ("guidance", "camera_hfov_deg"): (1, 179),
    ("memory", "min_conf_to_log"): (0.0, 1.0), ("memory", "log_cooldown_s"): (0, None),
    ("memory", "scene_every_n_s"): (0, None),
    ("tracker", "reembed_iou"): (0.0, 1.0), ("tracker", "reembed_every_frames"): (1, None),
    ("tracker", "match_iou"): (0.0, 1.0), ("tracker", "max_missed_frames"): (0, None),
    ("rerank", "deadline_s"): (0, None), ("rerank", "top_patches"): (1, None), ("rerank", "accept_patch_sim"): (-1.0, 1.0),
    ("rerank", "min_patch_margin"): (0.0, 2.0),
    ("gallery_policy", "duplicate_sim"): (-1.0, 1.0), ("gallery_policy", "drift_min_sim"): (-1.0, 1.0),
    ("enrol", "min_views"): (1, None), ("enrol", "max_views"): (1, None),
    ("enrol", "timeout_s"): (1, None), ("enrol", "augmentations_per_view"): (0, None),
    ("enrol", "no_detection_hint_s"): (0.1, None),
    ("guidance_policy", "near_zone_frac"): (0.0, 1.0),
    ("guidance_policy", "slight_zone_frac"): (0.0, 1.0),
    ("voice", "fuzzy_cutoff"): (0.0, 1.0),
}


class ConfigError(ValueError):
    """Raised when the configuration file is malformed or out of range."""


class Config(SimpleNamespace):
    """Attribute-access configuration; each section is a ``SimpleNamespace``."""

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {k: dict(vars(v)) for k, v in vars(self).items()}


def _validate(raw: dict[str, dict[str, Any]]) -> None:
    for section, keys in raw.items():
        if section not in DEFAULTS:
            raise ConfigError(f"unknown config section: {section!r}")
        if not isinstance(keys, dict):
            raise ConfigError(f"section {section!r} must be a mapping")
        for key, val in keys.items():
            if key not in DEFAULTS[section]:
                raise ConfigError(f"unknown key {section}.{key}")
            default = DEFAULTS[section][key]
            if isinstance(default, bool):
                if not isinstance(val, bool):
                    raise ConfigError(f"{section}.{key} must be a boolean, got {val!r}")
            elif isinstance(default, (int, float)):
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ConfigError(f"{section}.{key} must be a number, got {val!r}")
            elif isinstance(default, str) and not isinstance(val, str):
                raise ConfigError(f"{section}.{key} must be a string, got {val!r}")
            rng = _RANGES.get((section, key))
            if rng:
                lo, hi = rng
                if (lo is not None and val < lo) or (hi is not None and val > hi):
                    raise ConfigError(f"{section}.{key}={val} outside [{lo}, {hi}]")
    m, t, e = raw["matching"], raw["temporal"], raw["enrol"]
    if m["ambiguous_sim"] > m["accept_sim"]:
        raise ConfigError("matching.ambiguous_sim must be <= matching.accept_sim")
    if t["min_votes"] > t["window"]:
        raise ConfigError("temporal.min_votes must be <= temporal.window")
    if e["min_views"] > e["max_views"]:
        raise ConfigError("enrol.min_views must be <= enrol.max_views")
    if raw["gallery"]["update_min_sim"] < m["accept_sim"]:
        raise ConfigError("gallery.update_min_sim must be >= matching.accept_sim")
    if raw["guidance_policy"]["near_zone_frac"] > raw["guidance_policy"]["slight_zone_frac"]:
        raise ConfigError("guidance_policy.near_zone_frac must be <= slight_zone_frac")
    if raw["features"]["depth_model"]:
        raise ConfigError("features.depth_model is not implemented in this version; "
                          "distance uses the box-size heuristic. Set it to false.")
    if raw["detector"]["backend"] not in ("yolo", "fake"):
        raise ConfigError("detector.backend must be 'yolo' (or 'fake' for tests)")


def config_from_dict(overrides: dict[str, dict[str, Any]] | None = None) -> Config:
    """Merge ``overrides`` over defaults, validate, and return a ``Config``."""
    merged = copy.deepcopy(DEFAULTS)
    for section, keys in (overrides or {}).items():
        if section not in DEFAULTS:
            raise ConfigError(f"unknown config section: {section!r}")
        if not isinstance(keys, dict):
            raise ConfigError(f"section {section!r} must be a mapping")
        merged[section].update(keys)
    _validate(merged)
    return Config(**{k: SimpleNamespace(**v) for k, v in merged.items()})


def load_config(path: str | Path | None = None) -> Config:
    """Load ``config.yaml`` (or defaults if ``path`` is None) and validate it."""
    if path is None:
        return config_from_dict()
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    return config_from_dict(data)
