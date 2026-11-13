"""Real dataset/model ablation evaluation.

Every number comes from running the production ``GalleryStore``, ``Matcher``,
``apply_rerank`` and ``GalleryUpdater`` on embeddings of REAL images from
``dataset/<class>/*.jpg``. Nothing is simulated except where a row's description
says so explicitly (degraded views are produced by blurring / down-sampling real
images; the "active re-view" recovery re-captures the clean version of the same
image; the self-update stream feeds queries one at a time as stable matches).

Ablations (``--ablation``):
  1. baseline          global embedding + open-set rules; AMBIGUOUS counts as "not sure" (rejection)
  2. + cascade         patch re-rank ONLY for ambiguous matches between marked look-alike classes
                       (the most similar class pairs in the episode are auto-marked as confusers),
                       plus an "always re-rank" latency reference
  3. look-alike subset queries of the marked pairs (+ the unknown queries): global-only vs + patch re-rank
  4. degraded views    blur / low-resolution queries: errors, hint coverage and simulated recovery
  5. sequential update sessions of queries, self-update OFF vs ON (accuracy per session, auto-adds,
                       wrong-class auto-adds)

Missing data fails with a ``DatasetError`` that says what to add -- no placeholder results.
"""
from __future__ import annotations

import csv
import itertools
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import cv2
import numpy as np

from ..config import Config
from ..embedder.dino_embedder import select_salient
from ..gallery.matcher import Matcher
from ..gallery.patch_matcher import RerankResult, apply_rerank, rerank
from ..gallery.store import GalleryStore
from ..gallery.updater import GalleryUpdater
from ..quality.quality import laplacian_blur
from ..types import Decision, Match, Quality
from .episodes import DatasetError, Episode, check_capacity, load_image, sample_episodes, scan_dataset
from .metrics import OpenSetCounts

log = logging.getLogger(__name__)

DEGRADATIONS = ("blur", "lowres")


# --------------------------------------------------------------------- features
@dataclass
class Feat:
    cls: np.ndarray
    patches: np.ndarray
    blur: float


def degrade(img: np.ndarray, kind: str) -> np.ndarray:
    """Deterministic degradations of a real image."""
    h, w = img.shape[:2]
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), sigmaX=max(3.0, 0.03 * max(h, w)))
    if kind == "lowres":
        small = cv2.resize(img, (max(2, w // 8), max(2, h // 8)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"unknown degradation {kind!r}")


class FeatureCache:
    """Embeds every (image, degradation) once, in batches."""

    def __init__(self, embedder, top_patches: int, batch: int = 16) -> None:
        self.embedder, self.top_patches, self.batch = embedder, top_patches, batch
        self._c: dict[tuple[Path, Optional[str]], Feat] = {}

    def prefetch(self, items: Iterable[tuple[Path, Optional[str]]]) -> None:
        todo = [it for it in dict.fromkeys(items) if it not in self._c]
        for i in range(0, len(todo), self.batch):
            chunk = todo[i:i + self.batch]
            imgs = []
            for p, deg in chunk:
                img = load_image(p)
                imgs.append(degrade(img, deg) if deg else img)
            cls, pat = self.embedder.embed(imgs)
            for key, img, c, pp in zip(chunk, imgs, cls, pat):
                self._c[key] = Feat(np.asarray(c, np.float32), select_salient(np.asarray(pp, np.float32),
                                                                              self.top_patches),
                                    laplacian_blur(img))

    def get(self, path: Path, deg: Optional[str] = None) -> Feat:
        return self._c[(path, deg)]


# ------------------------------------------------------------------ prediction
@dataclass
class Pred:
    name: Optional[str]       # accepted class name, or None for "not sure"/unknown
    final: Match
    initial: Match
    slow: bool
    seconds: float


def _patch_rerank(cfg: Config, store: GalleryStore, m: Match, feat: Feat, cands: Sequence[int]) -> Match:
    t0 = time.perf_counter()
    ranking = rerank(feat.patches, {oid: store.patches_of(oid) for oid in cands})
    res = RerankResult(0, ranking, time.perf_counter() - t0, True)
    return apply_rerank(m, res, store.get_name, cfg.rerank.accept_patch_sim, cfg.rerank.min_gain,
                        cfg.rerank.min_patch_margin)


def predict(cfg: Config, store: GalleryStore, matcher: Matcher, feat: Feat, mode: str,
            force_rerank: bool = False) -> Pred:
    """``mode``: 'baseline' (global only) or 'cascade' (patch re-rank only when needed)."""
    t0 = time.perf_counter()
    m = matcher.match(feat.cls)
    final, slow = m, False
    if mode == "cascade" and cfg.features.patch_rerank and m.decision == Decision.AMBIGUOUS:
        cands = matcher.confuser_candidates(m)
        if cands:
            final, slow = _patch_rerank(cfg, store, m, feat, cands), True
    if force_rerank and not slow and len(m.ranking) >= 2:  # latency reference: always run the slow path
        _patch_rerank(cfg, store, m, feat, [o for o, _ in m.ranking[:2]])
        slow = True
    name = final.name if final.decision == Decision.ACCEPT else None
    return Pred(name, final, m, slow, time.perf_counter() - t0)


def _correct(true: Optional[str], pred: Optional[str]) -> bool:
    return pred is None if true is None else pred == true


def _unsafe(true: Optional[str], pred: Optional[str]) -> bool:
    """An accepted WRONG name, or any name for an unknown object."""
    return pred is not None and pred != true


# ----------------------------------------------------------------- aggregation
@dataclass
class Agg:
    counts: OpenSetCounts = field(default_factory=OpenSetCounts)
    n: int = 0
    slow: int = 0
    seconds: float = 0.0
    closed_total: int = 0
    closed_correct: int = 0

    def add(self, true: Optional[str], p: Pred) -> None:
        self.counts.add(true, p.name)
        self.n += 1
        self.slow += p.slow
        self.seconds += p.seconds
        if true is not None:
            self.closed_total += 1
            self.closed_correct += (p.initial.name == true)

    def merge(self, o: "Agg") -> None:
        for f in ("known_total", "known_correct", "known_wrong", "known_rejected", "unknown_total",
                  "unknown_accepted"):
            setattr(self.counts, f, getattr(self.counts, f) + getattr(o.counts, f))
        self.n += o.n
        self.slow += o.slow
        self.seconds += o.seconds
        self.closed_total += o.closed_total
        self.closed_correct += o.closed_correct

    def metrics(self) -> dict[str, float]:
        c = self.counts
        return {
            "top1_acc": c.top1_accuracy, "closed_top1": self.closed_correct / self.closed_total
            if self.closed_total else float("nan"),
            "FAR": c.far, "FRR": c.frr, "misid": c.misid_rate,
            "known_n": c.known_total, "unknown_n": c.unknown_total,
            "slow_path_frac": self.slow / self.n if self.n else float("nan"),
            "decision_ms": 1000.0 * self.seconds / self.n if self.n else float("nan"),
        }


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


# ------------------------------------------------------------------- episodes
@dataclass
class Env:
    ep: Episode
    plain: GalleryStore
    conf: GalleryStore
    pairs: list[tuple[str, str]]
    lookalike: set[str]
    queries: list[tuple[Optional[str], Path]]


def episode_paths(ep: Episode) -> list[Path]:
    out = [p for v in ep.support.values() for p in v]
    out += [p for v in ep.query_known.values() for p in v]
    out += [p for v in ep.query_unknown.values() for p in v]
    return out


def _enrol_support(cfg: Config, ep: Episode, cache: FeatureCache, dim: int) -> GalleryStore:
    store = GalleryStore(":memory:", dim=dim, top_patches=cfg.rerank.top_patches)
    for c in ep.known:
        oid = store.add_object(c)
        for p in ep.support[c]:
            f = cache.get(p)
            store.add_embedding(oid, f.cls, "enrol", patches=f.patches)
    return store


def most_similar_pairs(store: GalleryStore, names: Sequence[str], m: int) -> list[tuple[str, str]]:
    """The ``m`` most similar class pairs by prototype cosine (greedy, each class used once)."""
    protos = {n: store.prototype(store.get_id(n)) for n in names}
    cand = sorted(((float(protos[a] @ protos[b]), a, b) for a, b in itertools.combinations(names, 2)),
                  reverse=True)
    used: set[str] = set()
    out = []
    for _, a, b in cand:
        if a in used or b in used:
            continue
        out.append((a, b))
        used.update((a, b))
        if len(out) >= m:
            break
    return out


def build_env(cfg: Config, ep: Episode, cache: FeatureCache, n_pairs: int) -> Env:
    dim = cache.get(episode_paths(ep)[0]).cls.shape[0]
    plain = _enrol_support(cfg, ep, cache, dim)
    conf = _enrol_support(cfg, ep, cache, dim)
    pairs = most_similar_pairs(plain, ep.known, n_pairs)
    for a, b in pairs:
        conf.set_confusers(a, b)
    queries: list[tuple[Optional[str], Path]] = [(c, p) for c in ep.known for p in ep.query_known[c]]
    queries += [(None, p) for c in ep.unknown for p in ep.query_unknown[c]]
    return Env(ep, plain, conf, pairs, {x for pr in pairs for x in pr}, queries)


def run_queries(cfg: Config, store: GalleryStore, cache: FeatureCache, queries: Sequence[tuple[Optional[str], Path]],
                mode: str, deg: Optional[str] = None, force_rerank: bool = False) -> Agg:
    matcher = Matcher(store, cfg.matching.accept_sim, cfg.matching.ambiguous_sim, cfg.matching.min_margin,
                      cfg.matching.knn_k)
    agg = Agg()
    for true, path in queries:
        agg.add(true, predict(cfg, store, matcher, cache.get(path, deg), mode, force_rerank))
    return agg


# --------------------------------------------------------------- ablation rows
Rows = list[dict]


def _emit(rows: Rows, ablation: str, config: str, k: int, metrics: dict[str, float], meta: dict) -> None:
    for name, val in metrics.items():
        rows.append({"ablation": ablation, "config": config, "k_shot": k, "metric": name,
                     "value": float(val), **meta})


def ablation_baseline_cascade(cfg, envs: list[Env], cache, k, rows, meta) -> None:
    arms = [
        ("1 baseline", "1 baseline (global embedding only)", lambda e: (e.plain, "baseline", False)),
        ("2 cascade", "no re-rank, look-alikes marked", lambda e: (e.conf, "baseline", False)),
        ("2 cascade", "+ cascade (patch re-rank only if ambiguous)", lambda e: (e.conf, "cascade", False)),
        ("2 cascade", "reference: always patch re-rank (no cascade)", lambda e: (e.conf, "cascade", True)),
    ]
    for abl, label, sel in arms:
        total, per_ep = Agg(), []
        for e in envs:
            store, mode, force = sel(e)
            a = run_queries(cfg, store, cache, e.queries, mode, None, force)
            per_ep.append(a.metrics()["top1_acc"])
            total.merge(a)
        m = total.metrics()
        m["top1_std_over_episodes"] = float(np.nanstd(per_ep)) if per_ep else float("nan")
        _emit(rows, abl, label, k, m, meta)


def ablation_lookalike(cfg, envs, cache, k, rows, meta) -> None:
    arms = [
        ("global only, no look-alikes marked", lambda e: (e.plain, "baseline")),
        ("global only, look-alikes marked", lambda e: (e.conf, "baseline")),
        ("+ patch re-rank (cascade)", lambda e: (e.conf, "cascade")),
    ]
    for label, sel in arms:
        total = Agg()
        for e in envs:
            sub = [(t, p) for t, p in e.queries if t is None or t in e.lookalike]
            store, mode = sel(e)
            total.merge(run_queries(cfg, store, cache, sub, mode))
        _emit(rows, "3 look-alike subset", label, k, total.metrics(), meta)


def ablation_degraded(cfg, envs, cache, k, rows, meta) -> None:
    thr = cfg.quality.blur_var_min
    for deg in DEGRADATIONS:
        n = fail = flagged_fail = recovered = 0
        ok_clean = ok_deg = unsafe_clean = unsafe_deg = flag_clean = flag_deg = 0
        for e in envs:
            store = e.conf
            matcher = Matcher(store, cfg.matching.accept_sim, cfg.matching.ambiguous_sim,
                              cfg.matching.min_margin, cfg.matching.knn_k)
            for true, path in e.queries:
                fc, fd = cache.get(path), cache.get(path, deg)
                pc = predict(cfg, store, matcher, fc, "cascade").name
                pd = predict(cfg, store, matcher, fd, "cascade").name
                n += 1
                ok_clean += _correct(true, pc)
                ok_deg += _correct(true, pd)
                unsafe_clean += _unsafe(true, pc)
                unsafe_deg += _unsafe(true, pd)
                flag_clean += fc.blur < thr
                flag_deg += fd.blur < thr
                if not _correct(true, pd):
                    fail += 1
                    if fd.blur < thr:               # the active hint ("Hold the camera steady") fires
                        flagged_fail += 1
                        recovered += _correct(true, pc)   # simulated re-view = the clean image
        m = {
            "queries": n,
            "correct_rate_clean": _div(ok_clean, n), "correct_rate_degraded": _div(ok_deg, n),
            "unsafe_rate_clean": _div(unsafe_clean, n), "unsafe_rate_degraded": _div(unsafe_deg, n),
            "hint_flag_rate_clean": _div(flag_clean, n), "hint_flag_rate_degraded": _div(flag_deg, n),
            "failures_degraded": fail,
            "hint_coverage_of_failures": _div(flagged_fail, fail),
            "recovery_rate_after_review": _div(recovered, flagged_fail),
            "correct_rate_after_reviews": _div(ok_deg + recovered, n),
        }
        _emit(rows, "4 degraded views (simulated re-view)", f"degradation={deg}", k, m, meta)


def _eval_quality(cfg: Config, blur: float, margin: float) -> Quality:
    blurry = blur < cfg.quality.blur_var_min
    low = math.isfinite(margin) and margin < cfg.matching.min_margin
    return Quality(blur, blurry, frozenset(), False, low, 0.0 if (blurry or low) else 1.0)


def ablation_sequential(cfg, envs, cache, k, rows, meta, sessions: int, seed: int) -> None:
    n_s = max(1, sessions)
    per_arm: dict[str, list[Agg]] = {"self-update OFF": [Agg() for _ in range(n_s)],
                                     "self-update ON": [Agg() for _ in range(n_s)]}
    added = wrong_added = 0
    for e in envs:
        rng = random.Random(f"{seed}:{k}:{e.ep.index}:seq")
        buckets: list[list[tuple[Optional[str], Path]]] = [[] for _ in range(n_s)]
        by_class: dict[Optional[str], list[tuple[Optional[str], Path]]] = {}
        for t, p in e.queries:
            by_class.setdefault(t if t is not None else f"__unk__{p.parent.name}", []).append((t, p))
        for items in by_class.values():
            items = list(items)
            rng.shuffle(items)
            for i, it in enumerate(items):
                buckets[i % n_s].append(it)
        for b in buckets:
            rng.shuffle(b)
        for arm in per_arm:
            store = _enrol_support(cfg, e.ep, cache, cache.get(episode_paths(e.ep)[0]).cls.shape[0])
            matcher = Matcher(store, cfg.matching.accept_sim, cfg.matching.ambiguous_sim,
                              cfg.matching.min_margin, cfg.matching.knn_k)
            up = None
            if arm.endswith("ON") and cfg.features.self_improve:
                up = GalleryUpdater(store, cfg.temporal.window, cfg.gallery.max_embeddings_per_object,
                                    cfg.gallery.update_min_sim, cfg.gallery.update_min_margin, 0.0,
                                    cfg.gallery_policy.duplicate_sim, cfg.gallery_policy.drift_min_sim,
                                    clock=lambda: 0.0)   # cooldown is meaningless for offline images
            for s, bucket in enumerate(buckets):
                for true, path in bucket:
                    feat = cache.get(path)
                    p = predict(cfg, store, matcher, feat, "cascade")
                    per_arm[arm][s].add(true, p)
                    if up is not None and p.final.decision == Decision.ACCEPT and p.final.object_id is not None:
                        res = up.maybe_add(p.final.object_id, feat.cls, p.final,
                                           _eval_quality(cfg, feat.blur, p.final.margin),
                                           stable_accepts=cfg.temporal.window, confuser_agrees=True)
                        if res.added:
                            added += 1
                            wrong_added += _unsafe(true, p.name)
            store.close()
    for arm, aggs in per_arm.items():
        for s, a in enumerate(aggs):
            m = a.metrics()
            _emit(rows, "5 sequential self-update", f"{arm} / session {s + 1}", k,
                  {x: m[x] for x in ("top1_acc", "FAR", "FRR", "misid", "known_n", "unknown_n")}, meta)
    _emit(rows, "5 sequential self-update", "self-update ON / totals", k,
          {"auto_embeddings_added": added, "wrong_class_auto_adds": wrong_added,
           "wrong_add_rate": _div(wrong_added, added)}, meta)


# ------------------------------------------------------------------- reporting
_COUNT_METRICS = {"known_n", "unknown_n", "queries", "failures_degraded", "auto_embeddings_added",
                  "wrong_class_auto_adds"}


def _fmt(metric: str, v: float) -> str:
    if v != v:
        return "undef"
    if metric in _COUNT_METRICS:
        return str(int(v))
    return f"{v:.2f} ms" if metric.endswith("_ms") else f"{v:.3f}"


def format_tables(rows: Rows, header: Sequence[str]) -> str:
    out = list(header)
    ablations = list(dict.fromkeys(r["ablation"] for r in rows))
    for abl in ablations:
        sub = [r for r in rows if r["ablation"] == abl]
        metrics = list(dict.fromkeys(r["metric"] for r in sub))
        keys = list(dict.fromkeys((r["config"], r["k_shot"]) for r in sub))
        table = [["config", "K"] + metrics]
        for cfgname, k in keys:
            vals = {r["metric"]: r["value"] for r in sub if r["config"] == cfgname and r["k_shot"] == k}
            table.append([cfgname, str(k)] + [_fmt(m, vals[m]) if m in vals else "" for m in metrics])
        widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
        out.append(f"\n== ablation {abl} ==")
        for i, row in enumerate(table):
            out.append("  ".join(c.ljust(w) for c, w in zip(row, widths)))
            if i == 0:
                out.append("  ".join("-" * w for w in widths))
    out.append("\n'undef' = the denominator was zero for that metric (e.g. no failures to recover from).")
    return "\n".join(out)


def write_csv(rows: Rows, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = ["ablation", "config", "k_shot", "metric", "value", "embedder", "dataset", "seed", "episodes",
            "n_way", "n_query", "n_unknown"]
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def run_ablation(cfg: Config, embedder, embedder_label: str, dataset_root: str | Path, *,
                 ks: Sequence[int] = (1, 3, 5), n_way: int = 5, n_query: int = 5, n_unknown: int = 3,
                 episodes: int = 10, seed: int = 0, sessions: int = 3, max_per_class: Optional[int] = None,
                 lookalike_pairs: Optional[int] = None, full: bool = True,
                 out_csv: Optional[str | Path] = None,
                 progress: Callable[[str], None] = lambda s: None) -> tuple[Rows, str]:
    """Run the evaluation. Returns ``(rows, printable_report)``. Raises ``DatasetError`` on bad data."""
    index = scan_dataset(dataset_root, max_per_class)
    for k in ks:
        check_capacity(index, n_way, k, n_query, n_unknown)
    if embedder_label.startswith("vit-s14-random-init"):
        raise DatasetError("a random-initialised model produces meaningless accuracy; it is for latency "
                           "benchmarking only. Use --embedder dino for evaluation.")
    cache = FeatureCache(embedder, cfg.rerank.top_patches)
    n_pairs = lookalike_pairs if lookalike_pairs is not None else max(1, n_way // 3)
    rows: Rows = []
    t_start = time.perf_counter()
    for k in ks:
        eps = sample_episodes(index, n_way, k, n_query, n_unknown, episodes, seed)
        paths = list(dict.fromkeys(p for e in eps for p in episode_paths(e)))
        progress(f"K={k}: embedding {len(paths)} images ({len(eps)} episodes) ...")
        cache.prefetch((p, None) for p in paths)
        if full:
            qpaths = [p for e in eps for v in (*e.query_known.values(), *e.query_unknown.values()) for p in v]
            for deg in DEGRADATIONS:
                cache.prefetch((p, deg) for p in dict.fromkeys(qpaths))
        envs = [build_env(cfg, e, cache, n_pairs) for e in eps]
        meta = {"embedder": embedder_label, "dataset": str(dataset_root), "seed": seed, "episodes": episodes,
                "n_way": n_way, "n_query": n_query, "n_unknown": n_unknown}
        progress(f"K={k}: evaluating ...")
        ablation_baseline_cascade(cfg, envs, cache, k, rows, meta)
        if full:
            ablation_lookalike(cfg, envs, cache, k, rows, meta)
            ablation_degraded(cfg, envs, cache, k, rows, meta)
            ablation_sequential(cfg, envs, cache, k, rows, meta, sessions, seed)
        for e in envs:
            e.plain.close()
            e.conf.close()
    header = [
        "Few-shot personal object recognition: " + ("ablation" if full else "baseline") + " evaluation",
        f"embedder   : {embedder_label}",
        f"dataset    : {dataset_root}  ({len(index)} classes, {sum(len(v) for v in index.values())} images)",
        f"episodes   : {episodes} per K, n_way={n_way}, n_query={n_query}, n_unknown={n_unknown}, "
        f"K in {list(ks)}, seed={seed}",
        f"thresholds : accept_sim={cfg.matching.accept_sim} ambiguous_sim={cfg.matching.ambiguous_sim} "
        f"min_margin={cfg.matching.min_margin} accept_patch_sim={cfg.rerank.accept_patch_sim}",
        f"wall time  : {time.perf_counter() - t_start:.1f} s (includes embedding)",
        "top1_acc = known queries accepted as the right class; closed_top1 = argmax class ignoring thresholds;",
        "FAR = unknown accepted as a known class; FRR = known reported 'not sure'; misid = known accepted wrongly.",
    ]
    report = format_tables(rows, header)
    if out_csv:
        write_csv(rows, out_csv)
        report += f"\n\nCSV written to {out_csv}"
    return rows, report
