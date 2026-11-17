"""Tests for Few-Shot Metric Learning adapter, loss functions, and adaptation trainer."""
import numpy as np
import pytest
import torch

from fewshot_vi.metric_adapter.adapter import FewShotMetricAdapter
from fewshot_vi.metric_adapter.loss import (
    HardNegativeTripletLoss,
    MetricAdaptationLoss,
    SupervisedContrastiveLoss,
)
from fewshot_vi.metric_adapter.trainer import MetricAdapterTrainer, _compute_cluster_metrics
from fewshot_vi.gallery.store import GalleryStore


def test_metric_adapter_forward_shape_and_norm():
    adapter = FewShotMetricAdapter(in_dim=384, hidden_dim=256, out_dim=128)
    # Batch of 4 vectors
    x = torch.randn(4, 384)
    out = adapter(x)
    assert out.shape == (4, 128)
    # Check L2 normalization
    norms = torch.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    # Single vector (1D)
    x1 = torch.randn(384)
    out1 = adapter(x1)
    assert out1.shape == (128,)
    assert torch.allclose(torch.norm(out1), torch.tensor(1.0), atol=1e-5)


def test_metric_adapter_transform_numpy():
    adapter = FewShotMetricAdapter(in_dim=384, hidden_dim=256, out_dim=256)
    arr = np.random.randn(5, 384).astype(np.float32)
    adapted = adapter.transform_numpy(arr)
    assert adapted.shape == (5, 256)
    norms = np.linalg.norm(adapted, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    # Empty array
    empty = adapter.transform_numpy(np.zeros((0, 384), dtype=np.float32))
    assert empty.shape == (0, 256)


def test_metric_adapter_save_and_load(tmp_path):
    adapter = FewShotMetricAdapter(in_dim=384, hidden_dim=128, out_dim=64)
    ckpt_path = tmp_path / "adapter.pt"
    adapter.save(ckpt_path)

    loaded = FewShotMetricAdapter.load(ckpt_path)
    assert loaded.in_dim == 384
    assert loaded.hidden_dim == 128
    assert loaded.out_dim == 64

    test_input = torch.randn(2, 384)
    adapter.eval()
    loaded.eval()
    with torch.no_grad():
        out1 = adapter(test_input)
        out2 = loaded(test_input)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_supcon_loss_decreases_as_positives_cluster():
    loss_fn = SupervisedContrastiveLoss(temperature=0.1)

    # High cluster similarity: pos pairs have dot product close to 1
    v1 = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    v1 = torch.nn.functional.normalize(v1, dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    loss_clustered = loss_fn(v1, labels)

    # Less clustered features
    v2 = torch.tensor([[1.0, 0.0], [0.5, 0.866], [0.0, 1.0], [0.866, 0.5]])
    v2 = torch.nn.functional.normalize(v2, dim=-1)
    loss_dispersed = loss_fn(v2, labels)

    assert loss_clustered < loss_dispersed


def test_hard_negative_triplet_loss():
    loss_fn = HardNegativeTripletLoss(margin=0.2)
    # Features where negative is already far away
    features_good = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    loss_good = loss_fn(features_good, labels)
    assert loss_good.item() == 0.0

    # Features where negative violates margin
    features_bad = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.95, 0.05], [0.9, 0.1]])
    features_bad = torch.nn.functional.normalize(features_bad, dim=-1)
    loss_bad = loss_fn(features_bad, labels)
    assert loss_bad.item() > 0.0


def test_metric_adapter_training_improves_separation_margin():
    """Verify that training on few-shot samples genuinely expands separation margin."""
    rng = np.random.default_rng(42)
    dim = 384
    num_classes = 3
    samples_per_class = 5

    # Synthesize 3 classes with slight overlap
    centers = rng.normal(0, 1, size=(num_classes, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    E_list = []
    owner_list = []
    for c in range(num_classes):
        # Views with variations around center
        views = centers[c] + rng.normal(0, 0.25, size=(samples_per_class, dim)).astype(np.float32)
        views /= np.linalg.norm(views, axis=1, keepdims=True)
        E_list.append(views)
        owner_list.extend([c] * samples_per_class)

    E = np.vstack(E_list)
    owner = np.array(owner_list, dtype=np.int64)

    adapter = FewShotMetricAdapter(in_dim=dim, hidden_dim=256, out_dim=256)
    trainer = MetricAdapterTrainer(adapter=adapter, epochs=45, lr=2e-3)
    metrics = trainer.train_on_gallery(E, owner)

    # Verifications:
    # 1. Training loss decreased
    assert metrics.final_loss < metrics.initial_loss
    # 2. Intra-class similarity remains high (views of same object stay close)
    assert metrics.intra_class_sim_after >= 0.70
    # 3. Inter-class separation margin improves
    assert metrics.margin_after > metrics.margin_before
    assert metrics.margin_gain > 0.05
    # 4. Training ran fast (< 30 seconds on consumer CPU)
    assert metrics.training_time_s < 30.0


def test_failed_adaptation_keeps_previous_live_adapter(monkeypatch):
    store = GalleryStore(":memory:", dim=384)
    first, second = store.add_object("book"), store.add_object("keys")
    for oid, axis in ((first, 0), (second, 1)):
        for _ in range(2):
            view = np.zeros(384, dtype=np.float32)
            view[axis] = 1.0
            store.add_embedding(oid, view, source="enrol")
    previous = FewShotMetricAdapter(in_dim=384, hidden_dim=64, out_dim=32)
    store.set_adapter(previous)
    before = store.E_adapted.copy()

    def fail_training(*_args, **_kwargs):
        raise RuntimeError("simulated optimizer failure")

    monkeypatch.setattr(MetricAdapterTrainer, "train_on_gallery", fail_training)
    with pytest.raises(RuntimeError, match="simulated optimizer failure"):
        store.adapt_metric(epochs=10)

    assert store.adapter is previous
    assert store.use_adapter is True
    assert np.array_equal(store.E_adapted, before)
    store.close()
