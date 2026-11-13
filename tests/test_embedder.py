import numpy as np
import pytest

from fewshot_vi.embedder import dino_embedder as de
from fewshot_vi.embedder.color_hist import ColorHistEmbedder


def test_pad_to_square_uses_mean_colour_and_centres():
    img = np.full((10, 20, 3), (30, 60, 90), np.uint8)
    sq = de.pad_to_square(img)
    assert sq.shape == (20, 20, 3)
    assert tuple(sq[0, 0]) == (30, 60, 90) and (sq[5:15, :] == img).all()
    img2 = np.zeros((10, 20, 3), np.uint8)
    img2[:, :10] = 200
    sq2 = de.pad_to_square(img2)
    assert tuple(sq2[0, 0]) == (100, 100, 100)                       # mean colour of the crop
    assert de.pad_to_square(np.zeros((8, 8, 3), np.uint8)).shape == (8, 8, 3)


def test_preprocess_shape_and_imagenet_normalisation():
    white = np.full((30, 50, 3), 255, np.uint8)
    b = de.preprocess([white, white], 224)
    assert b.shape == (2, 3, 224, 224) and b.dtype == np.float32
    expected = (1.0 - de.IMAGENET_MEAN) / de.IMAGENET_STD            # RGB order
    assert np.allclose(b[0, :, 100, 112], expected, atol=1e-4)
    with pytest.raises(ValueError):
        de.preprocess([np.zeros((0, 0, 3), np.uint8)], 224)


def fake_backbone_factory(calls):
    def bb(batch):
        calls.append(batch.shape)
        n = batch.shape[0]
        rng = np.random.default_rng(0)
        return rng.normal(size=(n, 384)).astype(np.float32) * 5, rng.normal(size=(n, 256, 384)).astype(np.float32)
    return bb


def test_embed_is_batched_single_forward_and_normalised():
    calls = []
    e = de.DinoEmbedder(backbone=fake_backbone_factory(calls))
    crops = [np.random.default_rng(i).integers(0, 255, (40 + i, 60, 3), dtype=np.uint8) for i in range(5)]
    cls, patches = e.embed(crops)
    assert calls == [(5, 3, 224, 224)]                               # ONE model pass for all crops
    assert cls.shape == (5, 384) and patches.shape == (5, 256, 384)
    assert np.allclose(np.linalg.norm(cls, axis=1), 1, atol=1e-5)
    assert np.allclose(np.linalg.norm(patches, axis=2), 1, atol=1e-5)
    c0, p0 = e.embed([])
    assert c0.shape[0] == 0 and p0.shape[0] == 0 and len(calls) == 1
    e.warmup()
    assert len(calls) == 2


def test_select_salient_keeps_highest_norm_tokens_in_order():
    p = np.eye(6, dtype=np.float32) * np.arange(1, 7)[:, None]
    out = de.select_salient(p, 3)
    assert out.shape == (3, 6) and np.allclose(np.linalg.norm(out, axis=1), [4, 5, 6])
    assert de.select_salient(p, 10).shape == (6, 6)


def test_real_torch_wrapper_path_with_tiny_model(monkeypatch):
    """Exercises the real torch loading code (device choice, fp16 only on CUDA, inference mode) with a
    tiny stand-in network, so no weights are downloaded."""
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def forward_features(self, x):
            assert not torch.is_grad_enabled()                       # inference mode
            b = x.shape[0]
            return {"x_norm_clstoken": torch.ones(b, 384, dtype=x.dtype),
                    "x_norm_patchtokens": torch.ones(b, 256, 384, dtype=x.dtype)}
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: Tiny())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    e = de.DinoEmbedder(fp16=True)
    cls, patches = e.embed([np.zeros((20, 30, 3), np.uint8)])
    assert e.device == "cpu" and cls.shape == (1, 384) and patches.shape == (1, 256, 384)
    assert de.pick_device() == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert de.pick_device() == "cuda"


def test_color_hist_embedder_distinguishes_colours():
    e = ColorHistEmbedder()
    red = np.zeros((40, 40, 3), np.uint8); red[..., 2] = 220
    red2 = red.copy(); red2[..., 2] = 200
    blue = np.zeros((40, 40, 3), np.uint8); blue[..., 0] = 220
    cls, pat = e.embed([red, red2, blue])
    assert cls.shape == (3, 128) and pat.shape == (3, 9, 128)
    assert cls[0] @ cls[1] > cls[0] @ cls[2] + 0.3
    with pytest.raises(ValueError):
        e.embed([np.zeros((0, 0, 3), np.uint8)])
