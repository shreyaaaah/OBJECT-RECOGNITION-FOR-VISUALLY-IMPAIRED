"""API route tests with injected models; no sockets, weights, or camera hardware."""
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from fewshot_vi.api.server import AppContext, create_app
from fewshot_vi.api.schemas import RenameObjectRequest
from fewshot_vi.gallery.matcher import Matcher
from fewshot_vi.memory.object_memory import ObjectMemory
from fewshot_vi.metric_adapter.adapter import FewShotMetricAdapter
from fewshot_vi.types import Detection


class StaticDetector:
    def detect(self, frame):
        return [Detection(180, 120, 460, 400, 0.95, "book", 1)]


class StaticEmbedder:
    def embed(self, crops):
        features = np.zeros((len(crops), 384), dtype=np.float32)
        features[:, 0] = 1.0
        patches = np.repeat(features[:, None, :], 4, axis=1)
        return features, patches


def _endpoint(app, name):
    return next(route.endpoint for route in app.routes if getattr(route, "name", None) == name)


def _finish_now(awaitable):
    """Run coroutine endpoints that only await in-memory UploadFile reads."""
    try:
        awaitable.send(None)
    except StopIteration as completed:
        return completed.value
    raise RuntimeError("Route unexpectedly waited for external async I/O")


class MemoryUpload:
    def __init__(self, data: bytes):
        self.data = data

    async def read(self):
        return self.data


@pytest.fixture
def api(tmp_path):
    ctx = AppContext(db_path=str(tmp_path / "gallery.db"))
    ctx.adapter_checkpoint_path = Path(tmp_path) / "adapter.pt"
    oid1 = ctx.store.add_object("my mathematics book")
    oid2 = ctx.store.add_object("my blue water bottle")
    rng = np.random.default_rng(101)
    for oid in (oid1, oid2):
        center = np.zeros(384, dtype=np.float32)
        center[0 if oid == oid1 else 1] = 1.0
        for _ in range(5):
            view = center + rng.normal(0, 0.005, size=384).astype(np.float32)
            view /= np.linalg.norm(view)
            ctx.store.add_embedding(oid, view, source="enrol")

    ctx.metric_adapter = FewShotMetricAdapter(in_dim=384, hidden_dim=128, out_dim=64)
    ctx.store.set_adapter(ctx.metric_adapter)
    ctx.detector = StaticDetector()
    ctx.embedder = StaticEmbedder()
    ctx.matcher = Matcher(ctx.store, accept_sim=0.70, ambiguous_sim=0.55, min_margin=0.08)
    ctx.memory = ObjectMemory(ctx.store, min_conf=0.8, cooldown_s=10.0)
    app = create_app(app_context=ctx, load_models=False)
    yield app
    ctx.memory.close()
    ctx.store.close()


def test_api_health(api):
    data = _endpoint(api, "health")()
    assert data["status"] == "healthy"
    assert data["gallery"]["objects_count"] == 2
    assert data["gallery"]["total_embeddings"] == 10


def test_api_list_objects(api):
    objects = jsonable_encoder(_endpoint(api, "list_objects")())
    assert len(objects) == 2
    assert {obj["name"] for obj in objects} == {"my mathematics book", "my blue water bottle"}


def test_api_rename_object(api):
    objects = _endpoint(api, "list_objects")()
    result = _endpoint(api, "rename_object")(objects[0].id, RenameObjectRequest(name="advanced calculus book"))
    assert result["name"] == "advanced calculus book"
    assert "advanced calculus book" in {name for _, name in api.state.context.store.list_objects()}


def test_api_delete_and_clear_objects(api):
    object_id, name = api.state.context.store.list_objects()[0]
    assert _endpoint(api, "delete_object")(object_id)["deleted_object"] == name
    assert len(api.state.context.store.list_objects()) == 1
    assert "wiped" in _endpoint(api, "clear_all_objects")()["message"]
    assert api.state.context.store.list_objects() == []
    assert api.state.context.store.adapter is None
    assert api.state.context.metric_adapter is None


def test_api_adapter_status_and_train(api):
    status = jsonable_encoder(_endpoint(api, "adapter_status")())
    assert status["num_classes"] == 2
    assert status["total_embeddings"] == 10
    trained = _endpoint(api, "trigger_adaptation")(epochs=15)
    assert trained["success"] is True
    assert trained["num_classes"] == 2


def test_api_recognize_mock_frame(api):
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (440, 350), (255, 0, 0), -1)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    upload = MemoryUpload(buf.tobytes())
    response = _finish_now(_endpoint(api, "recognize")(frame=upload, mode="IDENTIFY", target=None))
    data = jsonable_encoder(response)
    assert "detections" in data
    assert "spoken_announcements" in data
    assert data["fps"] > 0


def test_recognition_requires_loaded_models(tmp_path):
    ctx = AppContext(db_path=str(tmp_path / "empty.db"))
    app = create_app(ctx, load_models=False)
    ok, encoded = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert ok
    frame = MemoryUpload(encoded.tobytes())
    with pytest.raises(HTTPException) as exc:
        _finish_now(_endpoint(app, "recognize")(frame=frame, mode="IDENTIFY", target=None))
    assert exc.value.status_code == 503
    ctx.store.close()
