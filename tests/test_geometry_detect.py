"""YOLO adapter: classes by name, OBB corners -> Parts, predict proxy."""
import numpy as np
import pytest

from fbpipe.geometry import detect as D
from fbpipe.utils.fly_geometry import ABDOMEN, ABDOMEN_TIP, EYE, PROBOSCIS

NEW_MODEL_NAMES = {0: "abdomen", 1: "abdomenTip", 2: "eye", 3: "proboscis"}
OLD_MODEL_NAMES = {0: "eye", 1: "proboscis"}


class _OBB:
    def __init__(self, corners, cls, conf):
        self.xyxyxyxy = np.asarray(corners, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)

    def __len__(self):
        return len(self.cls)


class _Result:
    def __init__(self, obb):
        self.obb = obb


def test_class_map_is_built_by_name_in_any_id_order():
    m = D.class_map_by_name(NEW_MODEL_NAMES)
    assert m == {EYE: 2, PROBOSCIS: 3, ABDOMEN: 0, ABDOMEN_TIP: 1}
    assert D.class_map_by_name(["Eye", "PROBOSCIS", "abdomen", "abdomentip"]) == {EYE: 0, PROBOSCIS: 1, ABDOMEN: 2, ABDOMEN_TIP: 3}
    inv = D.id_to_label_map(NEW_MODEL_NAMES)
    assert inv == {2: EYE, 3: PROBOSCIS, 0: ABDOMEN, 1: ABDOMEN_TIP}


def test_class_map_refuses_a_model_without_the_abdomen_classes():
    with pytest.raises(ValueError, match="abdomen"):
        D.class_map_by_name(OLD_MODEL_NAMES)


def test_parts_from_result_labels_by_the_model_map_and_keeps_corner_order():
    corners = [
        [[10, 10], [50, 10], [50, 40], [10, 40]],    # cls 2 = eye
        [[60, 60], [70, 60], [70, 90], [60, 90]],    # cls 3 = proboscis
        [[0, 0], [1, 0], [1, 1], [0, 1]],            # cls 7 = unknown, skipped
    ]
    res = _Result(_OBB(corners, cls=[2, 3, 7], conf=[0.9, 0.5, 0.2]))
    parts = D.parts_from_result(res, D.id_to_label_map(NEW_MODEL_NAMES))
    assert [p.label for p in parts] == [EYE, PROBOSCIS]
    assert parts[0].score == pytest.approx(0.9)
    assert np.allclose(parts[0].pts, corners[0])
    assert D.parts_from_result(_Result(None), {}) == []
    assert D.parts_from_result(_Result(_OBB(np.zeros((0, 4, 2)), [], [])), {}) == []


def test_predict_proxy_injects_imgsz_and_iou():
    seen = {}

    class FakeModel:
        def predict(self, images, **kw):
            seen.update(kw)
            return ["r"] * len(images)

    proxy = D.PredictProxy(FakeModel(), imgsz=1088, iou=0.7)
    out = proxy.predict([1, 2], conf=0.25, verbose=False, device="cpu", half=False)
    assert out == ["r", "r"]
    assert seen["imgsz"] == 1088 and seen["iou"] == 0.7 and seen["conf"] == 0.25


def test_make_predict_fn_uses_the_production_backoff_and_cpu_fallback():
    pytest.importorskip("ultralytics")
    calls = []

    class FakeModel:
        def to(self, device):
            calls.append(("to", device))

        def predict(self, images, **kw):
            calls.append(("predict", len(images), kw.get("device"), kw.get("imgsz")))
            return ["r"] * len(images)

    fn = D.make_predict_fn(FakeModel(), imgsz=1088, iou=0.7, allow_cpu=True)
    assert fn([1, 2, 3], 0.25) == ["r", "r", "r"]
    assert any(c[0] == "predict" and c[3] == 1088 for c in calls)
