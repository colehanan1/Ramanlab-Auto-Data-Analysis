import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from rigmon.app import create_app, sse_frames
from test_rigmon_hub import PI2, _hub, snap  # tests/ is on sys.path


@pytest.fixture
def ctx(tmp_path):
    hub, store, notifier, transfer, finalizer, workers = _hub(tmp_path)
    transfer.staging_dir = lambda rig, batch: tmp_path / "s" / rig / batch
    finalizer.abort = lambda rig, batch: "aborted"
    client = TestClient(create_app(hub, start_polling=False))
    return hub, store, workers, finalizer, client, tmp_path


def test_state_endpoint(ctx):
    hub, *_, client, _ = ctx
    hub.ingest("flybehavior2", snap(), 100.0)
    body = client.get("/api/state").json()
    assert body["rigs"]["flybehavior2"]["status"]["run"]["batch_dir"] == "b"


def test_mute_round_trip_and_unknown_rig(ctx):
    hub, store, *_, client, _ = ctx
    assert client.post("/api/mute/flybehavior2", json={"muted": True}).json() == \
        {"rig": "flybehavior2", "muted": True}
    assert store.is_muted("flybehavior2")
    assert client.post("/api/mute/nope", json={"muted": True}).status_code == 404


def test_complete_status_abort(ctx):
    hub, store, workers, finalizer, client, _ = ctx
    url = "/api/sessions/flybehavior2/b"
    r = client.post(f"{url}/complete", json={"output_dir": "/pi/b", "push_to": "u@h:/d/",
                                             "manifest": [{"path": "a", "bytes": 1}]})
    assert r.status_code == 202 and r.json() == {"state": "pending"}
    assert client.get(f"{url}/status").json()["state"] == "pending"
    assert client.post(f"{url}/abort").json() == {"state": "aborted"}
    assert client.get("/api/sessions/flybehavior2/zzz/status").json()["state"] == "unknown"
    assert client.post("/api/sessions/nope/b/complete", json={
        "output_dir": "", "push_to": "", "manifest": []}).status_code == 404


def test_media_serves_staged_mp4_and_refuses_traversal(ctx):
    hub, store, *_, client, tmp_path = ctx
    d = tmp_path / "s" / "flybehavior2" / "b"
    d.mkdir(parents=True)
    (d / "t1.mp4").write_bytes(b"mp4data")
    r = client.get("/media/flybehavior2/b/t1.mp4")
    assert r.status_code == 200 and r.content == b"mp4data"
    assert client.get("/media/flybehavior2/b/t1.h264").status_code == 404
    assert client.get("/media/flybehavior2/..%2F..%2Fetc/passwd.mp4").status_code == 404
    assert client.get("/media/nope/b/t1.mp4").status_code == 404


def test_media_follows_a_landed_session(ctx):
    hub, store, *_, client, tmp_path = ctx
    dest = tmp_path / "Data" / "flys_New" / "C" / "b"
    dest.mkdir(parents=True)
    (dest / "t1.mp4").write_bytes(b"landed")
    store.set_session("flybehavior2", "b", "landed", dest=str(dest))
    assert client.get("/media/flybehavior2/b/t1.mp4").content == b"landed"


def test_index_is_served(ctx):
    *_, client, _ = ctx
    r = client.get("/")
    assert r.status_code == 200 and "Rig Monitor" in r.text


def test_sse_frames_emit_state_until_disconnect():
    states = iter([{"n": 1}, {"n": 2}])
    disconnected = iter([False, False, True])

    async def go():
        out = []
        async for frame in sse_frames(lambda: next(states),
                                      lambda: _async(next(disconnected)),
                                      sleep=lambda s: _async(None)):
            out.append(frame)
        return out

    frames = asyncio.run(go())
    assert [json.loads(f[len("data: "):]) for f in frames] == [{"n": 1}, {"n": 2}]
    assert all(f.endswith("\n\n") for f in frames)


async def _async(v):
    return v
