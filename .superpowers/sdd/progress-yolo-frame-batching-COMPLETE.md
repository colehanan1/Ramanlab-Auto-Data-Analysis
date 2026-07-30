# YOLO Frame-Batching — Progress Ledger

Plan: docs/superpowers/plans/2026-06-25-yolo-frame-batching.md (amended 2026-06-25 for dynamic .engine batching)
Branch: feature/yolo-frame-batching  (off main)
Skill: superpowers:subagent-driven-development
STATUS: ALL 7 TASKS COMPLETE + final review clean. Branch KEPT AS-IS (not merged/pushed) per user, 2026-06-26. HEAD=4aae4d0. Pending = manual GPU re-export to activate.

## Key decision (user)
- User runs a `.engine` (TensorRT), NOT a .pt. Plain plan would clamp engine to B=1 (no speedup).
- DECISION: re-export engine with `dynamic=True, batch=32` + add `engine_supports_batch` guard flag so batch-capable engines batch. Default INFERENCE_BATCH_SIZE=32 (user owns whole 24GB 3090, wants max throughput; 32 saturates at 640 letterbox).
- Parity: loop machinery stays bit-identical (Task 5 CPU tests); dynamic engine is deterministic/run-to-run stable but NOT bit-identical to old static engine -> one-time re-process of already-done videos.

## Baseline (branch start)
- tests/test_yolo_infer.py tests/test_pseudolabel_export.py tests/test_config_settings.py -> 15 passed (clean).
- Full-suite baseline to be confirmed in Task 7 (prior ledger noted ~374 passed/1 xfailed on merged perf branch).
- Unrelated uncommitted WIP left untouched on branch: scripts/analysis/envelope_combined.py (M), tests/test_secure_copy_cleanup.py (??).

## Suite baseline (full, on branch)
- 402 passed, 1 xfailed, 3 env-dependent errors (HomeAssistant/test_influxdb_enclosure.py — need live InfluxDB). "No new failures" = stay at this.

## Tasks
- Task 1: complete (commit aa7f9b6, review clean — Spec ✅, Approved, 0 issues).
- Task 2: complete (commit d30d871, Spec ✅, Approved; 2 Minor findings recorded below).
- Task 3: complete (commit e293e12, opus review — Spec ✅, Approved; OOM backoff provably correct; 2 Minor below).
- Task 4: complete (commits df2defc + fix b735537, opus review+re-review — Spec ✅, Approved). Parity bit-identical at B={1,3,4,8}; parity test PROVEN non-vacuous for both _next_id leak and prev_gray-boundary regression. Helper threads prev_gray + order correct. Includes Task 5 Step 1 (parity test).
- Task 5: complete (commit 21b6dc6, Spec ✅, Approved). 4 edge tests: partial-last-batch, EOF-on-boundary (proves no predict([])), max_frame cap, truncated-results AssertionError. 1 Minor below.
- Task 6: complete (commit 4aae4d0, Spec ✅, Approved). export_tensorrt.py --batch(32)/--dynamic(default True), threads dynamic+batch, no imgsz, gated reminder log. 1 Minor below (log wording).
- Task 7 (CI portion, controller-run): DONE. Targeted suite 28 passed; full suite 417 passed/1 xfailed/3 pre-existing InfluxDB errors = ZERO new failures; engine-guard truth table verified (pt->32, static engine->1, dynamic engine+flag->32); export --help shows --batch/--dynamic. Real-GPU steps (re-export/throughput/parity/OOM) are MANUAL — handed to user.
- Task 2: pending — config knobs inference_batch_size=32 + engine_supports_batch=False.
- Task 3: pending — batched_predict_fn + OOM backoff + scan adapter.
- Task 4: pending — _run_chunked_inference + main() rewire (engine guard).
- Task 5: pending — parity + edge tests (CPU).
- Task 6: pending — dynamic-batch TensorRT export script (--batch/--dynamic).
- Task 7: pending — verification (CI here; real-GPU re-export + throughput is MANUAL, user-run).

## Cross-cutting execution decisions (controller)
- Testability: build `batched_predict_fn` via a module-level factory `_make_batched_predict_fn(model, get_device, set_device, allow_cpu)` so the OOM-backoff loop is unit-testable with a fake model. main() calls the factory; single-frame `predict_fn` adapter wraps it.
- TDD ownership of tests/test_yolo_infer_batching.py (created Task 3):
  - Task 3 writes the OOM-backoff unit test (against the factory).
  - Task 4 writes the parity test (plan Task 5 Step 1) FIRST as its red test, then implements _run_chunked_inference + main() rewire.
  - Task 5 adds edge/mitigation tests (plan Task 5 Step 2): partial last batch, EOF-on-boundary, max_frame cap, truncated-results AssertionError.
- All three tasks edit the same test file sequentially (no conflict). importorskip("ultralytics") at top (ultralytics IS installed here).

## FINAL whole-branch review (opus, range 3dc1379..4aae4d0)
- Verdict: READY TO MERGE = YES. No Critical, no Important. Inference path end-to-end coherent; engine guard airtight (static engine never gets B>1, warm-up incl.); OOM backoff terminates + propagates non-CUDA errors; config knobs wired+consumed; all parity gates honored; tail+t0 preserved; no scope creep.
- All 5 rolled-up Minors triaged DEFER-safe (none block merge). Recommended (optional, next time file is touched): land the T2 delenv one-liner + symmetric engine_supports_batch YAML test.
- Remaining = MANUAL GPU (Task 7): dynamic re-export, util/throughput, blake2b determinism, OOM-under-pressure — to run on the 3090 before flipping engine_supports_batch:true in live config.

## Minor findings roll-up (for final review)
- Task 2: tests/test_config_inference_batch.py test_env_override — add monkeypatch.delenv("ENGINE_SUPPORTS_BATCH", raising=False) for hermeticity (latent, not a current defect).
- Task 2: no test for engine_supports_batch reading from YAML when env absent (coverage gap, not spec-required).
- Task 3: tests/test_yolo_infer_batching.py:2 — `import pytest; pytest.importorskip(...)` one-line compound (cosmetic).
- Task 3: no single test combining live-device-flip WITH sub-batch halving (each covered separately; optional hardening).
- Task 5: test_truncated_results_raises short_predict has a defensive `else results` branch inert for the params used (cosmetic; add a comment noting n=4,B=4 single-batch assumption).
- Task 6: export_tensorrt.py reminder log wording differs slightly from the brief's verbatim string (cosmetic; same content/knobs).
