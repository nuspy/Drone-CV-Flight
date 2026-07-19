"""End-to-end: the full one-command pipeline on the CI environment.

This is the CI gate: capture -> active-loop training -> blind localization
accuracy test -> autonomous target reach -> reliability report, all through
the real wire protocol against the headless sim. Sized to finish on a CPU
GitHub runner.
"""

import json
from pathlib import Path

import pytest

from dronecv.config import load_config

ROOT = Path(__file__).parent.parent.parent

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("artifacts")


@pytest.fixture(scope="module")
def pipeline(out_dir):
    """Run the whole pipeline once; individual tests assert on the pieces."""
    import asyncio

    cfg = load_config(
        "headless_ci",
        overrides={
            "report": {"out_dir": str(out_dir)},
            "active_loop": {"budget_captures": 500, "max_rounds": 2, "probe_captures": 40},
            "harness": {"episodes": 2, "time_cap_s": 90.0, "kidnap_test": True},
        },
        root=ROOT,
    )
    from dronecv.commands.test_flight_cmd import run_test_flight_async
    from dronecv.training.active_loop import run_active_loop

    train_result = asyncio.run(run_active_loop(cfg))
    rc = asyncio.run(run_test_flight_async(cfg, None))
    return cfg, train_result, rc


def test_active_loop_produced_bundle_and_report(pipeline):
    cfg, train_result, _rc = pipeline
    assert train_result["rounds"] >= 1
    assert train_result["stop_reason"] in ("converged", "plateau", "budget_exhausted", "max_rounds")
    assert (cfg.bundle_dir / "manifest.json").exists()
    assert (cfg.bundle_dir / "terrain_prior.npz").exists()
    report = json.loads((cfg.artifacts_dir / "training_report.json").read_text())
    assert report["history"], "training history must not be empty"
    # The loop measured how much data the environment needed.
    assert report["captures_used"] > 100


def test_anchor_provenance_stamped(pipeline):
    cfg, _tr, _rc = pipeline
    manifest = json.loads((cfg.bundle_dir / "manifest.json").read_text())
    # headless_ci embeds geo metadata -> anchor must come from the sim.
    assert manifest["anchor"]["source"] == "sim_metadata"
    assert manifest["anchor"]["lat0"] == pytest.approx(45.4642)


def test_flight_report_written_with_all_sections(pipeline):
    cfg, _tr, _rc = pipeline
    payload = json.loads((cfg.report_dir / "report.json").read_text())
    acc = payload["accuracy"]
    assert acc["n_frames"] > 50
    assert acc["pos_err_h_p95_m"] is not None
    reach = payload["reach"]
    assert reach["n_episodes"] == 2
    assert (cfg.report_dir / "report.html").exists()
    html = (cfg.report_dir / "report.html").read_text()
    assert "data:image/png;base64," in html


def test_reliability_gate(pipeline):
    """The actual CI quality gate: blind localization within CI thresholds and
    the autonomous reach behavior sane."""
    cfg, _tr, rc = pipeline
    payload = json.loads((cfg.report_dir / "report.json").read_text())
    acc = payload["accuracy"]
    reach = payload["reach"]
    # Localization: better than half the world size at p95 (sanity) and
    # within the configured CI threshold for the pass verdict.
    assert acc["pos_err_h_p95_m"] < cfg.world.size_m / 2
    assert reach["collisions"] == 0
    # The pipeline's own verdict must match the exit code.
    assert (rc == 0) == payload["passed"]
