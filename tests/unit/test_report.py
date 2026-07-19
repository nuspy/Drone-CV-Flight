import json

from dronecv.harness.report import write_report


def _fake_results():
    return {
        "passed": True,
        "accuracy": {
            "n_frames": 100,
            "n_initialized": 98,
            "pos_err_h_p50_m": 6.2,
            "pos_err_h_p95_m": 14.8,
            "pos_err_h_rmse_m": 7.5,
            "alt_err_mae_m": 2.1,
            "heading_err_p50_deg": 3.3,
            "vel_err_p50_ms": 0.8,
            "mean_confidence": 0.71,
            "within_target_frac": 0.9,
            "confidence_ece": 0.11,
            "reliability_bins": [
                {"bin": [0.4, 0.6], "mean_conf": 0.5, "empirical": 0.55, "n": 30},
                {"bin": [0.6, 0.8], "mean_conf": 0.7, "empirical": 0.75, "n": 50},
            ],
            "time_to_first_fix_s": 1.2,
            "kidnap_recovery_s": 8.4,
            "trajectory": {
                "truth": [[0, 0], [10, 5], [20, 10]],
                "est": [[1, 0], [11, 6], [19, 9]],
                "err_h": [1.0, 1.4, 1.4],
                "confidence": [0.3, 0.6, 0.8],
                "t": [0.0, 0.2, 0.4],
                "kidnap_at": None,
            },
        },
        "reach": {
            "n_episodes": 2,
            "success_rate": 1.0,
            "declared_arrival_rate": 1.0,
            "mean_duration_s": 45.0,
            "path_efficiency": 0.85,
            "standoff_violations": 0,
            "collisions": 0,
            "episodes": [
                {
                    "episode": 0, "target_source": "coordinates", "success": True,
                    "declared_arrival": True, "true_dist_at_arrival_m": 4.2,
                    "min_true_dist_to_target_m": 11.0, "standoff_violation": False,
                    "collided": False, "duration_s": 40.0,
                },
                {
                    "episode": 1, "target_source": "visual", "success": True,
                    "declared_arrival": True, "true_dist_at_arrival_m": 6.0,
                    "min_true_dist_to_target_m": 12.5, "standoff_violation": False,
                    "collided": False, "duration_s": 50.0,
                },
            ],
        },
    }


def test_write_report_json_and_html(tmp_path):
    anchor = {"lat0": 45.4642, "lon0": 9.19, "alt0": 120.0, "true_north_offset_deg": 0.0, "source": "env_config"}
    training = {
        "stop_reason": "converged", "rounds": 2, "captures_used": 500, "budget": 800,
        "history_curve": [
            {"captures_total": 280, "probe_p95_m": 40.0, "probe_alt_p95_m": 9.0},
            {"captures_total": 500, "probe_p95_m": 22.0, "probe_alt_p95_m": 6.0},
        ],
    }
    path = write_report(tmp_path, "test_env", anchor, _fake_results(), training)
    assert path.exists()
    html = path.read_text()
    assert "PASSED" in html
    assert "data:image/png;base64," in html  # plots embedded
    assert "45.4642" in html

    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["passed"] is True
    assert payload["accuracy"]["pos_err_h_p95_m"] == 14.8
    assert "trajectory" not in payload["accuracy"]  # bulky arrays stay out of JSON
    assert payload["reach"]["success_rate"] == 1.0
    assert payload["training"]["stop_reason"] == "converged"


def test_write_report_handles_missing_data(tmp_path):
    anchor = {"lat0": 0.0, "lon0": 0.0, "alt0": 0.0, "true_north_offset_deg": 0.0, "source": "env_config"}
    results = {
        "passed": False,
        "accuracy": {"n_frames": 0},
        "reach": {"n_episodes": 0},
    }
    path = write_report(tmp_path, "empty", anchor, results, None)
    assert "FAILED" in path.read_text()
