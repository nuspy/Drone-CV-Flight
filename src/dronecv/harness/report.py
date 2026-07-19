"""Reliability report: machine-readable JSON + human HTML with plots."""

from __future__ import annotations

import base64
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template

TEMPLATE = Template(
    """<!doctype html>
<html><head><meta charset="utf-8"><title>dronecv reliability report — {{ env }}</title>
<style>
 body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 960px; color: #1a2733; }
 h1, h2 { color: #0b3954; }
 .verdict { font-size: 1.3rem; padding: .6rem 1rem; border-radius: 8px; display: inline-block; }
 .pass { background: #d7f5dd; color: #14632a; } .fail { background: #fbdcdc; color: #8f1d1d; }
 table { border-collapse: collapse; margin: 1rem 0; }
 td, th { border: 1px solid #cdd7de; padding: .35rem .7rem; text-align: left; }
 th { background: #eef3f6; }
 img { max-width: 100%; border: 1px solid #e0e6ea; border-radius: 6px; margin: .5rem 0; }
 .muted { color: #5c6b78; font-size: .9rem; }
</style></head><body>
<h1>dronecv reliability report</h1>
<p class="muted">environment <b>{{ env }}</b> · generated {{ generated }} · anchor {{ anchor }}</p>
<p><span class="verdict {{ 'pass' if passed else 'fail' }}">{{ 'PASSED' if passed else 'FAILED' }}</span></p>

<h2>Localization accuracy</h2>
<table>
{% for k, v in accuracy_rows %}<tr><th>{{ k }}</th><td>{{ v }}</td></tr>{% endfor %}
</table>
{% for img in accuracy_figs %}<img src="data:image/png;base64,{{ img }}">{% endfor %}

<h2>Autonomous target reach</h2>
<table>
{% for k, v in reach_rows %}<tr><th>{{ k }}</th><td>{{ v }}</td></tr>{% endfor %}
</table>
<table>
<tr><th>ep</th><th>target</th><th>success</th><th>arrival declared</th><th>true dist at arrival</th><th>min dist to target</th><th>standoff violation</th><th>collided</th><th>duration</th></tr>
{% for e in episodes %}
<tr><td>{{ e.episode }}</td><td>{{ e.target_source }}</td><td>{{ '✅' if e.success else '❌' }}</td>
<td>{{ e.declared_arrival }}</td><td>{{ '%.1f m' % e.true_dist_at_arrival_m if e.true_dist_at_arrival_m is not none else '—' }}</td>
<td>{{ '%.1f m' % e.min_true_dist_to_target_m }}</td><td>{{ e.standoff_violation }}</td>
<td>{{ e.collided }}</td><td>{{ '%.0f s' % e.duration_s }}</td></tr>
{% endfor %}
</table>

{% if training_fig %}<h2>Active training loop</h2>
<p class="muted">error vs data curve — how much data this environment needed ({{ stop_reason }})</p>
<img src="data:image/png;base64,{{ training_fig }}">{% endif %}
</body></html>"""
)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _accuracy_figures(acc: dict) -> list[str]:
    figs = []
    traj = acc.get("trajectory")
    if traj and traj["truth"]:
        truth = np.array(traj["truth"])
        est = np.array(traj["est"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot(truth[:, 0], truth[:, 1], "-", color="#0b3954", label="truth")
        axes[0].plot(est[:, 0], est[:, 1], "--", color="#d1495b", label="estimate", alpha=0.8)
        axes[0].set_title("trajectory (ENU)")
        axes[0].set_xlabel("east m")
        axes[0].set_ylabel("north m")
        axes[0].axis("equal")
        axes[0].legend()
        t = np.array(traj["t"])
        axes[1].plot(t, traj["err_h"], color="#d1495b", label="horiz err m")
        ax2 = axes[1].twinx()
        ax2.plot(t, traj["confidence"], color="#3a7d44", alpha=0.7, label="confidence")
        ax2.set_ylim(0, 1.05)
        if traj.get("kidnap_at"):
            axes[1].axvline(traj["kidnap_at"], color="#888", linestyle=":", label="kidnap")
        axes[1].set_title("error and confidence over time")
        axes[1].set_xlabel("sim time s")
        axes[1].legend(loc="upper left")
        figs.append(_fig_to_b64(fig))
    rel = acc.get("reliability_bins") or []
    if rel:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot([0, 1], [0, 1], ":", color="#999")
        ax.plot([r["mean_conf"] for r in rel], [r["empirical"] for r in rel], "o-", color="#0b3954")
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("empirical P(err < target)")
        ax.set_title("confidence calibration")
        figs.append(_fig_to_b64(fig))
    return figs


def _training_figure(history: list[dict]) -> str | None:
    if not history:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.6))
    xs = [h["captures_total"] for h in history]
    ax.plot(xs, [h["probe_p95_m"] for h in history], "o-", color="#0b3954", label="probe p95 err m")
    ax.plot(xs, [h["probe_alt_p95_m"] for h in history], "s--", color="#d1495b", label="alt p95 m")
    ax.set_xlabel("captures used")
    ax.set_ylabel("error (m)")
    ax.legend()
    ax.set_title("active loop: error vs data")
    return _fig_to_b64(fig)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def write_report(
    out_dir: Path,
    env: str,
    anchor: dict,
    results: dict,
    training_report: dict | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "env": env,
        "generated_utc": datetime.now(UTC).isoformat(),
        "anchor": anchor,
        "passed": results["passed"],
        "accuracy": {k: v for k, v in results["accuracy"].items() if k != "trajectory"},
        "reach": results["reach"],
        "training": (
            {k: training_report[k] for k in ("stop_reason", "rounds", "captures_used", "budget")}
            if training_report
            else None
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2))

    acc = results["accuracy"]
    accuracy_rows = [
        ("frames evaluated", _fmt(acc.get("n_frames"))),
        ("horizontal error p50 / p95", f"{_fmt(acc.get('pos_err_h_p50_m'))} / {_fmt(acc.get('pos_err_h_p95_m'))} m"),
        ("horizontal RMSE", f"{_fmt(acc.get('pos_err_h_rmse_m'))} m"),
        ("altitude MAE", f"{_fmt(acc.get('alt_err_mae_m'))} m"),
        ("heading error p50", f"{_fmt(acc.get('heading_err_p50_deg'))} deg"),
        ("velocity error p50", f"{_fmt(acc.get('vel_err_p50_ms'))} m/s"),
        ("mean confidence", _fmt(acc.get("mean_confidence"))),
        ("confidence ECE", _fmt(acc.get("confidence_ece"))),
        ("time to first fix", f"{_fmt(acc.get('time_to_first_fix_s'))} s"),
        ("kidnap recovery", f"{_fmt(acc.get('kidnap_recovery_s'))} s"),
    ]
    reach = results["reach"]
    reach_rows = [
        ("episodes", _fmt(reach.get("n_episodes"))),
        ("success rate", _fmt(reach.get("success_rate"))),
        ("declared arrivals", _fmt(reach.get("declared_arrival_rate"))),
        ("mean duration (successes)", f"{_fmt(reach.get('mean_duration_s'))} s"),
        ("path efficiency", _fmt(reach.get("path_efficiency"))),
        ("standoff violations", _fmt(reach.get("standoff_violations"))),
        ("collisions", _fmt(reach.get("collisions"))),
    ]

    class _E:
        def __init__(self, d):
            self.__dict__.update(d)

    html = TEMPLATE.render(
        env=env,
        generated=payload["generated_utc"],
        anchor=f"{anchor.get('lat0'):.4f}, {anchor.get('lon0'):.4f} ({anchor.get('source')})",
        passed=results["passed"],
        accuracy_rows=accuracy_rows,
        reach_rows=reach_rows,
        episodes=[_E(e) for e in reach.get("episodes", [])],
        accuracy_figs=_accuracy_figures(acc),
        training_fig=_training_figure((training_report or {}).get("history_curve", [])),
        stop_reason=(training_report or {}).get("stop_reason", ""),
    )
    path = out_dir / "report.html"
    path.write_text(html)
    return path
