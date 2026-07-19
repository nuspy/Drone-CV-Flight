from pathlib import Path

import pytest

from dronecv.config import Config, _deep_merge, list_envs, load_config

ROOT = Path(__file__).parent.parent.parent


def test_deep_merge():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10}, "e": 4}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3, "e": 4}
    assert base["a"]["b"] == 1  # no mutation


def test_load_every_shipped_env():
    for env in list_envs(ROOT):
        cfg = load_config(env, root=ROOT)
        assert isinstance(cfg, Config)
        assert cfg.env.name == env


def test_env_overrides_default():
    cfg = load_config("headless_ci", root=ROOT)
    assert cfg.sim.image_width == 64  # overridden
    assert cfg.sim.fov_deg == 70.0  # default kept
    assert cfg.active_loop.budget_captures == 800
    assert cfg.guidance.standoff_m == 10.0


def test_cli_overrides_win():
    cfg = load_config("headless_ci", overrides={"active_loop": {"budget_captures": 123}}, root=ROOT)
    assert cfg.active_loop.budget_captures == 123


def test_unknown_env_lists_available():
    with pytest.raises(FileNotFoundError, match="headless_ci"):
        load_config("nope_not_real", root=ROOT)


def test_unknown_key_rejected():
    with pytest.raises(Exception):
        load_config("headless_ci", overrides={"guidance": {"standof_m": 5}}, root=ROOT)


def test_artifact_paths():
    cfg = load_config("headless_ci", root=ROOT)
    assert str(cfg.bundle_dir).endswith("artifacts/headless_ci/model_bundle")
