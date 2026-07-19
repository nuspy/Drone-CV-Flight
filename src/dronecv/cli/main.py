"""dronecv CLI.

The heavy imports (torch, cv2) happen inside commands so that `--help` stays
instant. Every command takes `--env` and resolves the layered config
(configs/default.yaml <- configs/envs/<env>.yaml <- flags).
"""

from __future__ import annotations

import typer

from dronecv.cli.console import console

app = typer.Typer(
    name="dronecv",
    help="GPS-denied drone visual navigation: train in sim, localize, guide, test.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.command("envs")
def envs_list() -> None:
    """List available environments (configs/envs/*.yaml)."""
    from dronecv.config import list_envs

    for name in list_envs():
        console.print(f"  [bold cyan]{name}[/bold cyan]")


@app.command()
def sim(
    env: str = typer.Option(..., "--env", help="Environment name"),
    port: int = typer.Option(None, "--port", help="Override sim port"),
) -> None:
    """Start the headless simulator server for ENV."""
    from dronecv.commands.sim_cmd import run_sim

    run_sim(env, port)


@app.command()
def capture(
    env: str = typer.Option(..., "--env"),
    plan: str = typer.Option("grid", "--plan", help="grid | orbit"),
    n: int = typer.Option(None, "--n", help="Cap the number of captures"),
    seed: int = typer.Option(None, "--seed"),
) -> None:
    """Collect a training capture dataset from the sim (headless or Unity)."""
    from dronecv.commands.capture_cmd import run_capture

    run_capture(env, plan, n, seed)


@app.command()
def train(
    env: str = typer.Option(..., "--env"),
    rounds: int = typer.Option(None, "--rounds", help="Max active-loop rounds"),
    budget: int = typer.Option(None, "--budget", help="Max total captures"),
    seed: int = typer.Option(None, "--seed"),
) -> None:
    """Run the active training loop (auto-sizes the dataset for ENV)."""
    from dronecv.commands.train_cmd import run_train

    run_train(env, rounds, budget, seed)


@app.command()
def evaluate(
    env: str = typer.Option(..., "--env"),
    bundle: str = typer.Option(None, "--model-bundle", help="Path to a model bundle"),
) -> None:
    """Evaluate a trained model bundle on fresh probe captures."""
    from dronecv.commands.evaluate_cmd import run_evaluate

    run_evaluate(env, bundle)


@app.command()
def localize(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7601, "--port"),
    env: str = typer.Option(..., "--env"),
    bundle: str = typer.Option(None, "--model-bundle"),
) -> None:
    """Stream live localization estimates from a running sim (companion-computer mode)."""
    from dronecv.commands.localize_cmd import run_localize

    run_localize(env, host, port, bundle)


@app.command("test-flight")
def test_flight(
    env: str = typer.Option(..., "--env"),
    bundle: str = typer.Option(None, "--model-bundle"),
    episodes: int = typer.Option(None, "--episodes"),
    seed: int = typer.Option(None, "--seed"),
) -> None:
    """Run the automated reliability test (accuracy + autonomous target reach)."""
    from dronecv.commands.test_flight_cmd import run_test_flight

    raise SystemExit(run_test_flight(env, bundle, episodes, seed))


@app.command("run-all")
def run_all(
    env: str = typer.Option(..., "--env"),
    budget: int = typer.Option(None, "--budget"),
    episodes: int = typer.Option(None, "--episodes"),
    seed: int = typer.Option(None, "--seed"),
    bundle: str = typer.Option(None, "--model-bundle", help="Skip training, use this bundle"),
) -> None:
    """Everything: capture -> auto-sized training -> flight test -> reliability report."""
    from dronecv.commands.run_all_cmd import run_all_pipeline

    raise SystemExit(run_all_pipeline(env, budget, episodes, seed, bundle))


@app.command("localize-photo")
def localize_photo(
    image: str = typer.Argument(..., help="Path to a photo (jpeg/png) of the trained environment"),
    env: str = typer.Option(..., "--env"),
    bundle: str = typer.Option(None, "--model-bundle"),
) -> None:
    """Localize a single photo (e.g. an aerial picture) against a trained model."""
    from dronecv.commands.photo_cmd import run_localize_photo

    run_localize_photo(env, image, bundle)


@app.command()
def serve(
    env: str = typer.Option(..., "--env"),
    bundle: str = typer.Option(None, "--model-bundle"),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Start the HTTP inference server (photo -> coordinates; used by the Android app)."""
    from dronecv.config import load_config
    from dronecv.server.app import run_server

    cfg = load_config(env)
    from pathlib import Path

    bundle_dir = Path(bundle) if bundle else cfg.bundle_dir
    run_server(bundle_dir, host, port, cfg.artifacts_dir / "mobile_bundle")


@app.command()
def export(
    env: str = typer.Option(..., "--env"),
    bundle: str = typer.Option(None, "--model-bundle"),
) -> None:
    """Export the trained model as an on-device (ONNX) bundle for the Android app."""
    from pathlib import Path

    from dronecv.config import load_config
    from dronecv.export.mobile import export_mobile_bundle
    from dronecv.training.bundle import ModelBundle

    cfg = load_config(env)
    mb = ModelBundle.load(Path(bundle) if bundle else cfg.bundle_dir)
    out = export_mobile_bundle(mb, cfg.artifacts_dir / "mobile_bundle")
    console.print(f"[green]mobile bundle exported to {out}[/green]")


protocol_app = typer.Typer(help="Protocol utilities")
app.add_typer(protocol_app, name="protocol")


@protocol_app.command("verify")
def protocol_verify(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7601, "--port"),
) -> None:
    """Acceptance test for a live sim bridge (use this to validate the Unity side)."""
    from dronecv.commands.protocol_verify_cmd import run_protocol_verify

    raise SystemExit(run_protocol_verify(host, port))


report_app = typer.Typer(help="Report utilities")
app.add_typer(report_app, name="report")


@report_app.command("open")
def report_open(env: str = typer.Option(..., "--env")) -> None:
    """Print the path of the latest reliability report for ENV."""
    from dronecv.config import load_config

    cfg = load_config(env)
    path = cfg.report_dir / "report.html"
    if path.exists():
        console.print(f"[bold green]{path.resolve()}[/bold green]")
    else:
        console.print(f"[red]no report yet for {env} — run: dronecv run-all --env {env}[/red]")
        raise SystemExit(1)


if __name__ == "__main__":
    app()
