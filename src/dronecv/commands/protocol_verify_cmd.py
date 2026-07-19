"""Acceptance test for a live sim bridge — run it against the Unity scene to
validate the C# implementation without any Python knowledge of Unity:

    dronecv protocol verify --host 127.0.0.1 --port 7601

Every check exercises one protocol behavior the pipeline depends on."""

from __future__ import annotations

import asyncio

import numpy as np
from rich.table import Table

from dronecv.cli.console import console
from dronecv.protocol import messages as m
from dronecv.protocol.framing import ProtocolError
from dronecv.protocol.sim_client import SimClient


async def _checks(host: str, port: int) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def ok(name: str, detail: str = "") -> None:
        results.append((name, True, detail))

    def fail(name: str, detail: str) -> None:
        results.append((name, False, detail))

    # --- handshake ---
    try:
        client = await SimClient.connect(host, port, role="capture", timeout=5.0)
        ok("handshake", f"{client.ack.sim_kind} env='{client.ack.env_id}' v{client.ack.version}")
    except Exception as e:  # noqa: BLE001
        fail("handshake", str(e))
        return results

    if client.ack.camera:
        ok("camera info", f"{client.ack.camera.width}x{client.ack.camera.height} fov {client.ack.camera.fov_deg}")
    else:
        fail("camera info", "hello_ack.camera missing")
    ok("geo metadata", "embedded" if client.geo_meta else "absent (env-config fallback will be used)")

    # --- env info ---
    try:
        info = await client.env_info()
        span = np.array(info.bounds_max_sim) - np.array(info.bounds_min_sim)
        assert (span > 0).all()
        ok("env_info", f"volume {span[0]:.0f} x {span[1]:.0f} x {span[2]:.0f} m")
        center = (np.array(info.bounds_max_sim) + np.array(info.bounds_min_sim)) / 2
        probe_y = info.bounds_max_sim[1] - 1.0
    except Exception as e:  # noqa: BLE001
        fail("env_info", str(e))
        await client.close()
        return results

    # --- capture: rgb + depth + sun ---
    try:
        result, blobs = await client.capture(
            np.array([center[0], probe_y, center[2]]), yaw_deg=45.0, pitch_deg=60.0
        )
        img = blobs.get("rgb")
        depth = blobs.get("depth")
        assert img is not None and img.ndim == 3 and img.shape[2] == 3, "rgb blob missing/malformed"
        assert depth is not None and depth.ndim == 2, "depth blob missing/malformed"
        assert (depth > 0).any(), "depth image contains no hits"
        ok("capture rgb+depth", f"rgb {img.shape}, depth valid {(depth > 0).mean():.0%}")
        if result.sun_azimuth_deg is not None:
            ok("capture sun angles", f"az {result.sun_azimuth_deg:.1f} el {result.sun_elevation_deg:.1f}")
        else:
            fail("capture sun angles", "sun angles not returned (want included 'sun')")
        # Determinism: same pose twice -> identical pixels.
        _, blobs2 = await client.capture(np.array([center[0], probe_y, center[2]]), 45.0, 60.0)
        if np.array_equal(img, blobs2["rgb"]):
            ok("capture determinism", "identical pixels for identical pose")
        else:
            fail("capture determinism", "same pose returned different pixels")
    except Exception as e:  # noqa: BLE001
        fail("capture rgb+depth", str(e))
    await client.close()

    # --- role enforcement: localizer must NOT get truth ---
    try:
        loc = await SimClient.connect(host, port, role="localizer", timeout=5.0)
        try:
            await loc.subscribe(["truth"])
            fail("truth denied to localizer", "subscription was granted — MUST be refused")
        except ProtocolError:
            ok("truth denied to localizer")
        await loc.close()
    except Exception as e:  # noqa: BLE001
        fail("truth denied to localizer", str(e))

    # --- harness: reset, truth+sensors streaming, command ---
    try:
        har = await SimClient.connect(host, port, role="harness", timeout=5.0)
        done = await har.reset(seed=1)
        ok("reset", f"spawn at {np.round(done.pos_sim, 1).tolist()}")
        await har.subscribe(["sensors", "truth"])
        ok("truth granted to harness")
        await har.send_command(vel_sim=np.array([2.0, 0.0, 0.0]), yaw_rate_dps=10.0)
        frames = truths = 0
        first_pos = last_pos = None
        async with asyncio.timeout(20.0):
            async for msg, blobs in har.stream():
                if isinstance(msg, m.SensorFrame):
                    frames += 1
                    assert "rgb" in blobs, "sensor_frame without rgb blob"
                elif isinstance(msg, m.TruthState):
                    truths += 1
                    if first_pos is None:
                        first_pos = np.array(msg.pos_sim)
                    last_pos = np.array(msg.pos_sim)
                if frames >= 6 and truths >= 6:
                    break
        ok("sensor streaming", f"{frames} frames received")
        ok("truth streaming", f"{truths} states received")
        if last_pos is not None and first_pos is not None and np.linalg.norm(last_pos - first_pos) > 0.2:
            ok("command applied", f"drone moved {np.linalg.norm(last_pos - first_pos):.1f} m")
        else:
            fail("command applied", "drone did not move under a velocity command")
        await har.close()
    except Exception as e:  # noqa: BLE001
        fail("harness streaming", str(e))

    return results


def run_protocol_verify(host: str, port: int) -> int:
    results = asyncio.run(_checks(host, port))
    table = Table(title=f"protocol verify — {host}:{port}")
    table.add_column("check")
    table.add_column("result")
    table.add_column("detail")
    failed = 0
    for name, passed, detail in results:
        table.add_row(name, "[green]PASS[/green]" if passed else "[red]FAIL[/red]", detail)
        failed += not passed
    console.print(table)
    if failed:
        console.print(f"[red]{failed} check(s) failed[/red]")
        return 1
    console.print("[green]bridge implements the protocol correctly[/green]")
    return 0
