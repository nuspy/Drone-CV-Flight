"""Automated flight test: localization accuracy + autonomous target reach.

Architecture (the separation is the point of the whole test system):

    headless sim / Unity bridge
        |-- sensors channel --> LocalizationService (role=localizer, BLIND)
        |-- truth channel   --> this harness (role=harness)

The harness compares the blind localizer's estimates against truth frame by
frame, and in the reach phase lets a FlightController fly the drone purely on
estimates while truth is used only for judging success, collisions and
standoff violations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dronecv.config import Config
from dronecv.geo.anchor import GeoAnchor
from dronecv.geo.frames import heading_deg_from_enu_vector
from dronecv.guidance.controller import FlightController
from dronecv.guidance.guidance import compute_guidance
from dronecv.guidance.target import CoordinateTarget, VisualTarget, resolve_target
from dronecv.harness.metrics import EpisodeRecord, FrameRecord, accuracy_metrics, reach_metrics
from dronecv.localization.service import LocalizationService
from dronecv.protocol import messages as m
from dronecv.protocol.sim_client import SimClient
from dronecv.training.bundle import ModelBundle
from dronecv.util.logging import get_logger
from dronecv.util.seeding import rng as make_rng

log = get_logger("dronecv.harness")


@dataclass
class _Truth:
    pos_sim: np.ndarray
    vel_sim: np.ndarray
    yaw_deg: float
    collided: bool
    sim_time: float


class TruthTracker:
    """Background consumer of the harness truth channel."""

    def __init__(self, client: SimClient, anchor: GeoAnchor):
        self.client = client
        self.anchor = anchor
        self.by_frame: dict[int, _Truth] = {}
        self.latest_id = -1
        self.collided_ever = False
        self._task: asyncio.Task | None = None

    def truth_enu(self, frame_id: int) -> _Truth | None:
        return self.by_frame.get(frame_id)

    async def wait_for(self, frame_id: int, timeout: float = 0.5) -> _Truth | None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if frame_id in self.by_frame:
                return self.by_frame[frame_id]
            # Truth for a frame OLDER than what we already have will never
            # arrive — bail out instead of burning the whole timeout.
            if self.latest_id > frame_id + 50:
                return None
            await asyncio.sleep(0.005)
        return None

    def start(self) -> None:
        self._task = asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        try:
            async for msg, _blobs in self.client.stream():
                if isinstance(msg, m.TruthState):
                    import math

                    q = msg.quat_sim
                    yaw = math.degrees(2 * math.atan2(q[1], q[3]))
                    self.by_frame[msg.frame_id] = _Truth(
                        pos_sim=np.array(msg.pos_sim),
                        vel_sim=np.array(msg.vel_sim),
                        yaw_deg=yaw % 360.0,
                        collided=msg.collided,
                        sim_time=msg.sim_time,
                    )
                    self.collided_ever |= msg.collided
                    self.latest_id = msg.frame_id
                    if len(self.by_frame) > 4000:
                        for k in sorted(self.by_frame)[:-3000]:
                            del self.by_frame[k]
        except (asyncio.CancelledError, ConnectionError):
            pass

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        await self.client.close()


class FlightTestHarness:
    def __init__(self, cfg: Config, bundle: ModelBundle | Path):
        """`bundle`: single ModelBundle, or a tiled bundle directory (from
        hierarchical training). Tiled mode uses coordinate targets only and
        resolves terrain via the tile bundle under the target."""
        self.cfg = cfg
        self.bundle = bundle
        self._tiled = not isinstance(bundle, ModelBundle)
        self._tile_cache: dict[str, ModelBundle] = {}
        if self._tiled:
            import json as _json

            self._tiles = _json.loads((Path(bundle) / "tiling.json").read_text())["tiles"]

    def _bundle_at(self, e: float, n: float) -> ModelBundle:
        """The fine bundle responsible for a position (single-bundle mode:
        the bundle itself)."""
        if not self._tiled:
            return self.bundle
        from dronecv.training.tiled import tile_of

        tile = tile_of(self._tiles, e, n) or self._tiles[0]
        if tile["id"] not in self._tile_cache:
            if len(self._tile_cache) > 2:
                self._tile_cache.pop(next(iter(self._tile_cache)))
            self._tile_cache[tile["id"]] = ModelBundle.load(Path(self.bundle) / "tiles" / tile["id"])
        return self._tile_cache[tile["id"]]

    # ------------------------------------------------------------ entry point

    async def run(self) -> dict:
        results: dict = {}
        results["accuracy"] = await self._accuracy_phase()
        results["reach"] = await self._reach_phase()
        results["passed"] = self._passed(results)
        return results

    def _passed(self, results: dict) -> bool:
        thr = self.cfg.harness.pass_thresholds
        acc = results["accuracy"]
        reach = results["reach"]
        ok = True
        if acc.get("pos_err_h_p95_m") is None or acc["pos_err_h_p95_m"] > thr.p95_pos_err_m:
            ok = False
        if reach.get("success_rate") is None or reach["success_rate"] < thr.reach_success_rate:
            ok = False
        return ok

    async def _connect(self) -> tuple[SimClient, GeoAnchor, LocalizationService, TruthTracker]:
        """Three separate connections on purpose:
        - `harness`: control (reset/teleport/commands/target captures). No
          background reader, so request/response calls are safe.
        - a dedicated truth connection consumed ONLY by the TruthTracker.
          Sharing one connection here once deadlocked the whole test: the
          tracker's background recv interleaved with a capture() recv and
          corrupted the framing.
        - the localizer service's own sensor connection (blind role).
        """
        harness = await SimClient.connect(self.cfg.sim.host, self.cfg.sim.port, role="harness")
        anchor = GeoAnchor.resolve(self.cfg.env.anchor, harness.geo_meta)
        service = LocalizationService(self.cfg, self.bundle)
        await service.connect()
        truth_client = await SimClient.connect(self.cfg.sim.host, self.cfg.sim.port, role="harness")
        await truth_client.subscribe(["truth"])
        tracker = TruthTracker(truth_client, anchor)
        return harness, anchor, service, tracker

    # -------------------------------------------------------- accuracy phase

    async def _accuracy_phase(self) -> dict:
        log.info("accuracy phase: scripted flight, blind localizer vs truth")
        harness, anchor, service, tracker = await self._connect()
        try:
            done = await harness.reset(seed=self.cfg.env.seed, utc=self.cfg.env.start_utc)
            tracker.start()
            start = np.array(done.pos_sim)

            # Square waypoint pattern around the spawn, flown ON TRUTH by the
            # harness (the localizer only watches).
            half = self.cfg.world.size_m * 0.12
            wps = [
                start + np.array([half, 0, 0]),
                start + np.array([half, 10, half]),
                start + np.array([-half, 0, half]),
                start + np.array([-half, -10, -half]),
                start,
            ]
            frames: list[FrameRecord] = []
            kidnap_at = None
            kidnap_recovery_s = None
            n_frames = int(self.cfg.harness.time_cap_s * self.cfg.sim.sensor_hz * 0.5)
            wp_i = 0
            kidnapped = False

            gen = make_rng(self.cfg.env.seed + 5)
            async for frame_id, est in service.estimates():
                truth = await tracker.wait_for(frame_id)
                if truth is None:
                    continue
                # Harness-side truth pilot toward the current waypoint.
                delta = wps[wp_i] - truth.pos_sim
                if np.linalg.norm(delta[[0, 2]]) < 8.0 and wp_i < len(wps) - 1:
                    wp_i += 1
                vel = np.clip(delta * 0.4, -8.0, 8.0)
                await harness.send_command(vel_sim=vel, yaw_rate_dps=None)

                truth_enu = anchor.sim_to_enu(truth.pos_sim)
                truth_heading = (truth.yaw_deg + anchor.true_north_offset_deg) % 360.0
                frames.append(
                    FrameRecord(
                        t=truth.sim_time,
                        est_pos=est.pos_enu,
                        truth_pos=truth_enu,
                        est_heading=est.heading_deg,
                        truth_heading=truth_heading,
                        est_vel=est.vel_enu,
                        truth_vel=anchor.sim_to_enu(truth.vel_sim),
                        confidence=est.confidence,
                        initialized=est.initialized,
                        lost=est.lost,
                    )
                )

                # Mid-flight kidnap: teleport and time the re-localization.
                if (
                    self.cfg.harness.kidnap_test
                    and not kidnapped
                    and len(frames) == int(n_frames * 0.6)
                ):
                    offset = gen.uniform(-1, 1, 3) * np.array([80.0, 0.0, 80.0])
                    new_pos = np.clip(
                        truth.pos_sim + offset,
                        [-self.cfg.world.size_m * 0.4, 30, -self.cfg.world.size_m * 0.4],
                        [self.cfg.world.size_m * 0.4, 120, self.cfg.world.size_m * 0.4],
                    )
                    await harness.teleport(new_pos, yaw_deg=float(gen.uniform(0, 360)))
                    kidnapped = True
                    kidnap_at = truth.sim_time
                    log.info("kidnap: teleported the drone")
                if kidnapped and kidnap_recovery_s is None:
                    err = np.linalg.norm(frames[-1].est_pos[:2] - frames[-1].truth_pos[:2])
                    if truth.sim_time > kidnap_at and err < self.cfg.active_loop.thresholds.p95_pos_err_m:
                        kidnap_recovery_s = truth.sim_time - kidnap_at
                if len(frames) >= n_frames:
                    break

            metrics = accuracy_metrics(frames, self.cfg.active_loop.thresholds.p95_pos_err_m)
            metrics["kidnap_recovery_s"] = kidnap_recovery_s
            metrics["trajectory"] = {
                "truth": [f.truth_pos[:2].tolist() for f in frames],
                "est": [f.est_pos[:2].tolist() for f in frames],
                "err_h": [float(np.linalg.norm(f.est_pos[:2] - f.truth_pos[:2])) for f in frames],
                "confidence": [f.confidence for f in frames],
                "t": [f.t for f in frames],
                "kidnap_at": kidnap_at,
            }
            return metrics
        finally:
            await tracker.stop()
            await service.close()
            await harness.close()

    # ----------------------------------------------------------- reach phase

    async def _reach_phase(self) -> dict:
        log.info("reach phase: autonomous flight to targets on estimates only")
        episodes: list[EpisodeRecord] = []
        gen = make_rng(self.cfg.env.seed + 77)
        for ep in range(self.cfg.harness.episodes):
            episodes.append(await self._run_episode(ep, gen))
        return reach_metrics(episodes)

    async def _run_episode(self, ep: int, gen: np.random.Generator) -> EpisodeRecord:
        harness, anchor, service, tracker = await self._connect()
        try:
            done = await harness.reset(seed=self.cfg.env.seed * 100 + ep, utc=self.cfg.env.start_utc)
            tracker.start()
            start_sim = np.array(done.pos_sim)

            # Pick a target: random point well inside the world, at terrain+AGL.
            half = self.cfg.world.size_m * 0.35
            te, tn = float(gen.uniform(-half, half)), float(gen.uniform(-half, half))
            ground = self._bundle_at(te, tn).terrain.elevation(te, tn)
            t_alt = ground + float(gen.uniform(35.0, 60.0))
            target_enu = np.array([te, tn, t_alt])

            visual = (not self._tiled) and gen.random() < self.cfg.harness.visual_target_fraction
            if visual:
                # Photograph the target location through the harness client and
                # hand ONLY the image to the guidance stack.
                pos_sim = anchor.enu_to_sim(target_enu)
                result, blobs = await harness.capture(
                    pos_sim, yaw_deg=float(gen.uniform(0, 360)), pitch_deg=self.cfg.sim.camera_tilt_deg
                )
                fix = resolve_target(VisualTarget(blobs["rgb"]), anchor, self.bundle)
                source = "visual"
            else:
                lat, lon, alt = anchor.enu_to_geodetic(target_enu)
                fix = resolve_target(CoordinateTarget(lat, lon, alt), anchor)
                source = "coordinates"

            controller = FlightController(self.cfg.guidance, anchor)
            path_len = 0.0
            prev_truth_pos: np.ndarray | None = None
            min_dist = float("inf")
            standoff_violation = False
            declared = False
            true_dist_at_arrival = None
            at_goal_streak = 0
            t0 = None
            t_last = 0.0

            async for frame_id, est in service.estimates():
                truth = await tracker.wait_for(frame_id)
                if truth is None:
                    continue
                if t0 is None:
                    t0 = truth.sim_time
                t_last = truth.sim_time

                guidance = compute_guidance(
                    est.pos_enu, fix.pos_enu, self.cfg.guidance, fallback_bearing_deg=est.heading_deg
                )
                cmd = controller.command(est, guidance)
                await harness.send_command(vel_sim=cmd.vel_sim, yaw_rate_dps=cmd.yaw_rate_dps)

                truth_enu = anchor.sim_to_enu(truth.pos_sim)
                d_target = float(np.linalg.norm(truth_enu[:2] - fix.pos_enu[:2]))
                min_dist = min(min_dist, d_target)
                if d_target < self.cfg.guidance.standoff_m * 0.9:
                    standoff_violation = True
                if prev_truth_pos is not None:
                    path_len += float(np.linalg.norm(truth_enu - prev_truth_pos))
                prev_truth_pos = truth_enu

                at_goal_streak = at_goal_streak + 1 if guidance.at_goal else 0
                if at_goal_streak >= 3:
                    declared = True
                    goal_true_dist = float(np.linalg.norm(truth_enu[:2] - guidance.goal_enu[:2]))
                    true_dist_at_arrival = goal_true_dist
                    break
                if truth.sim_time - t0 > self.cfg.harness.time_cap_s:
                    break

            start_enu = anchor.sim_to_enu(start_sim)
            straight = float(np.linalg.norm(fix.pos_enu[:2] - start_enu[:2]))
            success = (
                declared
                and true_dist_at_arrival is not None
                and true_dist_at_arrival < self.cfg.harness.success_radius_m
                and not standoff_violation
                and not tracker.collided_ever
            )
            rec = EpisodeRecord(
                episode=ep,
                target_source=source,
                success=success,
                declared_arrival=declared,
                true_dist_at_arrival_m=true_dist_at_arrival,
                min_true_dist_to_target_m=min_dist,
                standoff_violation=standoff_violation,
                collided=tracker.collided_ever,
                duration_s=(t_last - t0) if t0 is not None else 0.0,
                path_length_m=path_len,
                straight_dist_m=straight,
                extra={"target_confidence": fix.confidence},
            )
            log.info(
                f"episode {ep} ({source}): {'SUCCESS' if success else 'fail'} "
                f"(arrival={declared}, true_dist={true_dist_at_arrival}, min_dist={min_dist:.1f})"
            )
            return rec
        finally:
            await tracker.stop()
            await service.close()
            await harness.close()


def truth_heading_enu(anchor: GeoAnchor, vel_sim: np.ndarray) -> float:
    return heading_deg_from_enu_vector(anchor.sim_to_enu(vel_sim))
