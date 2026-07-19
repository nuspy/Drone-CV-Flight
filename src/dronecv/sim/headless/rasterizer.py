"""Numpy ray-casting renderer: RGB + exact per-pixel range, no GPU, no EGL.

Terrain is ray-marched against the heightmap (with bisection refinement);
landmarks are intersected analytically (AABB slabs / vertical cylinder).
Shading is Lambert from the real sun direction (dronecv.geo.celestial), plus
an ambient term; the sun and moon are drawn as bright discs so the celestial
heading cue can be exercised end-to-end.

Everything is vectorized across all rays; at 96x96 a frame costs tens of
milliseconds on CPU, which keeps CI end-to-end runs fast.
"""

from __future__ import annotations

import math

import numpy as np

from dronecv.sim.headless.world import World

SKY_HORIZON = np.array([0.62, 0.74, 0.88], dtype=np.float32)
SKY_ZENITH = np.array([0.15, 0.32, 0.62], dtype=np.float32)
SUN_ANGULAR_RADIUS_DEG = 2.5  # oversized for detectability at small resolutions
MOON_ANGULAR_RADIUS_DEG = 2.0
NO_HIT = 1.0e9


def ray_directions(width: int, height: int, fov_deg: float, yaw_deg: float, pitch_down_deg: float) -> np.ndarray:
    """Unit ray directions in the sim world frame, shape (H, W, 3).

    Camera convention: looking along sim +Z when yaw=0, pitch rotates the view
    down toward the ground (positive = down). Pixel (0,0) is top-left.
    """
    tan_h = math.tan(math.radians(fov_deg) / 2.0)
    tan_v = tan_h * height / width
    us = np.linspace(-tan_h, tan_h, width)
    vs = np.linspace(tan_v, -tan_v, height)
    uu, vv = np.meshgrid(us, vs)
    d = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    pitch = math.radians(pitch_down_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rot_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])  # pitch down
    yaw = math.radians(yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Unity-style yaw about +Y: positive yaw turns +Z toward +X.
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return d @ rot_x.T @ rot_y.T


def _march_terrain(world: World, origin: np.ndarray, dirs: np.ndarray, max_range: float) -> np.ndarray:
    """First terrain intersection distance per ray (NO_HIT where none)."""
    flat = dirs.reshape(-1, 3)
    n = flat.shape[0]
    t_hit = np.full(n, NO_HIT, dtype=np.float64)
    active = np.ones(n, dtype=bool)

    # Rays starting above the highest terrain and pointing up can never hit.
    max_h = float(world.heightmap.max())
    if origin[1] > max_h:
        active &= flat[:, 1] < 0

    # Geometric step schedule: fine near the camera, coarse far away. A ray
    # still active at step k was above the terrain at step k-1, so the first
    # step that finds it below brackets the crossing in (prev_t, t].
    steps = np.geomspace(1.0, max_range, 112)
    prev_t = np.zeros(n)
    for t in steps:
        if not active.any():
            break
        p = origin[None, :] + flat[active] * t
        inside = (np.abs(p[:, 0]) < world.half) & (np.abs(p[:, 2]) < world.half)
        h = world.height_at(p[:, 0], p[:, 2])
        below = (p[:, 1] < h) & inside
        idx = np.where(active)[0]
        crossed = idx[below]
        if crossed.size:
            # Bisection refine between prev_t and t.
            lo = prev_t[crossed].copy()
            hi = np.full(crossed.size, t)
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                pm = origin[None, :] + flat[crossed] * mid[:, None]
                below_m = pm[:, 1] < world.height_at(pm[:, 0], pm[:, 2])
                hi = np.where(below_m, mid, hi)
                lo = np.where(below_m, lo, mid)
            t_hit[crossed] = hi
            active[crossed] = False
        # Rays that left the world heading up and out can be retired early.
        idx_a = np.where(active)[0]
        if idx_a.size:
            pa = origin[None, :] + flat[idx_a] * t
            gone = (pa[:, 1] > world.bounds_max[1]) & (flat[idx_a][:, 1] > 0)
            active[idx_a[gone]] = False
        prev_t[active] = t
    return t_hit.reshape(dirs.shape[:2])


def _intersect_landmarks(
    world: World, origin: np.ndarray, dirs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest landmark hit: (t, color_index, normal). t=NO_HIT where none.

    color_index packs (landmark_index, stripe_parity, face_shade) so the
    shader can compute the final color without another pass.
    """
    flat = dirs.reshape(-1, 3)
    n = flat.shape[0]
    best_t = np.full(n, NO_HIT)
    best_lm = np.full(n, -1, dtype=int)
    best_normal = np.zeros((n, 3))

    ox, oy, oz = origin
    dx, dy, dz = flat[:, 0], flat[:, 1], flat[:, 2]

    for li, lm in enumerate(world.landmarks):
        y0, y1 = lm.base_y - 2.0, lm.base_y + lm.height
        if lm.kind == "box":
            with np.errstate(divide="ignore", invalid="ignore"):
                tx1 = (lm.x - lm.half_x - ox) / dx
                tx2 = (lm.x + lm.half_x - ox) / dx
                ty1 = (y0 - oy) / dy
                ty2 = (y1 - oy) / dy
                tz1 = (lm.z - lm.half_z - oz) / dz
                tz2 = (lm.z + lm.half_z - oz) / dz
            tmin = np.maximum.reduce([np.minimum(tx1, tx2), np.minimum(ty1, ty2), np.minimum(tz1, tz2)])
            tmax = np.minimum.reduce([np.maximum(tx1, tx2), np.maximum(ty1, ty2), np.maximum(tz1, tz2)])
            hit = (tmax > np.maximum(tmin, 1e-3)) & (tmin < best_t)
            if not hit.any():
                continue
            t = tmin[hit]
            # Entering face normal: which slab produced tmin.
            nx = np.where(np.isclose(t, np.minimum(tx1, tx2)[hit]), -np.sign(dx[hit]), 0.0)
            ny = np.where(np.isclose(t, np.minimum(ty1, ty2)[hit]), -np.sign(dy[hit]), 0.0)
            nz = np.where(np.isclose(t, np.minimum(tz1, tz2)[hit]), -np.sign(dz[hit]), 0.0)
            normal = np.stack([nx, ny, nz], axis=-1)
            norm = np.linalg.norm(normal, axis=-1, keepdims=True)
            normal = np.where(norm > 0, normal / np.maximum(norm, 1e-9), np.array([0.0, 1.0, 0.0]))
        else:  # vertical cylinder
            cx, cz, r = lm.x, lm.z, lm.half_x
            fx, fz = ox - cx, oz - cz
            a = dx * dx + dz * dz
            b = 2 * (fx * dx + fz * dz)
            c = fx * fx + fz * fz - r * r
            disc = b * b - 4 * a * c
            with np.errstate(invalid="ignore", divide="ignore"):
                sq = np.sqrt(np.maximum(disc, 0.0))
                t_side = np.where(disc > 0, (-b - sq) / (2 * a), NO_HIT)
            y_at = oy + dy * t_side
            side_ok = (t_side > 1e-3) & (y_at > y0) & (y_at < y1)
            # Top cap.
            with np.errstate(divide="ignore", invalid="ignore"):
                t_cap = (y1 - oy) / dy
            px, pz = ox + dx * t_cap - cx, oz + dz * t_cap - cz
            cap_ok = (t_cap > 1e-3) & (px * px + pz * pz < r * r)
            t_all = np.where(side_ok, t_side, NO_HIT)
            t_all = np.where(cap_ok & (t_cap < t_all), t_cap, t_all)
            hit = (t_all < NO_HIT) & (t_all < best_t)
            if not hit.any():
                continue
            t = t_all[hit]
            is_cap = cap_ok[hit] & np.isclose(t, t_cap[hit])
            hx = ox + dx[hit] * t - cx
            hz = oz + dz[hit] * t - cz
            rad = np.stack([hx, np.zeros_like(hx), hz], axis=-1)
            rad /= np.maximum(np.linalg.norm(rad, axis=-1, keepdims=True), 1e-9)
            normal = np.where(is_cap[:, None], np.array([0.0, 1.0, 0.0]), rad)

        best_t[hit] = t
        best_lm[hit] = li
        best_normal[hit] = normal

    return best_t, best_lm, best_normal


def render(
    world: World,
    pos_sim: np.ndarray,
    yaw_deg: float,
    pitch_down_deg: float,
    width: int,
    height: int,
    fov_deg: float,
    sun_azimuth_sim_deg: float | None,
    sun_elevation_deg: float | None,
    moon_azimuth_sim_deg: float | None = None,
    moon_elevation_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render RGB (uint8 HxWx3) and range image (float32 HxW, meters).

    Celestial azimuths are in the SIM frame (already corrected for the
    anchor's true-north offset by the caller) so the renderer stays anchor
    agnostic.
    """
    origin = np.asarray(pos_sim, dtype=np.float64)
    dirs = ray_directions(width, height, fov_deg, yaw_deg, pitch_down_deg)
    max_range = world.size * 1.6

    t_terrain = _march_terrain(world, origin, dirs, max_range)
    t_lm, lm_idx, lm_normal = _intersect_landmarks(world, origin, dirs)
    t_lm = t_lm.reshape(height, width)
    lm_idx = lm_idx.reshape(height, width)
    lm_normal = lm_normal.reshape(height, width, 3)

    use_lm = t_lm < t_terrain
    t_final = np.where(use_lm, t_lm, t_terrain).astype(np.float32)
    hit = t_final < NO_HIT / 2

    # Sun direction in sim frame (for shading and the sky disc).
    def _dir(az_deg: float, el_deg: float) -> np.ndarray:
        az, el = math.radians(az_deg), math.radians(el_deg)
        # Sim azimuth: 0 = +Z, 90 = +X (mirrors compass with sim Z=north).
        return np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])

    sun_up = sun_elevation_deg is not None and sun_elevation_deg > 0.0
    sun_dir = _dir(sun_azimuth_sim_deg, sun_elevation_deg) if sun_up else np.array([0.0, 1.0, 0.0])
    ambient = 0.35 if sun_up else 0.12
    sun_strength = max(0.15, math.sin(math.radians(sun_elevation_deg))) if sun_up else 0.0

    rgb = np.zeros((height, width, 3), dtype=np.float32)

    # --- sky ---
    sky_mask = ~hit
    if sky_mask.any():
        elev = np.clip(dirs[..., 1], 0.0, 1.0)[sky_mask, None]
        sky = SKY_HORIZON[None, :] * (1 - elev) + SKY_ZENITH[None, :] * elev
        rgb[sky_mask] = sky
        cos_sun = math.cos(math.radians(SUN_ANGULAR_RADIUS_DEG))
        if sun_up:
            disc = (dirs[sky_mask] @ sun_dir) > cos_sun
            sky[disc] = np.array([1.0, 0.97, 0.85])
            rgb[sky_mask] = sky
        if moon_elevation_deg is not None and moon_elevation_deg > 0.0:
            moon_dir = _dir(moon_azimuth_sim_deg, moon_elevation_deg)
            disc = (dirs[sky_mask] @ moon_dir) > math.cos(math.radians(MOON_ANGULAR_RADIUS_DEG))
            sky[disc] = np.array([0.92, 0.92, 0.88])
            rgb[sky_mask] = sky

    # --- terrain ---
    terr_mask = hit & ~use_lm
    if terr_mask.any():
        p = origin[None, :] + dirs[terr_mask] * t_terrain[terr_mask][:, None]
        albedo = world.albedo_at(p[:, 0], p[:, 2])
        normal = world.normal_at(p[:, 0], p[:, 2])
        lambert = np.clip(normal @ sun_dir, 0.0, 1.0)[:, None] * sun_strength
        shade = albedo * (ambient + (1 - ambient) * lambert)
        # Distance haze toward the horizon color.
        haze = np.clip(t_terrain[terr_mask] / max_range, 0, 1)[:, None] ** 1.5
        rgb[terr_mask] = shade * (1 - haze) + SKY_HORIZON[None, :] * haze

    # --- landmarks ---
    lm_mask = hit & use_lm
    if lm_mask.any():
        p = origin[None, :] + dirs[lm_mask] * t_lm[lm_mask][:, None]
        idx = lm_idx[lm_mask]
        colors_a = np.stack([world.landmarks[i].color_a for i in range(len(world.landmarks))])
        colors_b = np.stack([world.landmarks[i].color_b for i in range(len(world.landmarks))])
        stripes = np.array([lm.stripe_m for lm in world.landmarks])
        bases = np.array([lm.base_y for lm in world.landmarks])
        stripe_par = (np.floor((p[:, 1] - bases[idx]) / stripes[idx]) % 2).astype(bool)
        albedo = np.where(stripe_par[:, None], colors_b[idx], colors_a[idx])
        # Slight face-dependent tint so yaw is visually observable up close.
        east_face = lm_normal[lm_mask][:, 0] > 0.5
        albedo = np.where(east_face[:, None], albedo * 0.75, albedo)
        lambert = np.clip(lm_normal[lm_mask] @ sun_dir, 0.0, 1.0)[:, None] * sun_strength
        rgb[lm_mask] = albedo * (ambient + (1 - ambient) * lambert)

    depth = np.where(hit, t_final, np.float32(0.0)).astype(np.float32)  # 0 = no hit
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), depth
