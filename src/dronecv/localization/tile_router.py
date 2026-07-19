"""TileRouter: localization over a hierarchically trained large area.

Coarse global place recognition answers "which tile", the tile's fine bundle
(standard Localizer) does the precise work. Tile switches re-initialize the
fine filter — measured re-localization after a hard jump is sub-second, so a
soft hand-off is not worth the state-mapping complexity in v1.
Exposes the same `step()` API as `Localizer`, so the service, harness and
CLI are untouched.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from dronecv.config import Config
from dronecv.geo.anchor import GeoAnchor
from dronecv.localization.localizer import Estimate, Localizer
from dronecv.models.retrieval import EmbeddingNet, LandmarkDB
from dronecv.training.bundle import ModelBundle
from dronecv.training.tiled import COARSE_DIM, COARSE_WIDTH, tile_of
from dronecv.util.logging import get_logger

log = get_logger("dronecv.tilerouter")


def is_tiled_bundle(bundle_dir: Path) -> bool:
    return (Path(bundle_dir) / "tiling.json").exists()


class TiledLocalizer:
    def __init__(self, cfg: Config, bundle_dir: Path, anchor: GeoAnchor | None = None, lru: int = 2):
        self.cfg = cfg
        self.root = Path(bundle_dir)
        tiling = json.loads((self.root / "tiling.json").read_text())
        self.tiles = tiling["tiles"]
        coarse_dir = self.root / "coarse"
        manifest = json.loads((coarse_dir / "manifest.json").read_text())
        self.coarse_net = EmbeddingNet(
            manifest.get("backbone_width", COARSE_WIDTH), manifest.get("embedding_dim", COARSE_DIM)
        )
        self.coarse_net.load_state_dict(torch.load(coarse_dir / "embed.pt", weights_only=True))
        self.coarse_net.eval()
        self.coarse_db = LandmarkDB.load(coarse_dir / "landmark_db.npz")
        self.anchor = anchor or GeoAnchor.from_dict(manifest["anchor"])
        self._lru: OrderedDict[str, Localizer] = OrderedDict()
        self._lru_max = lru
        self.current_tile: str | None = None

    def _coarse_fix(self, rgb: np.ndarray) -> np.ndarray:
        img = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            desc = self.coarse_net(img)[0].numpy()
        return np.asarray(self.coarse_db.query(desc, topk=5)["pos_enu"], dtype=float)

    def _localizer_for(self, tile_id: str) -> Localizer:
        if tile_id in self._lru:
            self._lru.move_to_end(tile_id)
            return self._lru[tile_id]
        bundle = ModelBundle.load(self.root / "tiles" / tile_id)
        loc = Localizer(self.cfg, bundle, self.anchor)
        self._lru[tile_id] = loc
        if len(self._lru) > self._lru_max:
            evicted, _ = self._lru.popitem(last=False)
            log.info(f"evicted tile bundle {evicted}")
        return loc

    def step(self, rgb: np.ndarray, lidar_range_m: float | None, utc: str, t: float) -> Estimate:
        # Route: trust the current fine localizer while it is healthy; consult
        # the coarse model when unset or lost, and at tile-exit.
        active = self._lru.get(self.current_tile) if self.current_tile else None
        need_route = active is None or active.fuser.lost
        if not need_route and active is not None and active.fuser.initialized:
            e, n = active.fuser.pos[0], active.fuser.pos[1]
            tile = tile_of(self.tiles, e, n)
            if tile is not None and tile["id"] != self.current_tile:
                need_route = True  # crossed the border per the fine estimate

        if need_route:
            fix = self._coarse_fix(rgb)
            tile = tile_of(self.tiles, float(fix[0]), float(fix[1]))
            if tile is None:
                # Outside every tile: clamp to the nearest by center distance.
                centers = [
                    ((t_["e_min"] + t_["e_max"]) / 2, (t_["n_min"] + t_["n_max"]) / 2, t_)
                    for t_ in self.tiles
                ]
                tile = min(centers, key=lambda c: (c[0] - fix[0]) ** 2 + (c[1] - fix[1]) ** 2)[2]
            if tile["id"] != self.current_tile:
                log.info(f"tile hand-off -> {tile['id']}")
                self.current_tile = tile["id"]

        return self._localizer_for(self.current_tile).step(rgb, lidar_range_m, utc, t)
