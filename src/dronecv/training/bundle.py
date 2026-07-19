"""Model bundle: everything the localizer needs, in one directory.

    model_bundle/
        manifest.json     anchor + provenance, model hyperparams, camera,
                          metrics, calibration factors, versions
        embed.pt          EmbeddingNet weights
        posenet.pt        PoseNet weights
        landmark_db.npz   descriptor index (embeddings, poses, headings)

The manifest pins the geo anchor the models were trained under; the localizer
refuses to run a bundle against a sim whose resolved anchor disagrees.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dronecv.geo.anchor import GeoAnchor
from dronecv.localization.terrain import TerrainPrior
from dronecv.models.pose_net import PoseNet
from dronecv.models.retrieval import EmbeddingNet, LandmarkDB

BUNDLE_VERSION = "1.0"


@dataclass
class ModelBundle:
    embed_net: EmbeddingNet
    pose_net: PoseNet
    landmark_db: LandmarkDB
    manifest: dict[str, Any]
    terrain: TerrainPrior

    @property
    def anchor(self) -> GeoAnchor:
        return GeoAnchor.from_dict(self.manifest["anchor"])

    @property
    def calibration_scale(self) -> float:
        return float(self.manifest.get("calibration", {}).get("apr_sigma_scale", 1.0))

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.embed_net.state_dict(), out_dir / "embed.pt")
        torch.save(self.pose_net.state_dict(), out_dir / "posenet.pt")
        self.landmark_db.save(out_dir / "landmark_db.npz")
        self.terrain.save(out_dir / "terrain_prior.npz")
        (out_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    @classmethod
    def load(cls, bundle_dir: Path) -> "ModelBundle":
        bundle_dir = Path(bundle_dir)
        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        hp = manifest["hyperparams"]
        embed = EmbeddingNet(hp["backbone_width"], hp["embedding_dim"])
        embed.load_state_dict(torch.load(bundle_dir / "embed.pt", weights_only=True))
        pose = PoseNet(hp["backbone_width"], hp["pos_scale_m"])
        pose.load_state_dict(torch.load(bundle_dir / "posenet.pt", weights_only=True))
        db = LandmarkDB.load(bundle_dir / "landmark_db.npz")
        terrain = TerrainPrior.load(bundle_dir / "terrain_prior.npz")
        embed.eval()
        pose.eval()
        return cls(embed, pose, db, manifest, terrain)
