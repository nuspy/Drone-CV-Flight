"""Visual place recognition: embedding network + landmark database.

EmbeddingNet maps an image to an L2-normalized descriptor trained with a
batch-hard triplet loss (positives = captures within pos_radius_m AND a yaw
window — same place seen from a similar direction). The LandmarkDB stores
descriptors + poses of the training captures ("recognizable POVs"); a query
returns a position fix by similarity-weighted consensus over the top-k
neighbors, with covariance from the neighbor spread — wide spread = ambiguous
place = weak measurement, which is exactly what the EKF wants to know.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dronecv.models.backbone import TinyConvNet


class EmbeddingNet(nn.Module):
    def __init__(self, width: int = 32, embedding_dim: int = 128):
        super().__init__()
        self.backbone = TinyConvNet(width)
        self.head = nn.Linear(self.backbone.out_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.backbone(x)), dim=-1)


def batch_hard_triplet_loss(
    embeddings: torch.Tensor, pos_mask: torch.Tensor, margin: float
) -> torch.Tensor:
    """Batch-hard triplet loss with a boolean positive mask.

    pos_mask[i, j] = True if j is a positive for i (same place, similar yaw).
    Diagonal must be False. Pairs with no positive/negative are skipped.
    """
    dist = torch.cdist(embeddings, embeddings)
    inf = torch.full_like(dist, float("inf"))
    hardest_pos = torch.where(pos_mask, dist, -inf).max(dim=1).values
    neg_mask = ~pos_mask & ~torch.eye(len(dist), dtype=torch.bool, device=dist.device)
    hardest_neg = torch.where(neg_mask, dist, inf).min(dim=1).values
    valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
    if not valid.any():
        return embeddings.sum() * 0.0
    return F.relu(hardest_pos[valid] - hardest_neg[valid] + margin).mean()


def positive_mask(
    pos_enu: torch.Tensor, heading_deg: torch.Tensor, radius_m: float, yaw_window_deg: float
) -> torch.Tensor:
    d = torch.cdist(pos_enu[:, :2], pos_enu[:, :2])
    dyaw = (heading_deg[:, None] - heading_deg[None, :] + 180.0) % 360.0 - 180.0
    mask = (d < radius_m) & (dyaw.abs() < yaw_window_deg)
    mask.fill_diagonal_(False)
    return mask


class LandmarkDB:
    """Descriptor index over the curated capture set. Brute-force numpy at v1
    sizes (<= tens of thousands); swap for FAISS behind the same interface if
    a real environment ever needs more."""

    def __init__(self, embeddings: np.ndarray, pos_enu: np.ndarray, heading_deg: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)  # (N, D), L2-normalized
        self.pos_enu = pos_enu.astype(np.float32)
        self.heading_deg = heading_deg.astype(np.float32)

    def __len__(self) -> int:
        return len(self.embeddings)

    def query(self, descriptor: np.ndarray, topk: int = 5) -> dict:
        """Similarity-weighted consensus fix from the top-k neighbors.

        Neighbors much less similar than the best match are dropped before
        the consensus: with an untrained or weak embedding all similarities
        cluster together and a plain softmax would average unrelated places.
        """
        sims = self.embeddings @ descriptor.astype(np.float32)
        k = min(topk, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        keep = sims[idx] > sims[idx[0]] - 0.08
        idx = idx[keep]
        top_sims = sims[idx]
        weights = np.exp((top_sims - top_sims.max()) / 0.02)
        weights /= weights.sum()
        positions = self.pos_enu[idx]
        mean = weights @ positions
        spread = positions - mean
        cov = (weights[:, None, None] * (spread[:, :, None] @ spread[:, None, :])).sum(axis=0)
        # Circular weighted mean of neighbor headings.
        h = np.radians(self.heading_deg[idx])
        heading = float(np.degrees(np.arctan2(weights @ np.sin(h), weights @ np.cos(h))) % 360.0)
        return {
            "pos_enu": mean,
            "cov": cov + np.eye(3) * 0.5,  # floor
            "heading_deg": heading,
            "similarity": float(top_sims[0]),
            "indices": idx,
            "spread_m": float(np.sqrt(max(np.trace(cov), 0.0))),
        }

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, embeddings=self.embeddings, pos_enu=self.pos_enu, heading_deg=self.heading_deg
        )

    @classmethod
    def load(cls, path: Path) -> LandmarkDB:
        data = np.load(path)
        return cls(data["embeddings"], data["pos_enu"], data["heading_deg"])


class PositivePairBatchSampler:
    """Yields batches made of anchor+positive PAIRS so the batch-hard triplet
    loss always has work to do. With plain random batches over a large world,
    two views of the same place almost never co-occur and the loss silently
    stays zero (observed: embeddings never trained)."""

    def __init__(
        self,
        pos_enu: np.ndarray,
        heading_deg: np.ndarray,
        radius_m: float,
        yaw_window_deg: float,
        batch_size: int,
        seed: int = 0,
    ):
        n = len(pos_enu)
        d = np.linalg.norm(pos_enu[:, None, :2] - pos_enu[None, :, :2], axis=-1)
        dyaw = np.abs((heading_deg[:, None] - heading_deg[None, :] + 180.0) % 360.0 - 180.0)
        mask = (d < radius_m) & (dyaw < yaw_window_deg)
        np.fill_diagonal(mask, False)
        self.positives = [np.flatnonzero(mask[i]) for i in range(n)]
        self.anchors = [i for i in range(n) if len(self.positives[i])]
        self.pairs_per_batch = max(2, batch_size // 2)
        self.gen = np.random.default_rng(seed)

    def __iter__(self):
        order = self.gen.permutation(self.anchors)
        for start in range(0, len(order) - 1, self.pairs_per_batch):
            chunk = order[start : start + self.pairs_per_batch]
            batch: list[int] = []
            for a in chunk:
                batch.append(int(a))
                batch.append(int(self.gen.choice(self.positives[a])))
            if len(batch) >= 4:
                yield batch

    def __len__(self):
        return max(1, len(self.anchors) // self.pairs_per_batch)


@torch.no_grad()
def embed_images(net: EmbeddingNet, images: torch.Tensor, batch: int = 64) -> np.ndarray:
    net.eval()
    outs = [net(images[i : i + batch]).cpu().numpy() for i in range(0, len(images), batch)]
    return np.concatenate(outs) if outs else np.zeros((0, net.head.out_features), dtype=np.float32)
