"""Optional monocular-depth plug-in for real photos.

`get_depth_fn()` returns a callable `(rgb uint8 HxWx3) -> float32 HxW` of
RELATIVE depth (unknown scale — that is what monocular models give), or None
when no backend is available. Detection order:

  1. `transformers` with Depth-Anything-V2 small (downloads from the
     HuggingFace hub on first use — works on a normal network; corporate
     proxies that block the hub simply leave the plug-in off);
  2. none -> the photo pipeline runs mask-only (documented baseline).

Relative depth is used by the ALIGNMENT stage only (median-normalized on
both sides, so the unknown scale cancels); retrieval descriptors keep their
depth bins empty because a relative histogram would not be comparable with
the metric index.
"""

from __future__ import annotations

from dronecv.util.logging import get_logger

log = get_logger("dronecv.geofusion.depth")

_cached = "unset"


def get_depth_fn():
    """A relative-depth callable or None. Result is cached per process."""
    global _cached
    if _cached != "unset":
        return _cached
    _cached = _try_depth_anything()
    return _cached


def _try_depth_anything():
    try:
        import numpy as np
        from transformers import pipeline as hf_pipeline

        pipe = hf_pipeline("depth-estimation",
                           model="depth-anything/Depth-Anything-V2-Small-hf")

        def fn(rgb: np.ndarray) -> np.ndarray:
            from PIL import Image

            out = pipe(Image.fromarray(rgb))
            depth = np.asarray(out["predicted_depth"])
            # model outputs inverse-ish relative depth; larger = closer.
            # Convert to "larger = farther" to match range images.
            d = depth.max() - depth + 1e-3
            return d.astype(np.float32)

        log.info("depth plug-in: Depth-Anything-V2-Small active")
        return fn
    except Exception as e:  # noqa: BLE001 — no transformers / hub unreachable
        log.info(f"depth plug-in unavailable ({type(e).__name__}); "
                 "photo pipeline runs mask-only")
        return None
