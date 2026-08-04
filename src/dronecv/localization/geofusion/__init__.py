"""Geometry-first fusion localization (the "cascata cooperativa").

Localizes against the 3D model by SHAPE, never by color: RGB is not compared
at any stage, so real-vs-generated color differences and lighting/shadow
changes cannot break it. Four cooperating modules:

    M1  retrieval      invariant-render descriptors (depth + classes +
                       skyline) over a pose grid -> where-am-I shortlist
    M2  fine pose      render-and-compare silhouette IoU + depth agreement,
                       hill-climbed around each candidate
    M3  constellation  scale-free (ratio-based) matching of the OBSERVED
                       building layout against the model's footprints —
                       vetoes look-alike poses ("edifici troppo simili")
    M4  particle       sequence fusion with odometry: ambiguity that
                       survives one frame dies over a trajectory

`InvariantView` is the interchange format: anything able to produce a depth
map + building mask (the built-in renderer for simulation, or a monocular
depth + segmentation foundation model for real photos) can feed the same
pipeline unchanged.
"""

from dronecv.localization.geofusion.constellation import constellation_score, extract_buildings
from dronecv.localization.geofusion.finepose import refine_pose
from dronecv.localization.geofusion.invariant import InvariantView, descriptor
from dronecv.localization.geofusion.particle import ParticleFuser
from dronecv.localization.geofusion.pipeline import (
    FusionFix,
    GeoFusionConfig,
    GeoFusionLocalizer,
)
from dronecv.localization.geofusion.retrieval import GeoRetrievalIndex

__all__ = [
    "FusionFix",
    "GeoFusionConfig",
    "GeoFusionLocalizer",
    "GeoRetrievalIndex",
    "InvariantView",
    "ParticleFuser",
    "constellation_score",
    "descriptor",
    "extract_buildings",
    "refine_pose",
]
