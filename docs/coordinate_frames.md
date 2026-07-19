# Coordinate frames and conventions

## Frames

| frame | definition | used by |
|---|---|---|
| **sim** | Unity local: left-handed, X right, Y up, Z forward | simulators, wire protocol |
| **ENU** | right-handed East/North/Up (m), origin at the sim origin | all internal navigation state (EKF, guidance) |
| **WGS84** | geodetic lat/lon (deg), MSL altitude (m) | API boundary only (estimates, targets) |
| **body RFU** | X right, Y forward, Z up | heading definition |
| **body-level** | camera rays de-tilted by the mounting pitch | VO, sun cue |

## The anchor

`GeoAnchor {lat0, lon0, alt0, true_north_offset_deg, source}` ties sim to
world. `true_north_offset_deg` = compass bearing of the sim +Z axis.
Resolution order: embedded sim metadata (`hello_ack.geo_meta`) → env YAML
(`env.anchor`). Provenance is stamped in the model bundle manifest and
enforced at load (`Localizer._check_anchor`).

## Conversions (`src/dronecv/geo/frames.py`)

- Axis map (offset 0): `E = x, N = z, U = y`; then a yaw rotation by the
  north offset. `sim_to_enu` / `enu_to_sim` are exact inverses.
- Rotations: `R_enu = Yaw(offset) · M · R_sim · Mᵀ` with the improper axis
  map M — a proper rotation mapping body RFU → ENU (the same M maps Unity's
  body frame to RFU, which is why it composes cleanly).
- Heading: `heading = atan2(E, N)` in degrees clockwise from true north, of
  the body forward (+Y in RFU). For pure Unity yaw:
  `heading = unity_yaw + true_north_offset` (mirrored in C#).
- ENU ↔ WGS84 via pymap3d (`enu2geodetic` / `geodetic2enu`).
- Angles wrap with `wrap_deg` → (-180, 180].

## Camera

Pinhole; config FOV is **horizontal** (the Unity bridge converts to Unity's
vertical `fieldOfView`). Mounting: yaw about world up, then a fixed downward
tilt (`sim.camera_tilt_deg`, positive = down). Pixel (0,0) is top-left.
`capture_request.pitch_deg` is positive-down; `-20` looks 20° above the
horizon.

## Cross-language guarantees

`tests/fixtures/frames_cases.json` (positions + quaternions + headings, with
north offsets) and `tests/fixtures/celestial_cases.json` (sun az/el vectors)
are asserted by BOTH pytest (`tests/unit/test_frames.py`,
`test_celestial.py`) and Unity EditMode tests (`FramesConversionTests.cs`,
`CelestialTests.cs`). If you change a convention, both suites fail together.
