"""Regenerate the golden protocol fixtures (tests/fixtures/protocol/*.bin)
and the celestial cross-language vectors.

Run after any intentional protocol change (and bump PROTOCOL_VERSION):
    python scripts/gen_protocol_fixtures.py
The Unity EditMode tests read the same files, so commit the result.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dronecv.geo import celestial  # noqa: E402
from dronecv.protocol.framing import encode_message  # noqa: E402
from tests.unit.test_protocol import _all_sample_messages  # noqa: E402


def main() -> None:
    out = ROOT / "tests" / "fixtures" / "protocol"
    out.mkdir(parents=True, exist_ok=True)
    for name, (msg, blobs) in _all_sample_messages().items():
        (out / f"{name}.bin").write_bytes(encode_message(msg, blobs))
        print(f"wrote {name}.bin")

    cases = []
    for utc, lat, lon in [
        ("2026-06-21T10:00:00+00:00", 45.4642, 9.19),
        ("2026-06-21T16:30:00+00:00", 45.4642, 9.19),
        ("2026-03-20T12:00:00+00:00", 0.0, 0.0),
        ("2026-12-21T09:00:00+00:00", -33.8688, 151.2093),
        ("2026-09-23T06:00:00+00:00", 51.5074, -0.1278),
    ]:
        p = celestial.sun_position(datetime.fromisoformat(utc).astimezone(UTC), lat, lon)
        cases.append(
            {
                "utc": utc,
                "lat": lat,
                "lon": lon,
                "sun_azimuth_deg": round(p.azimuth_deg, 4),
                "sun_elevation_deg": round(p.elevation_deg, 4),
            }
        )
    fixture = ROOT / "tests" / "fixtures" / "celestial_cases.json"
    fixture.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"wrote {fixture.name}")

    # Mirror everything into the Unity package so its EditMode tests are
    # self-contained (FramesConversionTests / CelestialTests / CodecGoldenTests).
    import shutil

    unity_fixtures = ROOT / "unity" / "com.dronecv.flight" / "Tests" / "Fixtures"
    (unity_fixtures / "protocol").mkdir(parents=True, exist_ok=True)
    for src in [ROOT / "tests" / "fixtures" / "frames_cases.json", fixture]:
        shutil.copy2(src, unity_fixtures / src.name)
    for src in out.glob("*.bin"):
        shutil.copy2(src, unity_fixtures / "protocol" / src.name)
    print(f"mirrored fixtures into {unity_fixtures}")


if __name__ == "__main__":
    main()
