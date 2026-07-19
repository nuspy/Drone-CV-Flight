"""Versioned wire messages shared by the Unity bridge and the headless sim.

Every message is a JSON header, optionally followed by binary blobs described
by `blobs`. The Python side cannot tell Unity and the headless simulator
apart: both speak exactly this schema. docs/protocol.md documents the framing;
tests/unit/test_protocol.py pins golden bytes that the C# MessageCodec is
validated against (fixture files under tests/fixtures/protocol/).

Channels: sensors | control | capture | truth. The `truth` channel carries
ground truth and may only be subscribed by clients that declared
role="harness" in their hello — this is how the test system guarantees the
localizer never sees the answer.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

PROTOCOL_VERSION = "1.0"

Role = Literal["localizer", "harness", "capture", "verify"]
Channel = Literal["sensors", "control", "capture", "truth"]


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BlobSpec(Base):
    """Describes one binary blob following the JSON header, in order."""

    name: str
    dtype: Literal["uint8", "float32", "bytes"] = "bytes"
    shape: list[int] | None = None  # None for encoded streams (jpeg/png)
    encoding: Literal["raw", "jpeg", "png"] = "raw"
    byte_len: int


class Header(Base):
    """Fields common to every message."""

    version: str = PROTOCOL_VERSION
    msg_id: int = 0
    sim_time: float = 0.0
    blobs: list[BlobSpec] = Field(default_factory=list)


class Hello(Header):
    type: Literal["hello"] = "hello"
    role: Role = "localizer"


class CameraInfo(Base):
    width: int
    height: int
    fov_deg: float
    tilt_deg: float


class HelloAck(Header):
    type: Literal["hello_ack"] = "hello_ack"
    sim_kind: Literal["unity", "headless"]
    env_id: str
    capabilities: list[str] = Field(default_factory=list)
    geo_meta: dict[str, float] | None = None  # embedded environment geo metadata
    camera: CameraInfo | None = None


class EnvInfoRequest(Header):
    type: Literal["env_info_request"] = "env_info_request"


class EnvInfo(Header):
    type: Literal["env_info"] = "env_info"
    bounds_min_sim: list[float]  # flyable volume, sim frame (x, y, z)
    bounds_max_sim: list[float]
    ground_alt_min_m: float = 0.0
    ground_alt_max_m: float = 0.0


class Subscribe(Header):
    type: Literal["subscribe"] = "subscribe"
    channels: list[Channel]


class SubscribeAck(Header):
    type: Literal["subscribe_ack"] = "subscribe_ack"
    channels: list[Channel]


class CaptureRequest(Header):
    type: Literal["capture_request"] = "capture_request"
    pos_sim: list[float]
    yaw_deg: float  # sim-frame yaw (rotation about sim Y, 0 = +Z)
    pitch_deg: float = 0.0  # camera pitch below horizon (positive = down)
    want: list[Literal["rgb", "depth", "sun"]] = Field(default_factory=lambda: ["rgb"])


class CaptureResult(Header):
    type: Literal["capture_result"] = "capture_result"
    pos_sim: list[float]  # ground-truth pose actually rendered
    quat_sim: list[float]  # (x, y, z, w) body->sim-world
    yaw_deg: float
    pitch_deg: float
    utc: str
    sun_azimuth_deg: float | None = None
    sun_elevation_deg: float | None = None
    moon_azimuth_deg: float | None = None
    moon_elevation_deg: float | None = None
    # blobs: "rgb" (jpeg/png), optional "depth" (float32 HxW)


class SensorFrame(Header):
    type: Literal["sensor_frame"] = "sensor_frame"
    frame_id: int
    utc: str
    lidar_range_m: float | None = None  # None = dropout / out of range
    # blobs: "rgb". Deliberately NO pose fields: the localizer flies blind.


class Command(Header):
    type: Literal["command"] = "command"
    vel_sim: list[float] | None = None  # commanded velocity, sim world frame (m/s)
    yaw_rate_dps: float | None = None


class TruthState(Header):
    type: Literal["truth_state"] = "truth_state"
    frame_id: int
    pos_sim: list[float]
    quat_sim: list[float]
    vel_sim: list[float]
    collided: bool = False


class Reset(Header):
    type: Literal["reset"] = "reset"
    seed: int = 0
    start_pos_sim: list[float] | None = None
    start_yaw_deg: float = 0.0
    utc: str | None = None  # override sim clock start


class ResetDone(Header):
    type: Literal["reset_done"] = "reset_done"
    pos_sim: list[float]


class Teleport(Header):
    """Harness-only: kidnap the drone mid-flight (re-localization test)."""

    type: Literal["teleport"] = "teleport"
    pos_sim: list[float]
    yaw_deg: float = 0.0


class ErrorMsg(Header):
    type: Literal["error"] = "error"
    message: str


class Ping(Header):
    type: Literal["ping"] = "ping"


class Pong(Header):
    type: Literal["pong"] = "pong"


AnyMessage = Annotated[
    Hello | HelloAck | EnvInfoRequest | EnvInfo | Subscribe | SubscribeAck | CaptureRequest | CaptureResult | SensorFrame | Command | TruthState | Reset | ResetDone | Teleport | ErrorMsg | Ping | Pong,
    Field(discriminator="type"),
]

message_adapter: TypeAdapter[AnyMessage] = TypeAdapter(AnyMessage)


def parse_message(header: dict) -> AnyMessage:
    return message_adapter.validate_python(header)
