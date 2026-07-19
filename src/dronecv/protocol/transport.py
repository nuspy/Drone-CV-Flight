"""Asyncio TCP transport for the dronecv protocol."""

from __future__ import annotations

import asyncio
import json
import struct

import numpy as np

from dronecv.protocol.framing import MAX_BLOB_LEN, MAX_HEADER_LEN, ProtocolError, decode_blob, encode_message
from dronecv.protocol.messages import AnyMessage, parse_message


class Connection:
    """One framed message stream over a TCP connection."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self._send_lock = asyncio.Lock()
        self._msg_id = 0

    @property
    def peer(self) -> str:
        info = self.writer.get_extra_info("peername")
        return f"{info[0]}:{info[1]}" if info else "?"

    async def send(
        self,
        msg: AnyMessage,
        blobs: dict[str, tuple[np.ndarray | bytes, str]] | None = None,
        jpeg_quality: int = 92,
    ) -> None:
        async with self._send_lock:
            self._msg_id += 1
            msg.msg_id = self._msg_id
            self.writer.write(encode_message(msg, blobs, jpeg_quality))
            await self.writer.drain()

    async def recv(self) -> tuple[AnyMessage, dict[str, np.ndarray | bytes]]:
        raw_len = await self.reader.readexactly(4)
        (header_len,) = struct.unpack(">I", raw_len)
        if header_len > MAX_HEADER_LEN:
            raise ProtocolError(f"header length {header_len} exceeds limit")
        header = json.loads((await self.reader.readexactly(header_len)).decode("utf-8"))
        msg = parse_message(header)
        blobs: dict[str, np.ndarray | bytes] = {}
        for spec in msg.blobs:
            if spec.byte_len > MAX_BLOB_LEN:
                raise ProtocolError(f"blob '{spec.name}' length {spec.byte_len} exceeds limit")
            blobs[spec.name] = decode_blob(spec, await self.reader.readexactly(spec.byte_len))
        return msg, blobs

    async def close(self) -> None:
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except (ConnectionError, OSError):
            pass


# Keep kernel socket buffers small: the faster-than-realtime sim must not run
# hundreds of frames ahead of a slow consumer. With ~32 KiB each way only a
# handful of jpeg frames fit in flight, so TCP backpressure paces the sim to
# within a few ticks of its slowest sensor subscriber.
SOCKET_BUF_BYTES = 32768


def tune_socket(writer: asyncio.StreamWriter) -> None:
    import socket as socket_mod

    sock = writer.get_extra_info("socket")
    if sock is not None:
        try:
            sock.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_SNDBUF, SOCKET_BUF_BYTES)
            sock.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_RCVBUF, SOCKET_BUF_BYTES)
            sock.setsockopt(socket_mod.IPPROTO_TCP, socket_mod.TCP_NODELAY, 1)
        except OSError:
            pass


async def connect(host: str, port: int, timeout: float = 10.0) -> Connection:
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host, port, limit=1 << 22), timeout
    )
    tune_socket(writer)
    return Connection(reader, writer)
