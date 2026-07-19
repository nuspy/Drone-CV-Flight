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


async def connect(host: str, port: int, timeout: float = 10.0) -> Connection:
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    return Connection(reader, writer)
