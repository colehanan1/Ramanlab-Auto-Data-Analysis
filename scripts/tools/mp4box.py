"""Minimal ISO-BMFF reader for single-video-track MP4s (the rigs' libx264 remuxes)."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf"}


def iter_boxes(buf: bytes, start: int, end: int):
    """Yield (type, box_start, payload_start, box_end) for the boxes in buf[start:end]."""
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[pos:pos + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[pos + 8:pos + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr or pos + size > end:
            yield typ, pos, pos + hdr, None  # truncated / invalid
            return
        yield typ, pos, pos + hdr, pos + size
        pos += size


def top_boxes(buf: bytes):
    return [(t.decode("latin1"), s, e) for t, s, _, e in iter_boxes(buf, 0, len(buf))]


@dataclass
class Track:
    timescale: int = 0
    duration: int = 0
    sizes: list = field(default_factory=list)
    chunk_offsets: list = field(default_factory=list)
    stsc: list = field(default_factory=list)       # (first_chunk, samples_per_chunk, desc)
    stss: list = field(default_factory=list)       # 1-based sync sample numbers
    stts: list = field(default_factory=list)       # (count, delta)
    ctts: list = field(default_factory=list)       # (count, offset)
    sps: list = field(default_factory=list)
    pps: list = field(default_factory=list)
    length_size: int = 4
    width: int = 0
    height: int = 0

    def sample_offsets(self) -> list[int]:
        """File offset of every sample, in decode order."""
        out = []
        runs = self.stsc + [(len(self.chunk_offsets) + 1, 0, 0)]
        s = 0
        for i in range(len(runs) - 1):
            first, per, _ = runs[i]
            nxt = runs[i + 1][0]
            for chunk in range(first, nxt):
                if chunk > len(self.chunk_offsets):
                    break
                off = self.chunk_offsets[chunk - 1]
                for _ in range(per):
                    if s >= len(self.sizes):
                        break
                    out.append(off)
                    off += self.sizes[s]
                    s += 1
        return out


def _full(payload: bytes):
    return payload[0], payload[4:]


def parse_moov(buf: bytes, moov_start: int, moov_end: int) -> Track:
    """Parse the first video trak of the moov box at buf[moov_start:moov_end]."""
    tr = Track()

    def walk(start, end):
        for typ, _s, p, e in iter_boxes(buf, start, end):
            if e is None:
                return
            if typ in CONTAINERS:
                walk(p, e)
            elif typ == b"mdhd":
                ver = buf[p]
                if ver == 1:
                    tr.timescale, tr.duration = struct.unpack(">IQ", buf[p + 20:p + 32])
                else:
                    tr.timescale, tr.duration = struct.unpack(">II", buf[p + 12:p + 20])
            elif typ == b"stsd":
                # stsd: v/f(4) count(4) then sample entries
                for etyp, _es, ep, ee in iter_boxes(buf, p + 8, e):
                    if etyp in (b"avc1", b"avc3"):
                        tr.width, tr.height = struct.unpack(">HH", buf[ep + 24:ep + 28])
                        for ctyp, _cs, cp, ce in iter_boxes(buf, ep + 78, ee):
                            if ctyp == b"avcC":
                                c = buf[cp:ce]
                                tr.length_size = (c[4] & 3) + 1
                                n_sps = c[5] & 0x1F
                                q = 6
                                for _ in range(n_sps):
                                    ln = struct.unpack(">H", c[q:q + 2])[0]
                                    tr.sps.append(c[q + 2:q + 2 + ln])
                                    q += 2 + ln
                                n_pps = c[q]
                                q += 1
                                for _ in range(n_pps):
                                    ln = struct.unpack(">H", c[q:q + 2])[0]
                                    tr.pps.append(c[q + 2:q + 2 + ln])
                                    q += 2 + ln
            elif typ == b"stsz":
                size, count = struct.unpack(">II", buf[p + 4:p + 12])
                if size:
                    tr.sizes = [size] * count
                else:
                    tr.sizes = list(struct.unpack(f">{count}I", buf[p + 12:p + 12 + 4 * count]))
            elif typ == b"stco":
                n = struct.unpack(">I", buf[p + 4:p + 8])[0]
                tr.chunk_offsets = list(struct.unpack(f">{n}I", buf[p + 8:p + 8 + 4 * n]))
            elif typ == b"co64":
                n = struct.unpack(">I", buf[p + 4:p + 8])[0]
                tr.chunk_offsets = list(struct.unpack(f">{n}Q", buf[p + 8:p + 8 + 8 * n]))
            elif typ == b"stsc":
                n = struct.unpack(">I", buf[p + 4:p + 8])[0]
                v = struct.unpack(f">{3 * n}I", buf[p + 8:p + 8 + 12 * n])
                tr.stsc = [tuple(v[i:i + 3]) for i in range(0, len(v), 3)]
            elif typ == b"stss":
                n = struct.unpack(">I", buf[p + 4:p + 8])[0]
                tr.stss = list(struct.unpack(f">{n}I", buf[p + 8:p + 8 + 4 * n]))
            elif typ == b"stts":
                n = struct.unpack(">I", buf[p + 4:p + 8])[0]
                v = struct.unpack(f">{2 * n}I", buf[p + 8:p + 8 + 8 * n])
                tr.stts = [tuple(v[i:i + 2]) for i in range(0, len(v), 2)]
            elif typ == b"ctts":
                n = struct.unpack(">I", buf[p + 4:p + 8])[0]
                v = struct.unpack(f">{2 * n}i", buf[p + 8:p + 8 + 8 * n])
                tr.ctts = [tuple(v[i:i + 2]) for i in range(0, len(v), 2)]

    walk(moov_start + 8, moov_end)
    return tr


PLAUSIBLE_NAL = {1, 5, 6, 7, 8, 9, 12}


def nal_valid(buf: bytes, off: int, size: int, length_size: int = 4) -> bool:
    """True if buf[off:off+size] parses exactly as a run of length-prefixed H.264 NAL units."""
    end = off + size
    if off < 0 or size <= 0 or end > len(buf):
        return False
    p = off
    while p < end:
        if p + length_size + 1 > end:
            return False
        ln = int.from_bytes(buf[p:p + length_size], "big")
        if ln == 0 or p + length_size + ln > end:
            return False
        hdr = buf[p + length_size]
        if hdr & 0x80 or (hdr & 0x1F) not in PLAUSIBLE_NAL:
            return False
        p += length_size + ln
    return p == end
