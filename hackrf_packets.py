"""Small GSMTAP/CCCH parser, inspired by Oros42/IMSI-catcher. CC0-1.0.

Each paging identity is a separate observation: identities in the same paging
message can refer to different subscribers and must not be linked together.
"""

import struct


def mobile_identity(data):
    if not data:
        return None
    kind = data[0] & 7
    if kind == 4 and len(data) == 5:
        return {"identity_type": "TMSI", "identity": data[1:].hex()}
    if kind != 1:
        return None
    digits = [data[0] >> 4]
    for byte in data[1:]:
        digits.extend((byte & 15, byte >> 4))
    if not data[0] & 8:
        if digits[-1] != 15:
            return None
        digits.pop()
    if not 5 <= len(digits) <= 15 or any(d > 9 for d in digits):
        return None
    return {"identity_type": "IMSI", "identity": "".join(map(str, digits))}


def _lv(data, offset):
    if offset >= len(data):
        return None, len(data)
    end = offset + 1 + data[offset]
    if end > len(data):
        return None, len(data)
    return mobile_identity(data[offset + 1:end]), end


def parse_packet(packet):
    """Return events; reject unsupported headers and truncated L2 messages."""
    if len(packet) < 16 or packet[0] != 2 or packet[2] != 1:
        return []
    header_len = packet[1] * 4
    # Only BCCH, CCCH, AGCH and PCH carry the pseudo-length framing below.
    if header_len < 16 or header_len >= len(packet) or packet[12] not in (1, 2, 5, 6):
        return []
    raw_arfcn = struct.unpack_from("!H", packet, 4)[0]
    if raw_arfcn & 0x4000:  # uplink has different framing
        return []
    payload = packet[header_len:]
    if not payload or payload[0] & 3 != 1:
        return []
    size = payload[0] >> 2
    if size < 2 or len(payload) < size + 1:
        return []
    data = payload[1:1 + size]
    if data[0] != 6:  # Radio Resource management
        return []
    common = {"arfcn": raw_arfcn & 0x3fff, "timeslot": packet[3],
              "frame": struct.unpack_from("!I", packet, 8)[0]}
    events = []
    message = data[1]
    if packet[12] == 1 and message == 0x1b and len(data) >= 9:
        a, b, c = data[4:7]
        mcc = [a & 15, a >> 4, b & 15]
        mnc = [c & 15, c >> 4]
        if b >> 4 != 15:
            mnc.append(b >> 4)
        if any(d > 9 for d in mcc + mnc):
            return []
        events.append({"event": "cell", "mcc": "".join(map(str, mcc)),
                       "mnc": "".join(map(str, mnc)),
                       "cell": int.from_bytes(data[2:4], "big"),
                       "lac": int.from_bytes(data[7:9], "big")})
    elif packet[12] != 1:
        identities = []
        if message == 0x21 and len(data) >= 4:  # Paging Request 1
            identity, end = _lv(data, 3)
            identities.append(identity)
            if end < len(data) and data[end] == 0x17:
                identity, _ = _lv(data, end + 1)
                identities.append(identity)
        elif message in (0x22, 0x24):  # Paging Request 2 or 3
            count = 2 if message == 0x22 else 4
            end = 3 + count * 4
            if len(data) < end:
                return []
            for offset in range(3, end, 4):
                identities.append({"identity_type": "TMSI",
                                   "identity": data[offset:offset + 4].hex()})
            if message == 0x22 and end < len(data) and data[end] == 0x17:
                identity, _ = _lv(data, end + 1)
                identities.append(identity)
        elif message == 0x3f and len(data) >= 10 and data[2] >> 4 == 0:
            channel, description, last = data[3:6]
            channel_type = channel >> 3
            subchannel = None
            if 8 <= channel_type <= 15:
                channel_name, subchannel = "SDCCH/8", channel_type - 8
            elif 4 <= channel_type <= 7:
                channel_name, subchannel = "SDCCH/4", channel_type - 4
            else:
                channel_name = "other"
            hopping = bool(description & 0x10)
            events.append({"event": "assignment", "channel": channel_name,
                           "subchannel": subchannel, "assigned_timeslot": channel & 7,
                           "hopping": hopping,
                           "assigned_arfcn": None if hopping else ((description & 3) << 8) | last})
        events.extend(dict(event="identity", **identity) for identity in identities if identity)
    return [dict(common, **event) for event in events]
