"""Parse Insta360 X6/X5/X4 binary trailer metadata.

Extracts:
- Factory MEI dual-fisheye optical calibration (offset_v3)
- 1000 Hz 6-DOF IMU stream (Record 0x03)
- Per-frame exposure timestamps and shutter durations (Records 0x04 and 0x0c)
- Rolling shutter readout time and IMU ranges from protobuf camera metadata (Record 0x01)
"""

from dataclasses import dataclass, field
import re
import struct
import numpy as np
from scipy.spatial.transform import Rotation

MAGIC = b"8db42d694ccc418790edff439fe026bf"


@dataclass
class MEILens:
    xi: float
    fx: float
    fy: float
    cx: float
    cy: float
    yaw: float
    pitch: float
    roll: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    s4: float = 0.0
    ref_w: float = 15488.0
    ref_h: float = 7744.0
    lens_type: int = 193
    R_ext: np.ndarray = field(default_factory=lambda: np.eye(3))
    lens_idx: int = 0


@dataclass
class Insta360ClipMeta:
    camera: str
    lenses: list[MEILens]
    imu_t: np.ndarray           # seconds from start of IMU (relative)
    imu_ts_abs: np.ndarray      # absolute hardware clock seconds (matches exp0_ts_abs)
    imu_accel: np.ndarray       # (N, 3) in g, camera frame (X=right, Y=down, Z=forward)
    imu_gyro: np.ndarray        # (N, 3) in rad/s, camera frame
    exp0_ts_abs: np.ndarray     # (F,) Lens 0 per-frame exposure center timestamp (s)
    exp0_shutter: np.ndarray    # (F,) Lens 0 per-frame shutter duration (s)
    exp1_ts_abs: np.ndarray     # (F,) Lens 1 per-frame exposure center timestamp (s)
    exp1_shutter: np.ndarray    # (F,) Lens 1 per-frame shutter duration (s)
    rolling_shutter_s: float    # Sensor vertical readout duration in seconds (e.g. 0.01536)
    first_frame_ts_abs: float   # From protobuf field 24 (seconds)
    raw_offset_str: str


def _read_trailer_records(path: str) -> dict[int, tuple[int, bytes]]:
    with open(path, "rb") as f:
        f.seek(0, 2)
        fsize = f.tell()
        f.seek(max(0, fsize - 72))
        foot = f.read(72)
        idx = foot.rfind(MAGIC)
        if idx < 8:
            raise ValueError(f"Insta360 magic trailer not found in {path}")
        extra_size, _version = struct.unpack("<II", foot[idx - 8 : idx])
        extra_start = fsize - extra_size
        f.seek(extra_start)
        blob = f.read(extra_size)

    pos = extra_size - 72
    _fmt, rid, rsize = struct.unpack("<BBI", blob[pos - 6 : pos])
    records: dict[int, tuple[int, bytes]] = {}
    if rid == 0 and rsize > 0 and pos - 6 - rsize >= 0:
        tbl = blob[pos - 6 - rsize : pos - 6]
        for off in range(0, len(tbl) - 9, 10):
            eid, efmt, esize, eoff = struct.unpack("<BBII", tbl[off : off + 10])
            if eid > 0 and esize > 0:
                records[eid] = (efmt, blob[eoff : eoff + esize])
    return records


def _parse_pb_scalars(data: bytes) -> dict:
    """Extract top-level protobuf scalar fields (varints, doubles, floats, bytes)."""
    out = {}
    i, n = 0, len(data)
    while i < n:
        key, shift = 0, 0
        while i < n:
            b = data[i]
            i += 1
            key |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        fn, wt = key >> 3, key & 7
        if wt == 0:
            val, shift = 0, 0
            while i < n:
                b = data[i]
                i += 1
                val |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            out[fn] = val
        elif wt == 1:
            if i + 8 > n:
                break
            out[fn] = struct.unpack("<d", data[i : i + 8])[0]
            i += 8
        elif wt == 2:
            ln, shift = 0, 0
            while i < n:
                b = data[i]
                i += 1
                ln |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            if i + ln > n:
                break
            out[fn] = data[i : i + ln]
            i += ln
        elif wt == 5:
            if i + 4 > n:
                break
            out[fn] = struct.unpack("<f", data[i : i + 4])[0]
            i += 4
        else:
            break
    return out


def parse_mei_offset(offset_str: str, lens_w: int, lens_h: int) -> list[MEILens]:
    parts = [float(x) for x in offset_str.strip().split("_") if x]
    if len(parts) < 55:
        raise ValueError(f"Expected >=55 MEI tokens, got {len(parts)}")
    lenses: list[MEILens] = []
    for li in range(2):
        s = 1 + li * 27
        xi = parts[s]
        fx_s, fy_s = parts[s + 1], parts[s + 2]
        cx_s, cy_s = parts[s + 3], parts[s + 4]
        yaw, pitch, roll = parts[s + 5], parts[s + 6], parts[s + 7]
        coeffs = parts[s + 11 : s + 24]
        ref_w, ref_h = parts[s + 24], parts[s + 25]
        ltype = int(parts[s + 26])

        cx_local = cx_s - (ref_h if li == 1 else 0.0)
        scale_x = lens_w / ref_h
        scale_y = lens_h / ref_h

        if li == 0:
            R = Rotation.from_euler("YXZ", [yaw, pitch, roll - 90.0], degrees=True).as_matrix()
        else:
            R_back = Rotation.from_euler("Y", 180.0, degrees=True)
            R_corr = Rotation.from_euler("YXZ", [yaw, pitch, roll - 90.0], degrees=True)
            R = (R_back * R_corr).as_matrix()

        lenses.append(
            MEILens(
                xi=xi,
                fx=fx_s * scale_x,
                fy=fy_s * scale_y,
                cx=cx_local * scale_x,
                cy=cy_s * scale_y,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                k1=coeffs[0],
                k2=coeffs[1],
                k3=coeffs[2],
                k4=coeffs[3],
                p1=coeffs[5],
                p2=coeffs[6],
                s1=coeffs[7],
                s2=coeffs[8],
                s3=coeffs[9],
                s4=coeffs[10],
                ref_w=ref_w,
                ref_h=ref_h,
                lens_type=ltype,
                R_ext=R,
                lens_idx=li,
            )
        )
    return lenses


def _detect_camera_name(meta_bytes: bytes) -> str:
    m = re.search(rb"Insta360 [A-Za-z0-9 ]{1,24}", meta_bytes)
    if m:
        return m.group().decode("ascii", errors="ignore").strip()
    return "Insta360"


def load_meta(path: str, lens_w: int = 1024, lens_h: int = 1024) -> Insta360ClipMeta:
    recs = _read_trailer_records(path)
    if 1 not in recs:
        raise ValueError(f"Insta360 metadata record (0x01) missing in {path}")
    meta_bytes = recs[1][1]
    pb = _parse_pb_scalars(meta_bytes)
    camera = _detect_camera_name(meta_bytes)
    m = re.search(rb"2_\d+\.\d+_[0-9._\-]+", meta_bytes)
    if not m:
        raise ValueError("MEI offset_v3 string not found in metadata")
    offset_str = m.group().decode("ascii")
    lenses = parse_mei_offset(offset_str, lens_w, lens_h)

    # Readout / rolling shutter duration (protobuf field 25 in ms, e.g. 15.36 ms)
    rs_ms = float(pb.get(25, 15.36))
    if not (0.0 <= rs_ms <= 50.0):
        rs_ms = 15.36
    rs_s = rs_ms / 1000.0
    first_frame_ts_abs = float(pb.get(24, 0)) / 1e6

    # Accel & gyro full-scale ranges (protobuf field 65 submessage: f1=accel_g, f2=gyro_dps)
    accel_range_g = 32.0
    gyro_range_dps = 2000.0
    if isinstance(pb.get(65), bytes):
        sub = _parse_pb_scalars(pb[65])
        if 1 in sub and 2 <= int(sub[1]) <= 64:
            accel_range_g = float(sub[1])
        if 2 in sub and 250 <= int(sub[2]) <= 4000:
            gyro_range_dps = float(sub[2])

    imu_t = np.array([], dtype=np.float64)
    imu_ts_abs = np.array([], dtype=np.float64)
    imu_acc = np.zeros((0, 3), dtype=np.float64)
    imu_gyr = np.zeros((0, 3), dtype=np.float64)
    if 3 in recs:
        gb = recs[3][1]
        n = len(gb) // 20
        dt = np.dtype(
            [
                ("ts", "<u8"),
                ("ax", "<u2"),
                ("ay", "<u2"),
                ("az", "<u2"),
                ("gx", "<u2"),
                ("gy", "<u2"),
                ("gz", "<u2"),
            ]
        )
        arr = np.frombuffer(gb[: n * 20], dtype=dt)
        t_us = arr["ts"].astype(np.float64)
        imu_ts_abs = t_us / 1e6
        imu_t = (t_us - t_us[0]) / 1e6
        ax = (arr["ax"].astype(np.float64) - 32768.0) / (32768.0 / accel_range_g)
        ay = (arr["ay"].astype(np.float64) - 32768.0) / (32768.0 / accel_range_g)
        az = (arr["az"].astype(np.float64) - 32768.0) / (32768.0 / accel_range_g)
        gx = np.deg2rad((arr["gx"].astype(np.float64) - 32768.0) / (32768.0 / gyro_range_dps))
        gy = np.deg2rad((arr["gy"].astype(np.float64) - 32768.0) / (32768.0 / gyro_range_dps))
        gz = np.deg2rad((arr["gz"].astype(np.float64) - 32768.0) / (32768.0 / gyro_range_dps))
        # Camera-frame basis: X=right, Y=down, Z=forward
        imu_acc = np.stack([az, -ax, ay], axis=-1)
        imu_gyr = np.stack([-gz, gx, -gy], axis=-1)

    def _read_exp(rec_id: int):
        if rec_id not in recs:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        eb = recs[rec_id][1]
        ne = len(eb) // 16
        edt = np.dtype([("ts", "<u8"), ("shutter", "<f8")])
        earr = np.frombuffer(eb[: ne * 16], dtype=edt)
        return earr["ts"].astype(np.float64) / 1e6, earr["shutter"].astype(np.float64)

    exp0_ts, exp0_sh = _read_exp(4)
    exp1_ts, exp1_sh = _read_exp(12)
    if len(exp1_ts) != len(exp0_ts):
        exp1_ts, exp1_sh = exp0_ts, exp0_sh

    return Insta360ClipMeta(
        camera=camera,
        lenses=lenses,
        imu_t=imu_t,
        imu_ts_abs=imu_ts_abs,
        imu_accel=imu_acc,
        imu_gyro=imu_gyr,
        exp0_ts_abs=exp0_ts,
        exp0_shutter=exp0_sh,
        exp1_ts_abs=exp1_ts,
        exp1_shutter=exp1_sh,
        rolling_shutter_s=rs_s,
        first_frame_ts_abs=first_frame_ts_abs,
        raw_offset_str=offset_str,
    )
