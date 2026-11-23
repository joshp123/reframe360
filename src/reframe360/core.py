"""Core 360° stitching, FlowState-grade stabilization, reframing, Resolve export, and cut QC."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from typing import Any

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

from .insta360_meta import Insta360ClipMeta, MEILens, load_meta


def find_binary(name: str, env_var: str) -> str:
    env_val = os.environ.get(env_var)
    if env_val:
        return env_val
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"{name} not found on PATH. Install ffmpeg or set the {env_var} environment variable."
    )


def load_keyframes(keyframes: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
    if isinstance(keyframes, list):
        return keyframes
    s = keyframes.strip()
    if s.startswith("["):
        return json.loads(s)
    with open(s, "r", encoding="utf-8") as f:
        return json.load(f)


def mei_project(Xl, Yl, Zl, L: MEILens, lens_w: int, lens_h: int):
    crop_dst = 7680.0 if abs(L.ref_h - 7744.0) < 5 else L.ref_h
    f_corr = L.ref_h / crop_dst
    fx, fy = L.fx * f_corr, L.fy * f_corr
    cx, cy = L.cx, L.cy

    norm = np.maximum(np.sqrt(Xl * Xl + Yl * Yl + Zl * Zl), 1e-10)
    Xs, Ys, Zs = Xl / norm, Yl / norm, Zl / norm
    theta_deg = np.degrees(np.arccos(np.clip(Zs, -1.0, 1.0)))

    denom = Zs + L.xi
    valid = (denom > 1e-5) & (theta_deg < 101.0)
    inv_d = np.where(valid, 1.0 / np.maximum(denom, 1e-5), 0.0)
    x, y = Xs * inv_d, Ys * inv_d

    r2 = x * x + y * y
    r4 = r2 * r2
    radial = 1.0 + L.k1 * r2 + L.k2 * r4 + L.k3 * r2 * r4 + L.k4 * r4 * r4
    xd = x * radial + 2.0 * L.p1 * x * y + L.p2 * (r2 + 2.0 * x * x) + L.s1 * r2 + L.s2 * r4
    yd = y * radial + L.p1 * (r2 + 2.0 * y * y) + L.p2 * x * y + L.s3 * r2 + L.s4 * r4

    u = fx * xd + cx
    v = fy * yd + cy
    r_px = np.sqrt((u - cx) ** 2 + (v - cy) ** 2)
    r_max = min(lens_w, lens_h) * 0.486
    valid = valid & (u >= 1) & (u < lens_w - 2) & (v >= 1) & (v < lens_h - 2) & (r_px < r_max)
    return u.astype(np.float32), v.astype(np.float32), valid, theta_deg.astype(np.float32)


def _rot_align_to_y(gv: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix aligning unit vector gv to [0, +1, 0] (camera down)."""
    gv = gv / max(float(np.linalg.norm(gv)), 1e-9)
    tgt = np.array([0.0, 1.0, 0.0])
    ax = np.cross(gv, tgt)
    s = float(np.linalg.norm(ax))
    c = float(np.clip(np.dot(gv, tgt), -1.0, 1.0))
    if s < 1e-6:
        return np.eye(3) if c > 0 else Rotation.from_euler("X", 180.0, degrees=True).as_matrix()
    return Rotation.from_rotvec((ax / s) * math.acos(c)).as_matrix()


def compute_flowstate_slerp(
    meta: Insta360ClipMeta,
    fps: float,
    n_frames: int,
    lock_horizon: bool = True,
    yaw_follow_sigma: float = 0.0,
) -> dict[str, Any]:
    """1000 Hz orientation solver with hardware exposure sync and centrifugal-gated horizon lock."""
    if len(meta.imu_ts_abs) < 20 or not lock_horizon:
        id_t = np.array([0.0, max(1.0, n_frames / fps + 1.0)])
        id_s = Slerp(id_t, Rotation.from_quat([[0, 0, 0, 1], [0, 0, 0, 1]]))
        exp_t = np.arange(n_frames, dtype=np.float64) / fps
        return {
            "slerp": id_s,
            "t_min": 0.0,
            "t_max": float(id_t[-1]),
            "exp0_ts": exp_t,
            "exp0_sh": np.zeros(n_frames),
            "exp1_ts": exp_t,
            "exp1_sh": np.zeros(n_frames),
            "rs_s": 0.0,
            "omega_ts": id_t,
            "omega_mag": np.zeros(2),
        }

    ts = meta.imu_ts_abs
    acc = meta.imu_accel
    gyr = meta.imu_gyro.copy()
    ng = len(ts)

    # Estimate static gyro bias only when genuinely stationary samples exist
    acc_norm = np.linalg.norm(acc, axis=1)
    gyr_norm = np.linalg.norm(gyr, axis=1)
    still = (np.abs(acc_norm - 1.0) < 0.03) & (gyr_norm < np.deg2rad(1.5))
    if np.count_nonzero(still) > 200:
        bias = np.median(gyr[still], axis=0)
        if np.linalg.norm(bias) < np.deg2rad(0.5):
            gyr = gyr - bias

    # Step 1: Full-rate 1000 Hz quaternion integration (cam -> raw_world)
    quats = np.empty((ng, 4), dtype=np.float64)
    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    quats[0] = (qx, qy, qz, qw)
    dt_arr = np.diff(ts)
    for i in range(1, ng):
        dt = float(dt_arr[i - 1])
        if 0.0 < dt < 0.05:
            wx = 0.5 * dt * float(gyr[i, 0])
            wy = 0.5 * dt * float(gyr[i, 1])
            wz = 0.5 * dt * float(gyr[i, 2])
            nx = qx + qw * wx + qy * wz - qz * wy
            ny = qy + qw * wy + qz * wx - qx * wz
            nz = qz + qw * wz + qx * wy - qy * wx
            nw = qw - qx * wx - qy * wy - qz * wz
            invn = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw)
            qx, qy, qz, qw = nx * invn, ny * invn, nz * invn, nw * invn
        quats[i] = (qx, qy, qz, qw)

    # Step 2: Centrifugal-gated zero-phase gravity estimation in raw_world frame
    sub = slice(None, None, 20)
    ts_sub = ts[sub]
    R_raw_sub = Rotation.from_quat(quats[sub]).as_matrix()
    acc_sub = acc[sub]
    an_sub = acc_norm[sub]
    gn_sub = gyr_norm[sub]

    # Trust gravity only when |a| ~ 1g and angular velocity is modest
    w_g = np.exp(-((an_sub - 1.0) / 0.08) ** 2) * np.exp(-((gn_sub / np.deg2rad(35.0)) ** 2))
    if np.sum(w_g) < 1.0:
        w_g = np.exp(-((an_sub - 1.0) / 0.25) ** 2) + 1e-3

    g_world = np.einsum("nij,nj->ni", R_raw_sub, acc_sub)
    sig_g = min(200.0, max(15.0, len(ts_sub) * 0.25))
    num = np.stack([gaussian_filter1d(g_world[:, k] * w_g, sig_g, mode="nearest") for k in range(3)], axis=-1)
    den = np.maximum(gaussian_filter1d(w_g, sig_g, mode="nearest"), 1e-8)[:, None]
    g_smooth = num / den
    g_smooth /= np.maximum(np.linalg.norm(g_smooth, axis=1, keepdims=True), 1e-8)

    q_corr_sub = np.array([Rotation.from_matrix(_rot_align_to_y(gv)).as_quat() for gv in g_smooth])
    for i in range(1, len(q_corr_sub)):
        if np.dot(q_corr_sub[i], q_corr_sub[i - 1]) < 0:
            q_corr_sub[i] = -q_corr_sub[i]
    q_corr_full = np.stack([np.interp(ts, ts_sub, q_corr_sub[:, k]) for k in range(4)], axis=-1)
    q_corr_full /= np.linalg.norm(q_corr_full, axis=1, keepdims=True)

    def qmul(a, b):
        ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return np.stack(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            axis=-1,
        )

    q_level = qmul(q_corr_full, quats)

    # Zero initial heading yaw so yaw=0 corresponds to camera front at clip start
    t_ref = meta.exp0_ts_abs[0] if len(meta.exp0_ts_abs) > 0 else ts[0]
    ref_idx = int(np.clip(np.searchsorted(ts, t_ref), 0, ng - 1))
    R0 = Rotation.from_quat(q_level[ref_idx]).as_matrix()
    fwd0 = R0[:, 2]
    yaw0 = math.atan2(fwd0[0], fwd0[2])
    q_yaw0 = Rotation.from_euler("Y", -yaw0, degrees=False).as_quat()
    q_level = qmul(np.broadcast_to(q_yaw0, q_level.shape), q_level)

    # Optional smooth yaw-follow mode: let virtual world gently follow physical heading
    if yaw_follow_sigma > 0.0:
        R_lvl_sub = Rotation.from_quat(q_level[sub]).as_matrix()
        fwd = R_lvl_sub[:, :, 2]
        yaw_cam = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
        dt_sub = float(np.median(np.diff(ts_sub))) if len(ts_sub) > 1 else 0.02
        yaw_sm = gaussian_filter1d(yaw_cam, max(1.0, yaw_follow_sigma / max(dt_sub, 1e-4)), mode="nearest")
        yaw_full = np.interp(ts, ts_sub, yaw_sm)
        half = -0.5 * yaw_full
        q_yfollow = np.stack([np.zeros_like(half), np.sin(half), np.zeros_like(half), np.cos(half)], axis=-1)
        q_level = qmul(q_yfollow, q_level)

    keep = np.concatenate([[True], np.diff(ts) > 0])
    slerp = Slerp(ts[keep], Rotation.from_quat(q_level[keep]))

    if len(meta.exp0_ts_abs) >= n_frames:
        exp0_ts = meta.exp0_ts_abs[:n_frames]
        exp0_sh = meta.exp0_shutter[:n_frames]
        exp1_ts = meta.exp1_ts_abs[:n_frames]
        exp1_sh = meta.exp1_shutter[:n_frames]
    elif len(meta.exp0_ts_abs) > 0:
        k0 = len(meta.exp0_ts_abs)
        extra = np.arange(1, n_frames - k0 + 1) / fps
        exp0_ts = np.concatenate([meta.exp0_ts_abs, meta.exp0_ts_abs[-1] + extra])
        exp0_sh = np.concatenate([meta.exp0_shutter, np.full(n_frames - k0, meta.exp0_shutter[-1])])
        exp1_ts = np.concatenate([meta.exp1_ts_abs, meta.exp1_ts_abs[-1] + extra])
        exp1_sh = np.concatenate([meta.exp1_shutter, np.full(n_frames - k0, meta.exp1_shutter[-1])])
    else:
        t0_fb = meta.first_frame_ts_abs if meta.first_frame_ts_abs > ts[0] else (ts[0] + 0.244)
        exp0_ts = t0_fb + np.arange(n_frames, dtype=np.float64) / fps
        exp0_sh = np.full(n_frames, 0.001)
        exp1_ts, exp1_sh = exp0_ts, exp0_sh

    return {
        "slerp": slerp,
        "t_min": float(ts[keep][0]),
        "t_max": float(ts[keep][-1]),
        "exp0_ts": exp0_ts,
        "exp0_sh": exp0_sh,
        "exp1_ts": exp1_ts,
        "exp1_sh": exp1_sh,
        "rs_s": meta.rolling_shutter_s,
        "omega_ts": ts,
        "omega_mag": gyr_norm,
    }


def blend_two_views(
    img0,
    img1,
    valid0,
    valid1,
    theta0,
    theta1,
    half_blend_deg: float = 3.5,
    gain_match: bool = True,
    multiband: bool = True,
):
    ov = valid0 & valid1
    diff = theta1 - theta0
    u = np.clip((diff + half_blend_deg) / (2.0 * half_blend_deg), 0.0, 1.0)
    w0 = u * u * (3.0 - 2.0 * u)
    w0 = np.where(valid0 & ~valid1, 1.0, w0)
    w0 = np.where(~valid0 & valid1, 0.0, w0)
    w0 = np.where(~valid0 & ~valid1, 0.0, w0)

    i0 = img0.astype(np.float32)
    i1 = img1.astype(np.float32)

    if gain_match and np.count_nonzero(ov) > 80:
        for c in range(3):
            v0, v1 = i0[:, :, c][ov], i1[:, :, c][ov]
            good = (v0 > 8) & (v1 > 8) & (v0 < 248) & (v1 < 248)
            if np.count_nonzero(good) > 40:
                ratio = float(np.clip(np.median(v0[good] / v1[good]), 0.72, 1.38))
                g = math.sqrt(ratio)
                i0[:, :, c] = np.clip(i0[:, :, c] / g, 0, 255)
                i1[:, :, c] = np.clip(i1[:, :, c] * g, 0, 255)

    if multiband and np.count_nonzero(ov) > 80:
        levels = 3
        h, w = i0.shape[:2]
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        a = cv2.copyMakeBorder(i0, 0, ph, 0, pw, cv2.BORDER_REFLECT_101) if (ph or pw) else i0
        b = cv2.copyMakeBorder(i1, 0, ph, 0, pw, cv2.BORDER_REFLECT_101) if (ph or pw) else i1
        m = (
            cv2.copyMakeBorder(w0.astype(np.float32), 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
            if (ph or pw)
            else w0.astype(np.float32)
        )
        v0p = (
            (cv2.copyMakeBorder(valid0.astype(np.uint8), 0, ph, 0, pw, cv2.BORDER_REFLECT_101) > 0)
            if (ph or pw)
            else valid0
        )
        v1p = (
            (cv2.copyMakeBorder(valid1.astype(np.uint8), 0, ph, 0, pw, cv2.BORDER_REFLECT_101) > 0)
            if (ph or pw)
            else valid1
        )
        a_c = np.where(v0p[:, :, None], a, b)
        b_c = np.where(v1p[:, :, None], b, a)

        ga, gb, gm = [a_c], [b_c], [np.repeat(m[:, :, None], 3, axis=2)]
        for _ in range(levels - 1):
            ga.append(cv2.pyrDown(ga[-1]))
            gb.append(cv2.pyrDown(gb[-1]))
            gm.append(cv2.pyrDown(gm[-1]))
        la = [
            ga[k] - cv2.pyrUp(ga[k + 1], dstsize=(ga[k].shape[1], ga[k].shape[0]))
            for k in range(levels - 1)
        ] + [ga[-1]]
        lb = [
            gb[k] - cv2.pyrUp(gb[k + 1], dstsize=(gb[k].shape[1], gb[k].shape[0]))
            for k in range(levels - 1)
        ] + [gb[-1]]
        out = la[-1] * gm[-1] + lb[-1] * (1.0 - gm[-1])
        for k in range(levels - 2, -1, -1):
            out = cv2.pyrUp(out, dstsize=(la[k].shape[1], la[k].shape[0])) + (
                la[k] * gm[k] + lb[k] * (1.0 - gm[k])
            )
        return np.clip(out[:h, :w], 0, 255).astype(np.uint8)

    w3 = w0[:, :, None]
    return np.clip(i0 * w3 + i1 * (1.0 - w3), 0, 255).astype(np.uint8)


def render_rays(
    f0,
    f1,
    rays_cam,
    meta: Insta360ClipMeta,
    half_blend_deg: float = 3.5,
    gain_match: bool = True,
    multiband: bool = True,
):
    H, W = rays_cam.shape[:2]
    lh, lw = f0.shape[:2]
    flat = rays_cam.reshape(-1, 3).T
    imgs, valids, thetas = [], [], []
    for fimg, L in zip((f0, f1), meta.lenses):
        rl = L.R_ext.T @ flat
        u, v, val, th = mei_project(rl[0].reshape(H, W), rl[1].reshape(H, W), rl[2].reshape(H, W), L, lw, lh)
        remapped = cv2.remap(fimg, u, v, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        imgs.append(remapped)
        valids.append(val)
        thetas.append(th)
    return blend_two_views(
        imgs[0],
        imgs[1],
        valids[0],
        valids[1],
        thetas[0],
        thetas[1],
        half_blend_deg=half_blend_deg,
        gain_match=gain_match,
        multiband=multiband,
    )


def render_frame_flowstate(
    f0,
    f1,
    rays_world,
    fi: int,
    fs: dict[str, Any],
    meta: Insta360ClipMeta,
    half_blend_deg: float = 3.5,
    gain_match: bool = True,
    multiband: bool = True,
    rolling_shutter: bool = True,
):
    """Render frame fi using per-lens exposure timestamps and 8-band vertical rolling shutter correction."""
    slerp = fs["slerp"]
    tmin, tmax = fs["t_min"], fs["t_max"]
    t0 = float(np.clip(fs["exp0_ts"][fi] + 0.5 * fs["exp0_sh"][fi], tmin, tmax))
    t1 = float(np.clip(fs["exp1_ts"][fi] + 0.5 * fs["exp1_sh"][fi], tmin, tmax))
    rs = fs["rs_s"] if rolling_shutter else 0.0

    oi = int(np.clip(np.searchsorted(fs["omega_ts"], t0), 0, len(fs["omega_mag"]) - 1))
    use_rs = (rs > 1e-4) and (float(fs["omega_mag"][oi]) > np.deg2rad(8.0))

    if not use_rs:
        R_c0 = slerp(t0).as_matrix().T
        rc = np.einsum("ij,hwj->hwi", R_c0, rays_world)
        return render_rays(
            f0,
            f1,
            rc,
            meta,
            half_blend_deg=half_blend_deg,
            gain_match=gain_match,
            multiband=multiband,
        )

    H, W = rays_world.shape[:2]
    lh, lw = f0.shape[:2]
    n_bands = 8
    fracs = np.linspace(-0.5, 0.5, n_bands)
    imgs, valids, thetas = [], [], []
    for fimg, L, tb in zip((f0, f1), meta.lenses, (t0, t1)):
        R_mid = slerp(tb).as_matrix().T
        rl0 = L.R_ext.T @ (R_mid @ rays_world.reshape(-1, 3).T)
        _, v0, _, _ = mei_project(rl0[0].reshape(H, W), rl0[1].reshape(H, W), rl0[2].reshape(H, W), L, lw, lh)
        bidx = np.clip(np.round((v0 / lh) * (n_bands - 1)).astype(np.int32), 0, n_bands - 1)
        t_bands = np.clip(tb + fracs * rs, tmin, tmax)
        Rs = [r.as_matrix().T for r in slerp(t_bands)]

        u_fin = np.zeros((H, W), np.float32)
        v_fin = np.zeros((H, W), np.float32)
        val_fin = np.zeros((H, W), bool)
        th_fin = np.full((H, W), 180.0, np.float32)
        for bi in range(n_bands):
            mask = bidx == bi
            if not np.any(mask):
                continue
            rl_b = L.R_ext.T @ (Rs[bi] @ rays_world[mask].T)
            ub, vb, valb, thb = mei_project(rl_b[0], rl_b[1], rl_b[2], L, lw, lh)
            u_fin[mask] = ub
            v_fin[mask] = vb
            val_fin[mask] = valb
            th_fin[mask] = thb
        remapped = cv2.remap(
            fimg, u_fin, v_fin, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        imgs.append(remapped)
        valids.append(val_fin)
        thetas.append(th_fin)
    return blend_two_views(
        imgs[0],
        imgs[1],
        valids[0],
        valids[1],
        thetas[0],
        thetas[1],
        half_blend_deg=half_blend_deg,
        gain_match=gain_match,
        multiband=multiband,
    )


class CubeLUT:
    """Vectorized .cube 3D LUT trilinear interpolator."""

    def __init__(self, path: str):
        size = None
        vals = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("LUT_3D_SIZE"):
                    size = int(s.split()[1])
                elif s[0] in "0123456789.-":
                    parts = s.split()
                    if len(parts) == 3:
                        vals.append((float(parts[0]), float(parts[1]), float(parts[2])))
        if size is None or len(vals) != size ** 3:
            raise ValueError(f"Invalid .cube LUT {path}: size={size}, entries={len(vals)}")
        self.size = size
        self.table = np.array(vals, dtype=np.float32).reshape(size, size, size, 3)

    def apply_bgr(self, bgr: np.ndarray) -> np.ndarray:
        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
        s = self.size - 1
        r = np.clip(rgb[:, :, 0] * s, 0.0, s - 1e-4)
        g = np.clip(rgb[:, :, 1] * s, 0.0, s - 1e-4)
        b = np.clip(rgb[:, :, 2] * s, 0.0, s - 1e-4)
        r0 = r.astype(np.int32)
        g0 = g.astype(np.int32)
        b0 = b.astype(np.int32)
        dr = (r - r0)[:, :, None]
        dg = (g - g0)[:, :, None]
        db = (b - b0)[:, :, None]
        T = self.table
        c000 = T[b0, g0, r0]
        c001 = T[b0, g0, r0 + 1]
        c010 = T[b0, g0 + 1, r0]
        c011 = T[b0, g0 + 1, r0 + 1]
        c100 = T[b0 + 1, g0, r0]
        c101 = T[b0 + 1, g0, r0 + 1]
        c110 = T[b0 + 1, g0 + 1, r0]
        c111 = T[b0 + 1, g0 + 1, r0 + 1]
        c00 = c000 * (1 - dr) + c001 * dr
        c01 = c010 * (1 - dr) + c011 * dr
        c10 = c100 * (1 - dr) + c101 * dr
        c11 = c110 * (1 - dr) + c111 * dr
        c0 = c00 * (1 - dg) + c01 * dg
        c1 = c10 * (1 - dg) + c11 * dg
        out = np.clip(c0 * (1 - db) + c1 * db, 0.0, 1.0)
        return (out[:, :, ::-1] * 255.0).astype(np.uint8)


_REC2020_TO_709 = np.array(
    [
        [1.6605, -0.5876, -0.0728],
        [-0.1246, 1.1329, -0.0083],
        [-0.0182, -0.1006, 1.1187],
    ],
    dtype=np.float32,
)


def grade_ilog(
    bgr: np.ndarray,
    warmth: float = 1.0,
    sat: float = 1.12,
    lut: CubeLUT | None = None,
) -> np.ndarray:
    """Rec.2020-aware I-Log -> Rec.709 preview grade, or exact .cube 3D LUT when supplied."""
    if lut is not None:
        return lut.apply_bgr(bgr)
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    x = np.clip((rgb - 0.095) / 0.84, 0.0, 1.0)
    lin = np.power(x, 1.65)
    m = 0.55 * np.eye(3, dtype=np.float32) + 0.45 * _REC2020_TO_709
    lin709 = np.clip(np.einsum("ij,hwj->hwi", m, lin), 0.0, 1.5)
    lin709 = lin709 / (1.0 + 0.18 * lin709) * 1.15
    y = np.power(np.clip(lin709, 0.0, 1.0), 1.0 / 1.95)
    y = y * (1.0 - 0.22) + (y * y * (3.0 - 2.0 * y)) * 0.22
    if warmth != 1.0:
        y[:, :, 0] = np.clip(y[:, :, 0] * warmth, 0.0, 1.0)
        y[:, :, 2] = np.clip(y[:, :, 2] / math.sqrt(warmth), 0.0, 1.0)
    if abs(sat - 1.0) > 1e-3:
        luma = 0.2126 * y[:, :, 0] + 0.7152 * y[:, :, 1] + 0.0722 * y[:, :, 2]
        y = np.clip(luma[:, :, None] + (y - luma[:, :, None]) * sat, 0.0, 1.0)
    return (y[:, :, ::-1] * 255.0).astype(np.uint8)


def equirect_rays(w: int, h: int) -> np.ndarray:
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    lon = (uu / w) * 2.0 * np.pi - np.pi
    lat = np.pi / 2.0 - (vv / h) * np.pi
    return np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], axis=-1)


def rectilinear_rays(
    w: int,
    h: int,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    hfov_deg: float,
) -> np.ndarray:
    f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    uu, vv = np.meshgrid(
        np.arange(w, dtype=np.float64) - (w - 1) / 2.0,
        np.arange(h, dtype=np.float64) - (h - 1) / 2.0,
    )
    rays = np.stack([uu, vv, np.full_like(uu, f)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    R = Rotation.from_euler("YXZ", [yaw_deg, pitch_deg, roll_deg], degrees=True).as_matrix()
    return np.einsum("ij,hwj->hwi", R, rays)


def open_insta360_reader(path: str):
    ffprobe = find_binary("ffprobe", "FFPROBE")
    p = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=index,width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(p.stdout)["streams"]
    dual_track = len(streams) >= 2 and streams[0]["width"] == streams[0]["height"]
    num, den = map(int, streams[0]["r_frame_rate"].split("/"))
    fps = num / den
    n_frames = int(streams[0].get("nb_frames") or 0)
    if dual_track:
        lw, lh = int(streams[0]["width"]), int(streams[0]["height"])
    else:
        lw, lh = int(streams[0]["width"]) // 2, int(streams[0]["height"])
    meta = load_meta(path, lens_w=lw, lens_h=lh)
    if n_frames == 0:
        cap = cv2.VideoCapture(path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return meta, fps, n_frames, lw, lh, dual_track


def inspect_clip_metadata(input_path: str) -> dict[str, Any]:
    """Read stream dimensions, frame rate, MEI calibration, IMU rate, and rolling shutter metadata."""
    meta, fps, n_frames, lw, lh, dual_track = open_insta360_reader(input_path)
    imu_rate_hz = 0.0
    if len(meta.imu_ts_abs) > 1:
        dt = float(np.median(np.diff(meta.imu_ts_abs)))
        if dt > 0:
            imu_rate_hz = round(1.0 / dt, 1)
    return {
        "input": input_path,
        "camera": meta.camera,
        "layout": "dual_track" if dual_track else "side_by_side",
        "lens_width": lw,
        "lens_height": lh,
        "fps": round(fps, 4),
        "frames": n_frames,
        "duration_s": round(n_frames / fps, 3) if fps > 0 else 0.0,
        "imu_samples": int(len(meta.imu_ts_abs)),
        "imu_rate_hz": imu_rate_hz,
        "exposure_timestamps": int(len(meta.exp0_ts_abs)),
        "rolling_shutter_ms": round(meta.rolling_shutter_s * 1000.0, 3),
        "lenses": [
            {
                "lens_idx": L.lens_idx,
                "xi": round(L.xi, 5),
                "fx": round(L.fx, 2),
                "fy": round(L.fy, 2),
                "cx": round(L.cx, 2),
                "cy": round(L.cy, 2),
            }
            for L in meta.lenses
        ],
    }


def _read_dual_track_pair(path: str, ts: float, lw: int, lh: int, ffmpeg: str):
    fbytes = lw * lh * 3
    frames = []
    for stream_idx in (0, 1):
        p = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-ss",
                f"{ts:.4f}",
                "-i",
                path,
                "-map",
                f"0:v:{stream_idx}",
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ],
            capture_output=True,
            check=True,
        )
        if len(p.stdout) < fbytes:
            return None, None
        frames.append(np.frombuffer(p.stdout[:fbytes], np.uint8).reshape(lh, lw, 3))
    return frames[0], frames[1]


def _draw_sphere_grid(eq: np.ndarray, ts: float, fi: int, omega_dps: float = 0.0, swing_g: float = 0.0):
    H, W = eq.shape[:2]
    for pdeg, col, thick in [(0, (0, 230, 255), 2), (30, (160, 200, 255), 1), (-30, (160, 200, 255), 1)]:
        y = int((90.0 - pdeg) / 180.0 * H)
        cv2.line(eq, (0, y), (W, y), col, thick)
        cv2.putText(eq, f"p{pdeg:+d}", (4, max(14, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    for deg in range(-150, 180, 30):
        x = int((deg + 180) / 360.0 * W)
        col = (0, 80, 255) if abs(deg) == 90 else ((0, 255, 180) if deg == 0 else (200, 240, 200))
        cv2.line(eq, (x, 0), (x, H), col, 1)
        cv2.putText(eq, f"{deg:+d}", (x + 2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    cv2.rectangle(eq, (0, H - 28), (320, H), (0, 0, 0), -1)
    cv2.putText(
        eq,
        f"t={ts:.1f}s f{fi} w={omega_dps:.0f}d/s sw={swing_g:.2f}g",
        (6, H - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_crop_box(
    eq: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    hfov_deg: float,
    aspect: float = 16 / 9,
):
    """Project the exact rectilinear viewport border onto the equirectangular sheet."""
    H, W = eq.shape[:2]
    f = 1.0 / math.tan(math.radians(hfov_deg) / 2.0)
    vh = 1.0 / aspect
    xs = np.concatenate([np.linspace(-1, 1, 24), np.full(16, 1.0), np.linspace(1, -1, 24), np.full(16, -1.0)])
    ys = np.concatenate([np.full(24, -vh), np.linspace(-vh, vh, 16), np.full(24, vh), np.linspace(vh, -vh, 16)])
    rays = np.stack([xs, ys, np.full_like(xs, f)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    R = Rotation.from_euler("YXZ", [yaw_deg, pitch_deg, roll_deg], degrees=True).as_matrix()
    rw = (R @ rays.T).T
    lon = np.arctan2(rw[:, 0], rw[:, 2])
    lat = np.arcsin(np.clip(-rw[:, 1], -1.0, 1.0))
    px = ((lon + np.pi) / (2.0 * np.pi) * W).astype(np.int32)
    py = ((np.pi / 2.0 - lat) / np.pi * H).astype(np.int32)
    for i in range(len(px)):
        j = (i + 1) % len(px)
        if abs(int(px[i]) - int(px[j])) < W // 2:
            cv2.line(eq, (int(px[i]), int(py[i])), (int(px[j]), int(py[j])), (0, 255, 255), 2, cv2.LINE_AA)
    rc = R @ np.array([0.0, 0.0, 1.0])
    cx = int((math.atan2(rc[0], rc[2]) + math.pi) / (2.0 * math.pi) * W)
    cy = int((math.pi / 2.0 - math.asin(max(-1.0, min(1.0, -rc[1])))) / math.pi * H)
    cv2.drawMarker(eq, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)


def eval_keyframes(
    kfs: list[dict[str, Any]],
    times: np.ndarray,
    smooth_sigma_s: float = 0.0,
    fps: float = 25.0,
):
    """Evaluate camera keyframes with smootherstep (2 KFs) or PCHIP + optional Gaussian smoothing (3+ KFs)."""
    kt = np.array([k["t"] for k in kfs], dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    n = len(times)
    if len(kt) == 1:
        return (
            np.full(n, float(kfs[0]["yaw"])),
            np.full(n, float(kfs[0].get("pitch", 0.0))),
            np.full(n, float(kfs[0].get("roll", 0.0))),
            np.full(n, float(kfs[0].get("fov", 80.0))),
        )

    raw_yaws = np.array([float(k["yaw"]) for k in kfs], dtype=np.float64)
    yaws = np.rad2deg(np.unwrap(np.deg2rad(raw_yaws)))
    pitches = np.array([float(k.get("pitch", 0.0)) for k in kfs], dtype=np.float64)
    rolls = np.array([float(k.get("roll", 0.0)) for k in kfs], dtype=np.float64)
    fovs = np.array([float(k.get("fov", 80.0)) for k in kfs], dtype=np.float64)
    tc = np.clip(times, kt[0], kt[-1])

    if len(kt) == 2:
        ease = kfs[0].get("ease", "smootherstep")
        u = np.clip((tc - kt[0]) / max(kt[1] - kt[0], 1e-6), 0.0, 1.0)
        if ease == "linear":
            s = u
        elif ease == "cubic":
            s = u * u * (3.0 - 2.0 * u)
        else:
            s = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
        return (
            yaws[0] + (yaws[1] - yaws[0]) * s,
            pitches[0] + (pitches[1] - pitches[0]) * s,
            rolls[0] + (rolls[1] - rolls[0]) * s,
            fovs[0] + (fovs[1] - fovs[0]) * s,
        )

    if smooth_sigma_s > 0.0:
        grid_fps = max(25.0, float(fps))
        ng = max(2, int(round((kt[-1] - kt[0]) * grid_fps)) + 1)
        tg = np.linspace(kt[0], kt[-1], ng)
        sig = max(0.5, smooth_sigma_s * grid_fps)
        out = []
        for arr in (yaws, pitches, rolls, fovs):
            dense = PchipInterpolator(kt, arr)(tg)
            dense = gaussian_filter1d(dense, sig, mode="nearest")
            out.append(np.interp(tc, tg, dense))
        return tuple(out)
    return tuple(PchipInterpolator(kt, arr)(tc) for arr in (yaws, pitches, rolls, fovs))


def inspect_sphere(
    input_path: str,
    output_path: str,
    count: int = 6,
    step: float | None = None,
    times: list[float] | None = None,
    keyframes: list[dict[str, Any]] | str | None = None,
    aspect: float = 16 / 9,
    smooth_pan: float = 0.35,
    mode: str = "world",
    yaw_smooth: float = 1.2,
    lut_path: str | None = None,
    no_stab: bool = False,
) -> dict[str, Any]:
    """Render a horizon-locked equirectangular contact sheet with yaw/pitch grid and optional crop overlay."""
    meta, fps, n_frames, lw, lh, dual_track = open_insta360_reader(input_path)
    yaw_sm = yaw_smooth if mode == "follow" else 0.0
    fs = compute_flowstate_slerp(meta, fps, n_frames, lock_horizon=not no_stab, yaw_follow_sigma=yaw_sm)
    lut = CubeLUT(lut_path) if lut_path else None
    kfs = load_keyframes(keyframes) if keyframes else None
    dur = n_frames / fps

    if times:
        sample_times = [float(x) for x in times]
    elif step:
        sample_times = [float(x) for x in np.arange(0.5, max(0.6, dur - 0.3), step)]
    elif kfs:
        sample_times = [float(x) for x in np.linspace(kfs[0]["t"], kfs[-1]["t"], count)]
    else:
        sample_times = [float(x) for x in np.linspace(0.5, max(0.6, dur - 0.5), count)]

    kf_eval = eval_keyframes(kfs, np.array(sample_times), smooth_sigma_s=smooth_pan, fps=fps) if kfs else None

    W, H = 960, 480
    rays_w = equirect_rays(W, H)
    tiles = []
    acc_dev = np.abs(np.linalg.norm(meta.imu_accel, axis=1) - 1.0) if len(meta.imu_accel) else np.zeros(1)

    ffmpeg = find_binary("ffmpeg", "FFMPEG") if dual_track else None
    cap = None if dual_track else cv2.VideoCapture(input_path)
    try:
        for ti, ts in enumerate(sample_times):
            fi = int(np.clip(round(ts * fps), 0, n_frames - 1))
            if dual_track:
                f0, f1 = _read_dual_track_pair(input_path, ts, lw, lh, ffmpeg)
                if f0 is None or f1 is None:
                    continue
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frm = cap.read()
                if not ok:
                    continue
                f0, f1 = frm[:, :lw], frm[:, lw:]

            eq = grade_ilog(
                render_frame_flowstate(f0, f1, rays_w, fi, fs, meta, multiband=False, rolling_shutter=True),
                lut=lut,
            )
            t_abs = fs["exp0_ts"][fi]
            oi = int(np.clip(np.searchsorted(fs["omega_ts"], t_abs), 0, len(fs["omega_mag"]) - 1))
            w_dps = float(np.rad2deg(fs["omega_mag"][oi]))
            sw_g = float(acc_dev[min(oi, len(acc_dev) - 1)])
            _draw_sphere_grid(eq, ts, fi, w_dps, sw_g)
            if kf_eval is not None:
                _draw_crop_box(eq, kf_eval[0][ti], kf_eval[1][ti], kf_eval[2][ti], kf_eval[3][ti], aspect=aspect)
            tiles.append(eq)
    finally:
        if cap is not None:
            cap.release()

    if not tiles:
        raise RuntimeError(f"Could not decode any frames from {input_path}")

    cols = 2 if len(tiles) >= 4 else 1
    rows = math.ceil(len(tiles) / cols)
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((H, W, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return {
        "output": os.path.abspath(output_path),
        "duration_s": round(dur, 3),
        "fps": round(fps, 4),
        "views": len(sample_times),
        "times": [round(t, 3) for t in sample_times],
        "mode": mode,
        "dual_track": dual_track,
    }


def reframe_clip(
    input_path: str,
    output_path: str,
    keyframes: list[dict[str, Any]] | str,
    width: int = 1280,
    height: int = 720,
    start: float | None = None,
    end: float | None = None,
    crf: int = 18,
    grade: str = "ilog",
    lut_path: str | None = None,
    warmth: float = 1.0,
    sat: float = 1.12,
    mode: str = "world",
    yaw_smooth: float = 1.2,
    smooth_pan: float = 0.35,
    no_rs: bool = False,
    no_stab: bool = False,
) -> dict[str, Any]:
    """Render a FlowState-stabilized, keyframed rectilinear 16:9 or 9:16 MP4 directly from raw dual-fisheye."""
    ffmpeg = find_binary("ffmpeg", "FFMPEG")
    kfs = load_keyframes(keyframes)
    meta, fps, n_frames, lw, lh, dual_track = open_insta360_reader(input_path)
    yaw_sm = yaw_smooth if mode == "follow" else 0.0
    fs = compute_flowstate_slerp(meta, fps, n_frames, lock_horizon=not no_stab, yaw_follow_sigma=yaw_sm)
    lut = CubeLUT(lut_path) if lut_path else None

    t0 = start if start is not None else float(kfs[0]["t"])
    t1 = end if end is not None else float(kfs[-1]["t"])
    f_start = int(np.clip(round(t0 * fps), 0, n_frames - 1))
    f_end = int(np.clip(round(t1 * fps), f_start + 1, n_frames))
    frame_indices = np.arange(f_start, f_end)
    frame_times = frame_indices / fps

    yaws, pitches, rolls, fovs = eval_keyframes(kfs, frame_times, smooth_sigma_s=smooth_pan, fps=fps)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ff_cmd = [
        ffmpeg,
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "pipe:0",
        "-ss",
        f"{t0:.4f}",
        "-t",
        f"{(f_end - f_start) / fps:.4f}",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-threads",
        "2",
        output_path,
    ]
    proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE)
    rendered = 0
    try:
        if not dual_track:
            cap = cv2.VideoCapture(input_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
            try:
                for k, fi in enumerate(frame_indices):
                    ok, frm = cap.read()
                    if not ok:
                        break
                    f0, f1 = frm[:, :lw], frm[:, lw:]
                    rw = rectilinear_rays(width, height, yaws[k], pitches[k], rolls[k], fovs[k])
                    out = render_frame_flowstate(
                        f0,
                        f1,
                        rw,
                        int(fi),
                        fs,
                        meta,
                        half_blend_deg=3.5,
                        gain_match=True,
                        multiband=True,
                        rolling_shutter=not no_rs,
                    )
                    if grade == "ilog" or lut is not None:
                        out = grade_ilog(out, warmth=warmth, sat=sat, lut=lut)
                    proc.stdin.write(out.tobytes())
                    rendered += 1
            finally:
                cap.release()
        else:
            def mk_pipe(stream_idx: int):
                return subprocess.Popen(
                    [
                        ffmpeg,
                        "-v",
                        "error",
                        "-ss",
                        f"{t0:.4f}",
                        "-i",
                        input_path,
                        "-map",
                        f"0:v:{stream_idx}",
                        "-frames:v",
                        str(len(frame_indices)),
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "bgr24",
                        "-threads",
                        "2",
                        "pipe:1",
                    ],
                    stdout=subprocess.PIPE,
                )

            p0, p1 = mk_pipe(0), mk_pipe(1)
            fbytes = lw * lh * 3
            try:
                for k, fi in enumerate(frame_indices):
                    b0 = p0.stdout.read(fbytes)
                    b1 = p1.stdout.read(fbytes)
                    if len(b0) < fbytes or len(b1) < fbytes:
                        break
                    f0 = np.frombuffer(b0, np.uint8).reshape(lh, lw, 3)
                    f1 = np.frombuffer(b1, np.uint8).reshape(lh, lw, 3)
                    rw = rectilinear_rays(width, height, yaws[k], pitches[k], rolls[k], fovs[k])
                    out = render_frame_flowstate(
                        f0,
                        f1,
                        rw,
                        int(fi),
                        fs,
                        meta,
                        half_blend_deg=3.5,
                        gain_match=True,
                        multiband=True,
                        rolling_shutter=not no_rs,
                    )
                    if grade == "ilog" or lut is not None:
                        out = grade_ilog(out, warmth=warmth, sat=sat, lut=lut)
                    proc.stdin.write(out.tobytes())
                    rendered += 1
            finally:
                p0.wait()
                p1.wait()
    finally:
        if proc.stdin:
            proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg exited with code {ret}")
    return {
        "output": os.path.abspath(output_path),
        "frames": rendered,
        "width": width,
        "height": height,
        "fps": round(fps, 4),
        "duration_s": round(rendered / fps, 3) if fps > 0 else 0.0,
        "mode": mode,
        "rolling_shutter": not no_rs,
    }


def export_resolve_fusion(
    keyframes: list[dict[str, Any]] | str,
    output_path: str,
    fps: float = 25.0,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Export DaVinci Resolve Fusion .setting macro (SphereMap + Shape3D + Camera3D + Renderer3D)."""
    kfs = load_keyframes(keyframes)
    t0 = float(kfs[0]["t"])

    def spline_entries(vals):
        lines = []
        for k, v in zip(kfs, vals):
            fr = int(round((float(k["t"]) - t0) * fps))
            lines.append(f"\t\t\t\t[{fr}] = {{ {v:.4f}, Flags = {{ Linear = false }} }},")
        return "\n".join(lines)

    yaws = [float(k["yaw"]) for k in kfs]
    pitches = [float(k.get("pitch", 0.0)) for k in kfs]
    rolls = [float(k.get("roll", 0.0)) for k in kfs]
    fls = [18.0 / math.tan(math.radians(float(k.get("fov", 80.0))) / 2.0) for k in kfs]

    setting = f"""{{
\tTools = ordered() {{
\t\tReframe360_SphereMap = SphereMap {{
\t\t\tInputs = {{
\t\t\t\tRotation = Input {{ Value = 1, }},
\t\t\t}},
\t\t}},
\t\tReframe360_Sphere = Shape3D {{
\t\t\tInputs = {{
\t\t\t\tShape = Input {{ Value = FuID {{ "SurfaceSphereInputs" }}, }},
\t\t\t\t["SurfaceSphereInputs.Radius"] = Input {{ Value = 100, }},
\t\t\t\t["SurfaceSphereInputs.SubdivisionLevelBase"] = Input {{ Value = 64, }},
\t\t\t\t["SurfaceSphereInputs.SubdivisionLevelHeight"] = Input {{ Value = 48, }},
\t\t\t\tMaterialInput = Input {{ SourceOp = "Reframe360_SphereMap", Source = "MaterialOutput", }},
\t\t\t}},
\t\t}},
\t\tReframe360_Camera = Camera3D {{
\t\t\tInputs = {{
\t\t\t\t["Transform3DOp.Rotate.Y"] = Input {{ SourceOp = "CamYaw", Source = "Value", }},
\t\t\t\t["Transform3DOp.Rotate.X"] = Input {{ SourceOp = "CamPitch", Source = "Value", }},
\t\t\t\t["Transform3DOp.Rotate.Z"] = Input {{ SourceOp = "CamRoll", Source = "Value", }},
\t\t\t\tFLength = Input {{ SourceOp = "CamFL", Source = "Value", }},
\t\t\t\tApertureW = Input {{ Value = 1.4173, }},
\t\t\t\tApertureH = Input {{ Value = {1.4173 * height / width:.4f}, }},
\t\t\t}},
\t\t}},
\t\tCamYaw = BezierSpline {{ KeyFrames = {{
{spline_entries(yaws)}
\t\t}} }},
\t\tCamPitch = BezierSpline {{ KeyFrames = {{
{spline_entries(pitches)}
\t\t}} }},
\t\tCamRoll = BezierSpline {{ KeyFrames = {{
{spline_entries(rolls)}
\t\t}} }},
\t\tCamFL = BezierSpline {{ KeyFrames = {{
{spline_entries(fls)}
\t\t}} }},
\t\tReframe360_Merge = Merge3D {{
\t\t\tInputs = {{
\t\t\t\tSceneInput1 = Input {{ SourceOp = "Reframe360_Sphere", Source = "Output", }},
\t\t\t\tSceneInput2 = Input {{ SourceOp = "Reframe360_Camera", Source = "Output", }},
\t\t\t}},
\t\t}},
\t\tReframe360_Render = Renderer3D {{
\t\t\tInputs = {{
\t\t\t\tWidth = Input {{ Value = {width}, }},
\t\t\t\tHeight = Input {{ Value = {height}, }},
\t\t\t\tSceneInput = Input {{ SourceOp = "Reframe360_Merge", Source = "Output", }},
\t\t\t\tRendererType = Input {{ Value = FuID {{ "RendererOpenGL" }}, }},
\t\t\t}},
\t\t}},
\t}},
}}
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(setting)
    return {
        "output": os.path.abspath(output_path),
        "keyframes": len(kfs),
        "fps": fps,
        "width": width,
        "height": height,
    }


def inspect_cuts(
    input_path: str,
    outdir: str,
    cuts: list[float] | None = None,
) -> dict[str, Any]:
    """Render 0.1s granular cut-boundary strips (-0.3s to +0.3s) plus EBU R128 loudness check."""
    ffmpeg = find_binary("ffmpeg", "FFMPEG")
    cut_times = [float(x) for x in cuts] if cuts else []
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    dur = n_frames / fps

    if not cut_times:
        prev = None
        for fi in range(0, n_frames, 2):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frm = cap.read()
            if not ok:
                break
            sm = cv2.resize(frm, (160, 90)).astype(np.float32)
            if prev is not None:
                diff = float(np.mean(np.abs(sm - prev)))
                if diff > 28.0:
                    t = fi / fps
                    if not cut_times or t - cut_times[-1] > 0.4:
                        cut_times.append(round(t, 2))
            prev = sm

    os.makedirs(outdir, exist_ok=True)
    offsets = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    tw = 240 if fw >= fh else 160
    th = int(round(tw * fh / fw))

    rows_all = []
    for ci, ct in enumerate(cut_times):
        strip = []
        for off in offsets:
            t = float(np.clip(ct + off, 0.0, max(0.0, dur - 0.04)))
            fi = int(np.clip(round(t * fps), 0, max(0, n_frames - 1)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frm = cap.read()
            im = cv2.resize(frm, (tw, th)) if ok else np.zeros((th, tw, 3), np.uint8)
            col = (0, 255, 255) if abs(off) < 0.01 else (220, 220, 220)
            cv2.rectangle(im, (0, 0), (tw, 20), (0, 0, 0), -1)
            cv2.putText(
                im,
                f"C{ci + 1} {off:+.1f}s ({t:.2f}s)",
                (3, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                col,
                1,
                cv2.LINE_AA,
            )
            if abs(off) < 0.01:
                cv2.rectangle(im, (0, 0), (tw - 1, th - 1), (0, 255, 255), 2)
            strip.append(im)
        rows_all.append(np.hstack(strip))
    cap.release()

    per_img = 8
    out_paths = []
    for i in range(0, len(rows_all), per_img):
        sheet = np.vstack(rows_all[i : i + per_img])
        p = os.path.abspath(os.path.join(outdir, f"cuts_0p1s_{i // per_img + 1}.jpg"))
        cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out_paths.append(p)

    eb = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", input_path, "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    summary = [ln.strip() for ln in eb.stderr.splitlines() if ln.strip().startswith(("I:", "LRA:", "Peak:"))][-4:]
    return {
        "cut_times": cut_times,
        "duration_s": round(dur, 3),
        "sheets": out_paths,
        "audio_qc": " | ".join(summary),
    }
