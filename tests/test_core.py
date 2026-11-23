"""Unit and MCP stdio protocol tests for reframe360 (no camera footage required)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from reframe360.core import CubeLUT, eval_keyframes, export_resolve_fusion, grade_ilog, rectilinear_rays
from reframe360.mcp_server import handle_message


def test_keyframes_smootherstep_and_pchip():
    kfs2 = [{"t": 0.0, "yaw": 170.0, "pitch": 0.0, "fov": 90.0}, {"t": 2.0, "yaw": -170.0, "pitch": 10.0, "fov": 70.0}]
    yaws, pitches, rolls, fovs = eval_keyframes(kfs2, np.array([0.0, 1.0, 2.0]))
    assert np.isclose(yaws[0], 170.0)
    assert np.isclose(yaws[1], 180.0)
    assert np.isclose(yaws[2], 190.0)
    assert np.isclose(pitches[1], 5.0)
    assert np.isclose(fovs[1], 80.0)
    assert np.allclose(rolls, 0.0)

    kfs3 = [
        {"t": 0.0, "yaw": 0.0, "pitch": 0.0, "fov": 85.0},
        {"t": 2.0, "yaw": 30.0, "pitch": 5.0, "fov": 80.0},
        {"t": 4.0, "yaw": 60.0, "pitch": 0.0, "fov": 75.0},
    ]
    y3, _, _, _ = eval_keyframes(kfs3, np.array([0.0, 2.0, 4.0]), smooth_sigma_s=0.2, fps=25.0)
    assert y3[0] < y3[1] < y3[2]


def test_rectilinear_rays_unit_norm():
    rays = rectilinear_rays(64, 36, yaw_deg=30.0, pitch_deg=-5.0, roll_deg=2.0, hfov_deg=80.0)
    assert rays.shape == (36, 64, 3)
    norms = np.linalg.norm(rays, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_cube_lut_identity_and_ilog_grade():
    with tempfile.TemporaryDirectory() as td:
        lut_path = Path(td) / "identity.cube"
        lines = ["LUT_3D_SIZE 2"]
        for b in (0.0, 1.0):
            for g in (0.0, 1.0):
                for r in (0.0, 1.0):
                    lines.append(f"{r:.4f} {g:.4f} {b:.4f}")
        lut_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        lut = CubeLUT(str(lut_path))
        img = np.array([[[10, 80, 200], [250, 120, 30]]], dtype=np.uint8)
        out = lut.apply_bgr(img)
        assert np.all(np.abs(out.astype(int) - img.astype(int)) <= 1)

        graded = grade_ilog(img)
        assert graded.shape == img.shape and graded.dtype == np.uint8


def test_export_resolve_fusion():
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "move.setting"
        kfs = [{"t": 1.0, "yaw": 20.0, "pitch": -3.0, "fov": 84.0}, {"t": 5.0, "yaw": 60.0, "pitch": 2.0, "fov": 76.0}]
        res = export_resolve_fusion(kfs, str(out_path), fps=25.0, width=1920, height=1080)
        text = out_path.read_text(encoding="utf-8")
        assert res["keyframes"] == 2
        assert "Reframe360_Camera" in text
        assert "[0] = { 20.0000" in text
        assert "[100] = { 60.0000" in text


def test_mcp_protocol_in_process_and_stdio():
    init = handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}
    )
    assert init["result"]["protocolVersion"] == "2025-03-26"
    assert init["result"]["serverInfo"]["name"] == "reframe360"

    listed = handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = {t["name"] for t in listed["result"]["tools"]}
    assert names == {
        "inspect_clip_metadata",
        "inspect_sphere",
        "reframe_clip",
        "export_resolve_fusion",
        "inspect_cuts",
    }

    with tempfile.TemporaryDirectory() as td:
        out_path = str(Path(td) / "mcp.setting")
        msgs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "export_resolve_fusion",
                    "arguments": {
                        "keyframes": [{"t": 0.0, "yaw": 0.0}, {"t": 2.0, "yaw": 45.0}],
                        "output_path": out_path,
                    },
                },
            },
        ]
        payload = "\n".join(json.dumps(m) for m in msgs) + "\n"
        proc = subprocess.run(
            [sys.executable, "-m", "reframe360.mcp_server"],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        replies = [json.loads(line) for line in proc.stdout.strip().splitlines()]
        assert len(replies) == 3
        assert replies[0]["id"] == 1
        assert len(replies[1]["result"]["tools"]) == 5
        assert replies[2]["id"] == 3
        assert not replies[2]["result"].get("isError")
        assert Path(out_path).exists()


if __name__ == "__main__":
    test_keyframes_smootherstep_and_pchip()
    test_rectilinear_rays_unit_norm()
    test_cube_lut_identity_and_ilog_grade()
    test_export_resolve_fusion()
    test_mcp_protocol_in_process_and_stdio()
    print("All reframe360 tests passed.")
