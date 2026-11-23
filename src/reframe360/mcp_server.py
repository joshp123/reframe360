"""Zero-dependency Model Context Protocol (MCP) stdio server for reframe360.

Speaks newline-delimited JSON-RPC 2.0 over stdin/stdout (MCP 2024-11-05 / 2025-03-26).
"""

from __future__ import annotations

import base64
import json
import sys
import traceback
from typing import Any

import cv2

from . import __version__
from .core import (
    export_resolve_fusion,
    inspect_clip_metadata,
    inspect_cuts,
    inspect_sphere,
    reframe_clip,
)

SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

KEYFRAME_SCHEMA = {
    "type": "array",
    "description": "Ordered camera keyframes: [{t, yaw, pitch, roll, fov, ease}].",
    "items": {
        "type": "object",
        "properties": {
            "t": {"type": "number", "description": "Timestamp in seconds."},
            "yaw": {"type": "number", "description": "Yaw in degrees (-180 to 180; 0 is camera front)."},
            "pitch": {"type": "number", "description": "Pitch in degrees (+ up, - down). Default 0."},
            "roll": {"type": "number", "description": "Roll in degrees. Default 0."},
            "fov": {"type": "number", "description": "Horizontal field of view in degrees. Default 80."},
            "ease": {
                "type": "string",
                "enum": ["smootherstep", "cubic", "linear"],
                "description": "Easing for 2-keyframe moves. Default smootherstep.",
            },
        },
        "required": ["t", "yaw"],
    },
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "inspect_clip_metadata",
        "description": (
            "Read Insta360 .lrv or .insv clip metadata: camera model, resolution, layout "
            "(side_by_side or dual_track), fps, duration, 1000 Hz IMU sample count, exposure "
            "timestamp count, rolling shutter readout time, and MEI lens calibration."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "Path to .lrv or .insv file."},
            },
            "required": ["input_path"],
        },
    },
    {
        "name": "inspect_sphere",
        "description": (
            "Render a FlowState horizon-locked equirectangular contact sheet with yaw/pitch/seam grid "
            "so an agent can visually pick reframing angles. Optionally overlays the exact rectilinear "
            "viewport crop box for proposed keyframes and returns the contact sheet image inline."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "Path to Insta360 .lrv or .insv clip."},
                "output_path": {"type": "string", "description": "Output JPEG path for the contact sheet."},
                "count": {"type": "integer", "description": "Number of evenly spaced views (default 6)."},
                "step": {"type": "number", "description": "Sample every N seconds (overrides count)."},
                "times": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Explicit timestamps in seconds (overrides step/count).",
                },
                "keyframes": KEYFRAME_SCHEMA,
                "aspect": {
                    "type": "number",
                    "description": "Viewport aspect ratio for crop-box overlay (e.g. 1.7778 for 16:9, 0.5625 for 9:16).",
                },
                "smooth_pan": {"type": "number", "description": "Gaussian smoothing in seconds for 3+ keyframes (default 0.35)."},
                "mode": {
                    "type": "string",
                    "enum": ["world", "follow"],
                    "description": "world: lock heading in world frame; follow: smoothly glide with physical yaw.",
                },
                "yaw_smooth": {"type": "number", "description": "Yaw-follow smoothing sigma in seconds (default 1.2)."},
                "lut_path": {"type": "string", "description": "Optional .cube 3D LUT file path."},
                "no_stab": {"type": "boolean", "description": "Disable IMU horizon stabilization."},
                "return_image": {
                    "type": "boolean",
                    "description": "Return the contact sheet inline as a base64 JPEG image content block (default true).",
                },
            },
            "required": ["input_path", "output_path"],
        },
    },
    {
        "name": "reframe_clip",
        "description": (
            "Render a FlowState-stabilized (1000 Hz SLERP + rolling shutter), keyframed rectilinear "
            "16:9 or 9:16 MP4 directly from raw dual-fisheye Insta360 .lrv or .insv footage."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "Path to Insta360 .lrv or .insv clip."},
                "output_path": {"type": "string", "description": "Output MP4 path."},
                "keyframes": KEYFRAME_SCHEMA,
                "width": {"type": "integer", "description": "Output width in pixels (default 1280)."},
                "height": {"type": "integer", "description": "Output height in pixels (default 720)."},
                "start": {"type": "number", "description": "Trim start in seconds (defaults to first keyframe t)."},
                "end": {"type": "number", "description": "Trim end in seconds (defaults to last keyframe t)."},
                "crf": {"type": "integer", "description": "x264 CRF quality (default 18)."},
                "grade": {
                    "type": "string",
                    "enum": ["ilog", "none"],
                    "description": "Apply built-in I-Log preview grade or leave raw (default ilog).",
                },
                "lut_path": {"type": "string", "description": "Optional .cube 3D LUT path (overrides built-in I-Log curve)."},
                "warmth": {"type": "number", "description": "Warmth multiplier for built-in grade (default 1.0)."},
                "sat": {"type": "number", "description": "Saturation multiplier for built-in grade (default 1.12)."},
                "mode": {"type": "string", "enum": ["world", "follow"], "description": "Yaw stabilization mode (default world)."},
                "yaw_smooth": {"type": "number", "description": "Yaw-follow smoothing sigma in seconds (default 1.2)."},
                "smooth_pan": {"type": "number", "description": "Gaussian smoothing in seconds for 3+ keyframes (default 0.35)."},
                "no_rs": {"type": "boolean", "description": "Disable 8-band vertical rolling shutter compensation."},
                "no_stab": {"type": "boolean", "description": "Disable IMU horizon stabilization."},
            },
            "required": ["input_path", "output_path", "keyframes"],
        },
    },
    {
        "name": "export_resolve_fusion",
        "description": (
            "Export a DaVinci Resolve Fusion .setting macro (SphereMap -> Shape3D -> Camera3D -> Renderer3D "
            "with BezierSpline keyframes) for GPU-accelerated finishing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "keyframes": KEYFRAME_SCHEMA,
                "output_path": {"type": "string", "description": "Destination .setting file path."},
                "fps": {"type": "number", "description": "Timeline frame rate (default 25.0)."},
                "width": {"type": "integer", "description": "Output width in pixels (default 1920)."},
                "height": {"type": "integer", "description": "Output height in pixels (default 1080)."},
            },
            "required": ["keyframes", "output_path"],
        },
    },
    {
        "name": "inspect_cuts",
        "description": (
            "Render 0.1-second (10 fps) filmstrips from -0.3s to +0.3s around every cut boundary "
            "in an edited video and run an EBU R128 loudness check."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "Path to edited video file."},
                "outdir": {"type": "string", "description": "Output directory for cut-boundary JPEG sheets."},
                "cuts": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional explicit cut timestamps in seconds (auto-detected if omitted).",
                },
                "return_images": {
                    "type": "boolean",
                    "description": "Return generated sheets inline as base64 JPEG content blocks (default true).",
                },
            },
            "required": ["input_path", "outdir"],
        },
    },
]


def _encode_image_block(path: str, max_dim: int = 1600) -> dict[str, Any]:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read rendered image at {path}")
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError(f"Failed to encode JPEG for {path}")
    return {
        "type": "image",
        "data": base64.b64encode(buf.tobytes()).decode("ascii"),
        "mimeType": "image/jpeg",
    }


def handle_tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    try:
        if name == "inspect_clip_metadata":
            res = inspect_clip_metadata(args["input_path"])
            return {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}

        if name == "inspect_sphere":
            return_image = bool(args.get("return_image", True))
            res = inspect_sphere(
                input_path=args["input_path"],
                output_path=args["output_path"],
                count=int(args.get("count", 6)),
                step=args.get("step"),
                times=args.get("times"),
                keyframes=args.get("keyframes"),
                aspect=float(args.get("aspect", 16 / 9)),
                smooth_pan=float(args.get("smooth_pan", 0.35)),
                mode=str(args.get("mode", "world")),
                yaw_smooth=float(args.get("yaw_smooth", 1.2)),
                lut_path=args.get("lut_path"),
                no_stab=bool(args.get("no_stab", False)),
            )
            content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(res, indent=2)}]
            if return_image:
                content.append(_encode_image_block(res["output"]))
            return {"content": content}

        if name == "reframe_clip":
            res = reframe_clip(
                input_path=args["input_path"],
                output_path=args["output_path"],
                keyframes=args["keyframes"],
                width=int(args.get("width", 1280)),
                height=int(args.get("height", 720)),
                start=args.get("start"),
                end=args.get("end"),
                crf=int(args.get("crf", 18)),
                grade=str(args.get("grade", "ilog")),
                lut_path=args.get("lut_path"),
                warmth=float(args.get("warmth", 1.0)),
                sat=float(args.get("sat", 1.12)),
                mode=str(args.get("mode", "world")),
                yaw_smooth=float(args.get("yaw_smooth", 1.2)),
                smooth_pan=float(args.get("smooth_pan", 0.35)),
                no_rs=bool(args.get("no_rs", False)),
                no_stab=bool(args.get("no_stab", False)),
            )
            return {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}

        if name == "export_resolve_fusion":
            res = export_resolve_fusion(
                keyframes=args["keyframes"],
                output_path=args["output_path"],
                fps=float(args.get("fps", 25.0)),
                width=int(args.get("width", 1920)),
                height=int(args.get("height", 1080)),
            )
            return {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}

        if name == "inspect_cuts":
            return_images = bool(args.get("return_images", True))
            res = inspect_cuts(
                input_path=args["input_path"],
                outdir=args["outdir"],
                cuts=args.get("cuts"),
            )
            content = [{"type": "text", "text": json.dumps(res, indent=2)}]
            if return_images:
                for sheet_path in res["sheets"][:4]:
                    content.append(_encode_image_block(sheet_path))
            return {"content": content}

        return {
            "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
            "isError": True,
        }
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return {
            "content": [{"type": "text", "text": f"Error: {exc}"}],
            "isError": True,
        }


def handle_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    req_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}

    # Notifications have no id and expect no reply
    if req_id is None:
        return None

    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "reframe360", "version": __version__},
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        result = handle_tool_call(name, arguments)
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue

        resp = handle_message(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
