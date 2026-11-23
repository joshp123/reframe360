"""Command-line interface for reframe360."""

from __future__ import annotations

import argparse
import json

from .core import (
    export_resolve_fusion,
    inspect_clip_metadata,
    inspect_cuts,
    inspect_sphere,
    reframe_clip,
)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="reframe360",
        description="FlowState-stabilized Insta360 dual-fisheye 360° reframer and cut QC.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("clip-info", help="Print clip metadata, IMU rate, and MEI calibration JSON")
    p0.add_argument("input")

    p1 = sub.add_parser("inspect-sphere", help="Render horizon-locked equirectangular contact sheet")
    p1.add_argument("input")
    p1.add_argument("output")
    p1.add_argument("--count", type=int, default=6)
    p1.add_argument("--step", type=float, default=None, help="Sample every N seconds")
    p1.add_argument("--times", type=str, default=None, help="Comma-separated sample timestamps (seconds)")
    p1.add_argument("--keyframes", type=str, default=None, help="Overlay exact rectilinear crop box (JSON or path)")
    p1.add_argument("--aspect", type=float, default=16 / 9)
    p1.add_argument("--smooth-pan", type=float, default=0.35)
    p1.add_argument("--mode", choices=["world", "follow"], default="world")
    p1.add_argument("--yaw-smooth", type=float, default=1.2)
    p1.add_argument("--lut", type=str, default=None, help="Optional .cube 3D LUT path")
    p1.add_argument("--no-stab", action="store_true")

    p2 = sub.add_parser("reframe", help="Render stabilized rectilinear 16:9 or 9:16 MP4")
    p2.add_argument("input")
    p2.add_argument("output")
    p2.add_argument("--keyframes", required=True, help="JSON array or file path of {t,yaw,pitch,roll,fov} keyframes")
    p2.add_argument("--width", type=int, default=1280)
    p2.add_argument("--height", type=int, default=720)
    p2.add_argument("--start", type=float, default=None)
    p2.add_argument("--end", type=float, default=None)
    p2.add_argument("--crf", type=int, default=18)
    p2.add_argument("--grade", choices=["ilog", "none"], default="ilog")
    p2.add_argument("--lut", type=str, default=None, help="Optional .cube 3D LUT path")
    p2.add_argument("--warmth", type=float, default=1.0)
    p2.add_argument("--sat", type=float, default=1.12)
    p2.add_argument("--mode", choices=["world", "follow"], default="world")
    p2.add_argument("--yaw-smooth", type=float, default=1.2)
    p2.add_argument("--smooth-pan", type=float, default=0.35, help="Gaussian smoothing (s) for 3+ keyframes")
    p2.add_argument("--no-rs", action="store_true", help="Disable per-row rolling shutter compensation")
    p2.add_argument("--no-stab", action="store_true")

    p3 = sub.add_parser("export-resolve", help="Export DaVinci Resolve Fusion .setting macro")
    p3.add_argument("keyframes", help="JSON array or file path")
    p3.add_argument("output")
    p3.add_argument("--fps", type=float, default=25.0)
    p3.add_argument("--width", type=int, default=1920)
    p3.add_argument("--height", type=int, default=1080)

    p4 = sub.add_parser("inspect-cuts", help="Render 0.1s cut-boundary strips + EBU R128 loudness check")
    p4.add_argument("input")
    p4.add_argument("outdir")
    p4.add_argument("--cuts", type=str, default=None, help="Comma-separated cut times in seconds")

    args = ap.parse_args(argv)

    if args.cmd == "clip-info":
        res = inspect_clip_metadata(args.input)
        print(json.dumps(res, indent=2))
    elif args.cmd == "inspect-sphere":
        times = [float(x) for x in args.times.split(",")] if args.times else None
        res = inspect_sphere(
            input_path=args.input,
            output_path=args.output,
            count=args.count,
            step=args.step,
            times=times,
            keyframes=args.keyframes,
            aspect=args.aspect,
            smooth_pan=args.smooth_pan,
            mode=args.mode,
            yaw_smooth=args.yaw_smooth,
            lut_path=args.lut,
            no_stab=args.no_stab,
        )
        print(f"Wrote {res['output']} ({res['duration_s']:.1f}s, {res['views']} views, mode={res['mode']})")
    elif args.cmd == "reframe":
        res = reframe_clip(
            input_path=args.input,
            output_path=args.output,
            keyframes=args.keyframes,
            width=args.width,
            height=args.height,
            start=args.start,
            end=args.end,
            crf=args.crf,
            grade=args.grade,
            lut_path=args.lut,
            warmth=args.warmth,
            sat=args.sat,
            mode=args.mode,
            yaw_smooth=args.yaw_smooth,
            smooth_pan=args.smooth_pan,
            no_rs=args.no_rs,
            no_stab=args.no_stab,
        )
        print(
            f"Rendered {res['output']} ({res['frames']} frames, {res['width']}x{res['height']} "
            f"@ {res['fps']:.1f}fps, mode={res['mode']}, rs={res['rolling_shutter']})"
        )
    elif args.cmd == "export-resolve":
        res = export_resolve_fusion(
            keyframes=args.keyframes,
            output_path=args.output,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
        print(f"Exported Resolve Fusion comp -> {res['output']}")
    elif args.cmd == "inspect-cuts":
        cuts = [float(x) for x in args.cuts.split(",")] if args.cuts else None
        res = inspect_cuts(args.input, args.outdir, cuts=cuts)
        print(f"Inspected {len(res['cut_times'])} cuts across {res['duration_s']:.2f}s -> {res['sheets']}")
        if res["audio_qc"]:
            print("Audio QC:", res["audio_qc"])


if __name__ == "__main__":
    main()
