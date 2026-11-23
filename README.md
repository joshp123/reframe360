# reframe360

An MCP server and CLI for reframing raw Insta360 X6/X5/X4 360° footage (`.lrv` proxies and 8K `.insv` originals) directly from dual-fisheye video—with 1000 Hz IMU horizon lock, rolling-shutter compensation, and DaVinci Resolve Fusion export.

Built so AI agents (or scripts) can inspect a 360° sphere visually, choose camera moves, render stabilized 16:9 or 9:16 shots, and hand keyframes to DaVinci Resolve without clicking through Insta360 Studio.

## What it does

- **Factory MEI dual-fisheye stitching (`insta360_meta.py`):** Reads the Insta360 v3 binary trailer (`offset_v3`), extracts both lenses' 56-token MEI calibration and extrinsics, and raycasts rectilinear frames straight from both fisheye sensors in one pass. Uses symmetric overlap gain matching and 3-level Laplacian pyramid blending to avoid hard stitch seams.
- **1000 Hz IMU horizon lock + rolling shutter:** Reads the embedded 1000 Hz 6-DOF IMU stream (Record `0x03`), exact per-frame exposure timestamps (Records `0x04` and `0x0c`), and vertical readout time (protobuf field 25). Uses centrifugal-gated zero-phase gravity leveling (resistant to fast selfie-stick swings) and 8-band per-row rolling shutter compensation. Supports `world` (locked heading) and `follow` (smooth yaw glide) modes.
- **Visual sphere scouting (`inspect_sphere`):** Renders a horizon-locked equirectangular contact sheet with yaw, pitch, and dual-lens seam overlays, plus an optional rectilinear crop-box preview of proposed keyframes. Over MCP, returns the contact sheet inline as an image.
- **Keyframed reframing (`reframe_clip`):** Smooth camera moves using C2 smootherstep (2 keyframes) or monotone cubic PCHIP with optional Gaussian trajectory smoothing (3+ keyframes). Supports `.cube` 3D LUTs (such as Insta360's official `I-Log` to Rec.709 LUT) or a built-in Rec.2020-aware I-Log preview grade.
- **DaVinci Resolve Fusion export (`export_resolve_fusion`):** Writes a Resolve Fusion `.setting` macro (`SphereMap` → `Shape3D` → `Camera3D` → `Renderer3D` with `BezierSpline` keyframes) for GPU-accelerated finishing.
- **0.1-second cut QC (`inspect_cuts`):** Generates 10 fps filmstrips (`-0.3s` to `+0.3s`) around cut boundaries and checks EBU R128 loudness (`ebur128`).

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH` (or set `FFMPEG` / `FFPROBE` environment variables)

## MCP setup

Run directly with `uvx` (zero manual install):

```json
{
  "mcpServers": {
    "reframe360": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/joshp123/reframe360",
        "reframe360-mcp"
      ]
    }
  }
}
```

Or install into a Python environment and point your MCP client at `reframe360-mcp`:

```bash
pip install git+https://github.com/joshp123/reframe360
```

### MCP tools

| Tool | Purpose |
| --- | --- |
| `inspect_clip_metadata` | Camera model, resolution, layout (`side_by_side` / `dual_track`), fps, duration, IMU rate, rolling shutter ms |
| `inspect_sphere` | Horizon-locked equirectangular contact sheet with yaw/pitch/seam grid + optional keyframe crop box (returns inline JPEG) |
| `reframe_clip` | Render stabilized rectilinear 16:9 or 9:16 MP4 directly from `.lrv` or `.insv` |
| `export_resolve_fusion` | Export a DaVinci Resolve Fusion `.setting` macro with `BezierSpline` camera keyframes |
| `inspect_cuts` | Render 0.1s cut-boundary strips and EBU R128 loudness summary (returns inline JPEGs) |

## CLI usage

```bash
# 1. Read clip metadata and IMU/calibration info
reframe360 clip-info clip.lrv

# 2. Render a horizon-locked 360° sphere sheet with yaw/pitch/seam grid
reframe360 inspect-sphere clip.lrv sphere.jpg --count 6

# 3. Preview proposed keyframes as a rectilinear crop box on the sphere
reframe360 inspect-sphere clip.lrv sphere_check.jpg \
  --keyframes '[{"t":1.0,"yaw":118,"pitch":-4,"fov":86},{"t":7.0,"yaw":148,"pitch":2,"fov":78}]'

# 4. Render a seamless 16:9 (or 9:16) stabilized clip
reframe360 reframe clip.lrv out.mp4 \
  --keyframes '[{"t":1.0,"yaw":118,"pitch":-4,"fov":86},{"t":7.0,"yaw":148,"pitch":2,"fov":78}]' \
  --width 1280 --height 720 --mode world --lut X6_I-Log_To_Rec.709.cube

# 5. Export a DaVinci Resolve Fusion comp
reframe360 export-resolve \
  '[{"t":1.0,"yaw":118,"pitch":-4,"fov":86},{"t":7.0,"yaw":148,"pitch":2,"fov":78}]' \
  move.setting --fps 25 --width 1920 --height 1080

# 6. Inspect cut boundaries at 0.1s resolution + EBU R128 loudness
reframe360 inspect-cuts final.mp4 qc_dir/
```

## License

MIT
