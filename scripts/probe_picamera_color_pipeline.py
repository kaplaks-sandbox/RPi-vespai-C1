#!/usr/bin/env python3
"""Probe Picamera2 color pipeline and quantify yellow/blue hue shifts.

This script captures:
1) Direct reference image from `rpicam-still`
2) One frame from Picamera2

Then it evaluates color-order variants:
- no_swap: treat Picamera2 frame as already BGR
- rgb_to_bgr: apply cv2.COLOR_RGB2BGR
- rgb_to_bgr_resize_jpeg: swap + resize + JPEG roundtrip
- no_swap_resize_jpeg: no swap + resize + JPEG roundtrip

Use this to confirm whether yellow->blue shift appears after an RGB/BGR swap.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def hue_metrics_bgr(image_bgr: np.ndarray) -> Dict[str, float]:
    """Compute simple hue metrics focused on yellow and blue regions."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Keep reasonably colorful/visible pixels for robust metrics.
    valid = (s > 60) & (v > 60)
    valid_count = int(np.count_nonzero(valid))

    yellow = valid & (h >= 18) & (h <= 40)
    blue = valid & (h >= 90) & (h <= 135)

    total_px = image_bgr.shape[0] * image_bgr.shape[1]
    if total_px <= 0:
        total_px = 1

    yellow_pct = 100.0 * float(np.count_nonzero(yellow)) / float(total_px)
    blue_pct = 100.0 * float(np.count_nonzero(blue)) / float(total_px)

    if valid_count > 0:
        mean_hue = float(np.mean(h[valid]))
    else:
        mean_hue = float(np.mean(h))

    return {
        "yellow_percent": round(yellow_pct, 3),
        "blue_percent": round(blue_pct, 3),
        "mean_hue": round(mean_hue, 3),
        "valid_color_pixels": valid_count,
    }


def resize_jpeg_roundtrip(image_bgr: np.ndarray, target_size: Tuple[int, int], quality: int) -> np.ndarray:
    resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed during roundtrip")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decode failed during roundtrip")
    return decoded


def capture_direct_reference(output_dir: Path, width: int, height: int) -> Tuple[Path | None, str | None]:
    tool = shutil.which("rpicam-still")
    if tool is None:
        return None, "rpicam-still not found in PATH"

    ref_path = output_dir / "direct_reference_rpicam_still.jpg"
    cmd = [
        tool,
        "-o",
        str(ref_path),
        "--width",
        str(width),
        "--height",
        str(height),
        "--nopreview",
        "--immediate",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, f"rpicam-still failed: {proc.stderr.strip() or proc.stdout.strip()}"
    return ref_path, None


def capture_picamera_frame(width: int, height: int, pixfmt: str, warmup_s: float) -> np.ndarray:
    from picamera2 import Picamera2  # pylint: disable=import-error

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (width, height), "format": pixfmt},
        controls={"FrameRate": 30},
    )
    picam2.configure(config)
    picam2.start()
    try:
        time.sleep(max(0.0, warmup_s))
        frame = picam2.capture_array()
    finally:
        picam2.stop()

    if frame is None:
        raise RuntimeError("Picamera2 returned no frame")

    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"Unsupported Picamera2 frame shape: {frame.shape}")

    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Picamera2 color-order behavior with hue metrics")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--preview-size", default="960x540", help="Resize target for JPEG roundtrip variants")
    parser.add_argument("--jpeg-quality", type=int, default=72)
    parser.add_argument("--picam-format", choices=["BGR888", "RGB888"], default="BGR888")
    parser.add_argument("--warmup", type=float, default=0.8, help="Picamera2 warmup in seconds")
    parser.add_argument("--output-dir", default="monitor/color_probe")
    args = parser.parse_args()

    try:
        pw, ph = args.preview_size.lower().split("x", 1)
        preview_size = (int(pw), int(ph))
    except Exception as error:
        raise SystemExit(f"Invalid --preview-size value '{args.preview_size}': {error}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, float]] = {}
    files = {}

    ref_path, ref_error = capture_direct_reference(output_dir, args.width, args.height)
    if ref_path:
        ref_img = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        if ref_img is not None:
            results["direct_reference_rpicam_still"] = hue_metrics_bgr(ref_img)
            files["direct_reference_rpicam_still"] = str(ref_path)
        else:
            ref_error = f"Failed to load reference image: {ref_path}"

    try:
        native = capture_picamera_frame(args.width, args.height, args.picam_format, args.warmup)
    except Exception as error:
        print(json.dumps({"error": f"Picamera2 capture failed: {error}", "ref_error": ref_error}, indent=2))
        return 1

    native_path = output_dir / "picamera_native_frame.png"
    cv2.imwrite(str(native_path), native)
    files["picamera_native_frame"] = str(native_path)

    no_swap = native.copy()
    swapped = cv2.cvtColor(native, cv2.COLOR_RGB2BGR)

    no_swap_path = output_dir / "variant_no_swap.png"
    swapped_path = output_dir / "variant_rgb_to_bgr.png"
    cv2.imwrite(str(no_swap_path), no_swap)
    cv2.imwrite(str(swapped_path), swapped)
    files["variant_no_swap"] = str(no_swap_path)
    files["variant_rgb_to_bgr"] = str(swapped_path)

    swap_jpeg = resize_jpeg_roundtrip(swapped, preview_size, args.jpeg_quality)
    no_swap_jpeg = resize_jpeg_roundtrip(no_swap, preview_size, args.jpeg_quality)

    swap_jpeg_path = output_dir / "variant_rgb_to_bgr_resize_jpeg.jpg"
    no_swap_jpeg_path = output_dir / "variant_no_swap_resize_jpeg.jpg"
    cv2.imwrite(str(swap_jpeg_path), swap_jpeg)
    cv2.imwrite(str(no_swap_jpeg_path), no_swap_jpeg)
    files["variant_rgb_to_bgr_resize_jpeg"] = str(swap_jpeg_path)
    files["variant_no_swap_resize_jpeg"] = str(no_swap_jpeg_path)

    results["variant_no_swap"] = hue_metrics_bgr(no_swap)
    results["variant_rgb_to_bgr"] = hue_metrics_bgr(swapped)
    results["variant_rgb_to_bgr_resize_jpeg"] = hue_metrics_bgr(swap_jpeg)
    results["variant_no_swap_resize_jpeg"] = hue_metrics_bgr(no_swap_jpeg)

    output = {
        "settings": {
            "width": args.width,
            "height": args.height,
            "preview_size": f"{preview_size[0]}x{preview_size[1]}",
            "jpeg_quality": args.jpeg_quality,
            "picam_format": args.picam_format,
            "warmup": args.warmup,
        },
        "ref_error": ref_error,
        "files": files,
        "metrics": results,
    }

    report_path = output_dir / "probe_report.json"
    report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("Color probe complete")
    print(f"Report: {report_path}")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
