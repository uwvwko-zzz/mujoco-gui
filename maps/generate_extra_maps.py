#!/usr/bin/env python3
"""Generate portable terrain-only MJCF files used by the runtime map picker."""

from pathlib import Path
import math
import random
import struct
import xml.etree.ElementTree as ET
import zlib


MAP_DIR = Path(__file__).resolve().parent


def vec(values):
    return " ".join(f"{value:.6g}" for value in values)


def root_for(model):
    root = ET.Element("mujoco", {"model": model})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d", "name": "groundplane", "builtin": "checker",
            "mark": "edge", "rgb1": "0.18 0.24 0.30",
            "rgb2": "0.08 0.12 0.16", "markrgb": "0.7 0.7 0.7",
            "width": "300", "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "groundplane", "texture": "groundplane",
            "texuniform": "true", "texrepeat": "8 8", "reflectance": "0.05",
        },
    )
    return root, asset, ET.SubElement(root, "worldbody")


def geom(world, **attrs):
    return ET.SubElement(world, "geom", {key: str(value) for key, value in attrs.items()})


def floor(world, z=0.0):
    geom(
        world, name="floor", type="plane", pos=f"0 0 {z:g}",
        size="0 0 0.05", material="groundplane",
    )


def write(root, filename):
    if hasattr(ET, "indent"):
        ET.indent(root, space="  ")
    ET.ElementTree(root).write(MAP_DIR / filename, encoding="utf-8", xml_declaration=True)


def write_gray_png(path, rows):
    """Write an 8-bit grayscale PNG using only the Python standard library."""
    height, width = len(rows), len(rows[0])

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    scanlines = b"".join(b"\x00" + bytes(row) for row in rows)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def generate_gap_course():
    root, _, world = root_for("gap jump course")
    # The low safety floor makes the gaps real while preventing an endless fall.
    geom(world, name="safety_floor", type="plane", pos="0 0 -1.2", size="0 0 0.05", rgba="0.08 0.10 0.14 1")
    x = -0.75
    lengths = [1.5, 1.0, 1.2, 0.9, 1.3, 1.0, 1.6]
    gaps = [0.20, 0.30, 0.40, 0.50, 0.60, 0.75]
    for index, length in enumerate(lengths):
        center = x + length / 2
        geom(
            world, name=f"platform_{index}", type="box",
            pos=vec((center, 0, -0.04)), size=vec((length / 2, 1.0, 0.04)),
            rgba=f"{0.18 + index * 0.025:.3f} {0.42 + index * 0.02:.3f} 0.62 1",
        )
        x += length
        if index < len(gaps):
            x += gaps[index]
    write(root, "gap_jump.xml")


def generate_hurdles():
    root, _, world = root_for("hurdle course")
    floor(world)
    heights = [0.12, 0.16, 0.20, 0.24, 0.28, 0.32]
    for index, height in enumerate(heights):
        x = 1.5 + index * 1.15
        geom(
            world, name=f"hurdle_{index}", type="box",
            pos=vec((x, 0, height / 2)), size=vec((0.045, 0.85, height / 2)),
            rgba="0.95 0.42 0.16 1",
        )
        # High contrast side markers make obstacle height easier to see in the browser.
        for side in (-1, 1):
            geom(
                world, name=f"marker_{index}_{side}", type="cylinder",
                pos=vec((x, side * 0.9, 0.3)), size="0.035 0.3",
                rgba="0.95 0.78 0.18 1", contype="0", conaffinity="0",
            )
    write(root, "hurdles.xml")


def generate_suspended_steps():
    root, _, world = root_for("suspended stepping course")
    geom(world, name="safety_floor", type="plane", pos="0 0 -1.0", size="0 0 0.05", rgba="0.06 0.08 0.12 1")
    geom(world, name="start", type="box", pos="0 0 -0.05", size="0.8 0.8 0.05", rgba="0.18 0.52 0.32 1")
    offsets = [0.0, 0.28, -0.24, 0.35, -0.32, 0.16, -0.20, 0.30, 0.0]
    heights = [0.05, 0.12, 0.20, 0.14, 0.28, 0.18, 0.34, 0.22, 0.10]
    for index, (y, z) in enumerate(zip(offsets, heights)):
        x = 1.25 + index * 0.72
        half_x = 0.24 if index < 6 else 0.20
        half_y = 0.34 if index < 6 else 0.28
        geom(
            world, name=f"step_{index}", type="box",
            pos=vec((x, y, z)), size=vec((half_x, half_y, 0.07)),
            rgba=f"0.22 {0.48 + index * 0.025:.3f} 0.72 1",
        )
    geom(world, name="finish", type="box", pos="8.1 0 0.05", size="0.9 0.9 0.08", rgba="0.20 0.62 0.34 1")
    write(root, "suspended_steps.xml")


def generate_rough_perlin():
    root, asset, world = root_for("perlin style rough terrain")
    # MuJoCo maps hfield rows to world X and columns to world Y. Therefore the
    # PNG is intentionally tall: 160 X samples (rows) by 64 Y samples (columns).
    x_samples, y_samples = 160, 64
    rows = []
    for ix in range(x_samples):
        row = []
        x = ix / (x_samples - 1) * 11.0
        for iy in range(y_samples):
            y = (iy / (y_samples - 1) - 0.5) * 4.4
            envelope = min(1.0, max(0.0, (x - 1.4) / 1.5))
            value = envelope * (
                0.47 * math.sin(0.72 * x + 0.35 * math.sin(0.85 * y))
                + 0.27 * math.sin(1.63 * x - 1.17 * y)
                + 0.15 * math.sin(3.21 * x + 2.05 * y)
            )
            # 128 maps to z=0 after the hfield is shifted down by 0.125 m.
            row.append(max(0, min(255, round(128 + 115 * value))))
        rows.append(row)
    # MuJoCo reads image rows bottom-to-top in world X coordinates.
    rows.reverse()
    write_gray_png(MAP_DIR / "imgs/perlin_rough.png", rows)
    ET.SubElement(
        asset, "hfield",
        {"name": "rough", "file": "imgs/perlin_rough.png", "size": "5.5 2.2 0.25 0.05"},
    )
    # Start on a regular box so the locomotion policy establishes its gait
    # before crossing onto the triangulated heightfield at x=1.2 m.
    geom(
        world, name="start_ground", type="box",
        pos="0.1 0 -0.05", size="1.1 2.2 0.05",
        rgba="0.24 0.50 0.29 1",
    )
    geom(
        world, name="perlin_rough", type="hfield", hfield="rough",
        pos="6.7 0 -0.125", rgba="0.24 0.50 0.29 1",
    )
    write(root, "perlin_rough.xml")


def generate_dynamic_obstacles():
    root, _, world = root_for("dynamic random obstacle course")
    floor(world)
    rng = random.Random(20260817)
    for index in range(14):
        sx = rng.uniform(0.10, 0.28)
        sy = rng.uniform(0.16, 0.48)
        sz = rng.uniform(0.08, 0.28)
        geom(
            world, name=f"obstacle_{index}", type="box",
            pos=vec((1.4 + index * 0.52, rng.uniform(-1.15, 1.15), sz)),
            size=vec((sx, sy, sz)), euler=vec((0, 0, rng.uniform(-35, 35))),
            rgba=f"{rng.uniform(0.45, 0.90):.3f} {rng.uniform(0.25, 0.65):.3f} {rng.uniform(0.12, 0.42):.3f} 1",
        )
    write(root, "dynamic_obstacles.xml")


def main():
    generate_gap_course()
    generate_hurdles()
    generate_suspended_steps()
    generate_rough_perlin()
    generate_dynamic_obstacles()


if __name__ == "__main__":
    main()
