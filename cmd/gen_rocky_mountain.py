#!/usr/bin/env python3
# Copyright (c) 2025, DYROS.
#
# tocabi variant of the rocky-mountain scene generator. Reads absolute world-frame
# swing-foot targets from cmd/global_command.csv (same row-oriented layout as
# gen_scene.py) and writes a "rocky mountain" into stair_simulation_scene.xml:
# instead of isolated stepping-stone cubes, the space between footholds is filled
# with a dense field of rock-colored columns whose heights are interpolated from
# the footstep heights, so the whole path looks like one continuous mountain climb.
#
# Layers generated:
#   1. startterrain        - spawn platform under the robot (same role as gen_scene.py)
#   2. footstep_ground     - large ground plane at the lowest terrain level
#   3. rock_XXXX           - columns forming the mountain body, either boxes
#                            (--rock-shape box, yaw-jittered) or cylinders
#                            (--rock-shape cylinder, basalt-like), on a square or
#                            hexagonal lattice (--layout). The ground behind the
#                            spawn point is left bare out to --start-clear so the
#                            robot starts on the platform rather than inside the
#                            rocks, while the rock ahead of it (where it actually
#                            walks) is untouched. Cells within --flush-radius of a
#                            stepping stone are set to exactly that stone's landing
#                            height (minus a 2 mm anti-z-fighting drop), so the
#                            stones blend flush into the surrounding rock. Beyond
#                            that, cell tops fade over --flush-blend to `clearance`
#                            below the interpolated footstep surface so the filler
#                            terrain never obstructs the swing foot.
#   4. cube_XX             - mocap stepping stones (same as gen_scene.py). Runtime
#                            still overwrites mocap_pos from the CSV; XY of the
#                            baked XML is the CSV target shifted by --offset-x/y
#                            in that foot's yaw frame (forward=x, left=y); Z uses
#                            --offset-z in world up. Boxes are yaw-aligned.
#
# The terrain only fills a corridor around the footstep path (distance-to-path
# falloff), so it reads as a mountain ridge rather than a filled field.
# Deterministic per-cell noise + color variation give a natural rocky look.
#
# Column geometry is two independent numbers: --spacing is how far apart the
# columns sit, --rock-width is how wide each one is. Making the width larger
# than the spacing overlaps neighbors and hides the seams. (--overlap is the
# older relative way to say the same thing: width = spacing * overlap.)
#
# Round columns do not tile a square lattice, so the width must exceed the
# spacing by 1.414x (grid) or 1.155x (hex) or gaps open up between neighbors.
# The script computes that threshold and warns, in millimeters, when the chosen
# width is under it.
#
# Usage:
#   python3 cmd/gen_rocky_mountain.py
#   python3 cmd/gen_rocky_mountain.py --spacing 0.15 --rock-width 0.18
#   python3 cmd/gen_rocky_mountain.py --spacing 0.10 --rock-width 0.13 --clearance 0.03
#   python3 cmd/gen_rocky_mountain.py --rock-shape cylinder --layout hex --spacing 0.14 --rock-width 0.17
#   python3 cmd/gen_rocky_mountain.py --csv cmd/global_command.csv --start R

import argparse
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from gen_scene import (  # noqa: E402
    CUBE_HX,
    CUBE_HY,
    CUBE_HZ,
    DEFAULT_CSV,
    SCENE_XML,
    foot_for_step,
    read_global_csv,
    yaw_to_quat,
)

COLOR_STONE = "0.6 0.5 0.4 1"
COLOR_GROUND = "0.6 0.5 0.4 1"
COLOR_PLATFORM = COLOR_STONE
ROCK_BASE_RGB = (0.6, 0.5, 0.4)

# Match gen_scene.py startterrain: top at z=0, centered at x=0.1 (spawn feet).
DEFAULT_PLATFORM_SIZE = (0.16, 0.30)
DEFAULT_PLATFORM_XY = (0.1, 0.0)
DEFAULT_PLATFORM_TOP = 0.0
DEFAULT_PLATFORM_HALF_HEIGHT = 0.05


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------

def rows_from_targets(targets, start):
    """Attach alternating swing-foot labels to gen_scene targets."""
    out = []
    for i, t in enumerate(targets):
        out.append({
            "foot": foot_for_step(i, start),
            "x": t["x"],
            "y": t["y"],
            "z": t["z"],
            "yaw": t["yaw"],
        })
    return out


def yaw_frame_offset(x, y, yaw, ox, oy):
    """Translate (x, y) by (ox, oy) expressed in the foot yaw frame.

    Foot frame: +x forward along yaw, +y left. World = R(yaw) * local.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * ox - s * oy, y + s * ox + c * oy


def cube_body(i, x, y, z, yaw, rgba):
    """Mocap stepping stone: same height-shaded rocky color as mountain cells."""
    qw, qx, qy, qz = yaw_to_quat(yaw)
    return (
        f'        <body name="cube_{i:02d}" mocap="true" '
        f'pos="{x:.4f} {y:.4f} {z - CUBE_HZ:.4f}" '
        f'quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}">'
        f'<geom type="box" group="3" size="{CUBE_HX:.4f} {CUBE_HY:.4f} {CUBE_HZ:.4f}" '
        f'{rgba}/></body>'
    )


def stone_landing_poses(rows, off):
    """Stepping-stone landings: XY offset in each foot's yaw frame, Z world-up."""
    ox, oy, oz = off
    out = []
    for r in rows:
        sx, sy = yaw_frame_offset(r["x"], r["y"], r["yaw"], ox, oy)
        out.append({
            "sx": sx,
            "sy": sy,
            "sz": r["z"] + oz,
            "yaw": r["yaw"],
            "foot": r["foot"],
        })
    return out


# ---------------------------------------------------------------------------
# deterministic per-cell noise (order-independent, no RNG state)
# ---------------------------------------------------------------------------

def cell_hash(i, j, seed, salt=0.0):
    """Deterministic pseudo-random value in [0, 1) for grid cell (i, j)."""
    v = math.sin(i * 12.9898 + j * 78.233 + seed * 37.719 + salt * 4.581) * 43758.5453
    return v - math.floor(v)


def rock_rgba(top, z_lo, z_span, i, j, seed, salt=2.0):
    """Height-shaded rock color with per-cell variation → 'rgba=\"...\"'."""
    rel = (top - z_lo) / z_span
    shade = 0.72 + 0.38 * max(0.0, min(1.0, rel))
    var = (cell_hash(i, j, seed, salt=salt) - 0.5) * 0.12
    r = max(0.0, min(1.0, (ROCK_BASE_RGB[0] + var) * shade))
    g = max(0.0, min(1.0, (ROCK_BASE_RGB[1] + var * 0.9) * shade))
    b = max(0.0, min(1.0, (ROCK_BASE_RGB[2] + var * 0.8) * shade))
    return f'rgba="{r:.3f} {g:.3f} {b:.3f} 1"'


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def point_segment_dist(px, py, ax, ay, bx, by):
    """2D distance from point P to segment AB."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return math.hypot(apx, apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def path_distance(px, py, path):
    """Min 2D distance from (px, py) to the footstep polyline."""
    d = float("inf")
    for k in range(len(path) - 1):
        d = min(d, point_segment_dist(px, py, path[k][0], path[k][1], path[k + 1][0], path[k + 1][1]))
    return d


def idw_height(px, py, points, power=2.0, eps=0.02):
    """Inverse-distance-weighted surface height at (px, py)."""
    wsum = 0.0
    zsum = 0.0
    for (x, y, z) in points:
        d2 = (px - x) ** 2 + (py - y) ** 2 + eps * eps
        w = 1.0 / (d2 ** (power / 2.0))
        wsum += w
        zsum += w * z
    return zsum / wsum


# ---------------------------------------------------------------------------
# column lattice & gap-free coverage
# ---------------------------------------------------------------------------

HEX_ROW_DY = math.sqrt(3.0) / 2.0  # triangular-lattice row pitch, in cell units


def lattice_cells(x0, y0, x1, y1, cell, layout):
    """Yield (i, j, cx, cy) column centers covering the [x0,x1]x[y0,y1] box."""
    row_dy = cell * (HEX_ROW_DY if layout == "hex" else 1.0)
    nx = max(1, int(math.ceil((x1 - x0) / cell)) + (1 if layout == "hex" else 0))
    ny = max(1, int(math.ceil((y1 - y0) / row_dy)))
    for j in range(ny):
        cy = y0 + (j + 0.5) * row_dy
        x_off = 0.5 * cell if (layout == "hex" and j % 2) else 0.0
        for i in range(nx):
            yield i, j, x0 + (i + 0.5) * cell + x_off, cy


def column_width(args):
    """Effective rock column width [m]: box edge-to-edge, or cylinder diameter."""
    if args.rock_width is not None:
        return args.rock_width
    return args.spacing * args.overlap


def required_width(shape, layout, spacing):
    """Narrowest gap-free column width [m] for a lattice at this spacing."""
    return spacing * required_overlap(shape, layout)


def required_overlap(shape, layout):
    """Smallest --overlap that leaves no gaps between neighboring columns."""
    if shape == "box":
        return 1.0
    return 2.0 / math.sqrt(3.0) if layout == "hex" else math.sqrt(2.0)


# ---------------------------------------------------------------------------
# terrain generation
# ---------------------------------------------------------------------------

def build_rock_cells(stones, args, z_ground):
    """Lattice of columns approximating a rocky mountain under the footsteps."""
    stone_xyz = [(s["sx"], s["sy"], s["sz"]) for s in stones]
    plat_x, plat_y = args.platform_xy
    points = [(plat_x, plat_y, args.platform_top)] + stone_xyz
    path = [(p[0], p[1]) for p in points]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    z_lo, z_hi = min(zs), max(zs)
    z_span = max(z_hi - z_lo, 1e-6)

    cell = args.spacing
    half = 0.5 * column_width(args)
    margin = args.terrain_margin
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin

    w0, w1 = args.corridor
    lines = []
    n_cells = 0
    reach = half * math.sqrt(2.0) if args.rock_shape == "box" else half
    start_keep_out = args.start_clear + reach

    for i, j, cx, cy in lattice_cells(x0, y0, x1, y1, cell, args.layout):
        if (
            args.start_clear > 0.0
            and cx < plat_x + args.start_clear_front
            and math.hypot(cx - plat_x, cy - plat_y) < start_keep_out
        ):
            continue

        d = path_distance(cx, cy, path)
        if d >= w1:
            continue
        if d <= w0:
            falloff = 1.0
        else:
            falloff = 0.5 * (1.0 + math.cos(math.pi * (d - w0) / (w1 - w0)))

        z_surf = idw_height(cx, cy, points, power=args.idw_power)
        u = cell_hash(i, j, args.seed)
        base_top = z_surf - args.clearance - args.noise * u

        r0 = args.flush_radius
        r1 = r0 + args.flush_blend
        z_in = [z for (sx_, sy_, z) in stone_xyz if (cx - sx_) ** 2 + (cy - sy_) ** 2 <= r0 * r0]
        if z_in:
            top = min(z_in) - args.flush_drop
        else:
            ds_min, z_near = min((math.hypot(cx - sx_, cy - sy_), z) for (sx_, sy_, z) in stone_xyz)
            if ds_min < r1:
                t = 0.5 * (1.0 - math.cos(math.pi * (ds_min - r0) / args.flush_blend))
                top = (1.0 - t) * (z_near - args.flush_drop) + t * base_top
            else:
                top = base_top
        top = z_ground + (top - z_ground) * falloff

        h = top - z_ground
        if h < args.min_height:
            continue
        n_cells += 1

        hz = 0.5 * h
        zc = z_ground + hz
        rgba = rock_rgba(top, z_lo, z_span, i, j, args.seed)

        if args.rock_shape == "cylinder":
            lines.append(
                f'        <geom name="rock_{n_cells:04d}" type="cylinder" group="4" '
                f'size="{half:.4f} {hz:.4f}" pos="{cx:.4f} {cy:.4f} {zc:.4f}" {rgba}/>'
            )
        else:
            yaw = (cell_hash(i, j, args.seed, salt=1.0) - 0.5) * 2.0 * args.max_cell_yaw
            qw, qx, qy, qz = yaw_to_quat(yaw)
            lines.append(
                f'        <geom name="rock_{n_cells:04d}" type="box" group="4" '
                f'size="{half:.4f} {half:.4f} {hz:.4f}" pos="{cx:.4f} {cy:.4f} {zc:.4f}" '
                f'quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}" {rgba}/>'
            )
    return lines


# ---------------------------------------------------------------------------
# scene assembly
# ---------------------------------------------------------------------------

def build_worldbody(rows, off, args):
    stones = stone_landing_poses(rows, off)
    zs_top = [s["sz"] for s in stones]
    z_min = min(zs_top)
    platform_top = args.platform_top
    plat_hx, plat_hy = args.platform_size
    plat_x, plat_y = args.platform_xy

    if z_min >= platform_top:
        z_ground = platform_top
        plat_hz = DEFAULT_PLATFORM_HALF_HEIGHT
        plat_zc = platform_top - plat_hz
    else:
        z_ground = z_min
        plat_hz = (platform_top - z_min) * 0.5
        plat_zc = z_min + plat_hz

    xs = [s["sx"] for s in stones]
    ys = [s["sy"] for s in stones]
    plane_cx = 0.5 * (min(xs) + max(xs))
    plane_cy = 0.5 * (min(ys) + max(ys))
    plane_hx = 0.5 * (max(xs) - min(xs)) + args.plane_margin
    plane_hy = 0.5 * (max(ys) - min(ys)) + args.plane_margin

    lines = [
        f'        <geom name="startterrain" type="box" group="3" '
        f'size="{plat_hx:.4f} {plat_hy:.4f} {plat_hz:.4f}" '
        f'pos="{plat_x:.4f} {plat_y:.4f} {plat_zc:.4f}" rgba="{COLOR_PLATFORM}"/>',
        f'        <geom name="footstep_ground" type="plane" '
        f'pos="{plane_cx:.4f} {plane_cy:.4f} {z_ground:.4f}" '
        f'size="{plane_hx:.4f} {plane_hy:.4f} 0.1" rgba="{COLOR_GROUND}" group="3"/>',
    ]

    rock_lines = build_rock_cells(stones, args, z_ground)
    lines += rock_lines

    # mocap cubes stay at the (possibly offset) landings so mujoco_ros can find cube_XX
    z_lo = min(zs_top + [platform_top])
    z_span = max(max(zs_top) - z_lo, 1e-6)
    cubes = []
    for i, s in enumerate(stones):
        rgba = rock_rgba(s["sz"], z_lo, z_span, i, 0, args.seed, salt=3.0)
        cubes.append(cube_body(i + 1, s["sx"], s["sy"], s["sz"], s["yaw"], rgba))

    inner = "\n".join(lines) + "\n\n" + "\n".join(cubes)
    return inner, z_ground, z_min, len(rock_lines), len(cubes)


def build_xml(worldbody_inner):
    """Same scene wrapper as gen_scene.py, without the hardcoded startterrain."""
    return f"""<mujoco model="scene">
    <option timestep='0.0005' iterations="50" tolerance="1e-5" solver="Newton" jacobian="dense" cone="elliptic" noslip_iterations="30" noslip_tolerance="1e-5"/>
    <size njmax="8000" nconmax="4000"/>
    <compiler angle="radian" meshdir="../meshes/" balanceinertia="true"/>
    <default>
        <motor ctrllimited="true"/>
        <default class="viz">
            <geom contype="0" conaffinity="0" group="1" type="mesh" rgba=".6 .6 .7 1" />
        </default>
        <default class="cls">
            <geom group="2" rgba="0.79216 0.81961 0.93333 0.5"/>
        </default>
        <default class="cls_f">
            <geom group="2" rgba="0.79216 0.81961 0.93333 0.1" friction="1 0.005 0.0001"/>
        </default>
        <default class="FTsensor">
            <site type="cylinder" size="0.005 0.005" group="4" rgba=".1 .1 .9 1"/>
        </default>
        <default class="shg20_100_2so">
            <joint damping="0.0248" frictionloss="9.9"/>
        </default>
        <default class="shd20_100_2sh">
            <joint damping="0.0161"  frictionloss="22.0"/>
        </default>
        <default class="shg25_100_2so">
            <joint damping="0.0417"  frictionloss="14"/>
        </default>
        <default class="shg17_100_2so">
            <joint damping="0.0148"  frictionloss="6.5"/>
        </default>
        <default class="shg14_100_2so">
            <joint damping="0.0047"  frictionloss="3.7"/>
        </default>
        <default class="csf_11_100_2xh_f">
            <joint damping="0.0029" frictionloss="1.5"/>
        </default>
    </default>

    <worldbody>
        <!-- mocap body: m->body_pos / m->body_quat 으로 런타임 위치 제어 -->
{worldbody_inner}

    </worldbody>

    <visual>
        <quality shadowsize="2048" offsamples="16"/>
        <map stiffness="10" znear="0.05"/>
    </visual>
</mujoco>
"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate a rocky-mountain footstep terrain in stair_simulation_scene.xml."
    )
    p.add_argument("--csv", default=DEFAULT_CSV, help="global_command.csv path")
    p.add_argument("--xml", default=SCENE_XML, help="MuJoCo scene XML to write")
    p.add_argument("--start", choices=["R", "L"], default="R",
                   help="first swing foot (R=blue, L=red; must match convert_to_global)")

    t = p.add_argument_group("terrain")
    t.add_argument("--rock-shape", choices=["box", "cylinder"], default="box",
                   help="box: yaw-jittered blocks (default); cylinder: round columns")
    t.add_argument("--layout", choices=["grid", "hex"], default="grid",
                   help="grid: square lattice (default); hex: staggered triangular lattice")
    t.add_argument("--spacing", "--cell", type=float, default=0.2, dest="spacing",
                   metavar="M", help="distance between neighboring rock columns [m]")
    t.add_argument("--rock-width", type=float, default=None, metavar="M",
                   help="width of each rock column [m]; omit to use spacing * overlap")
    t.add_argument("--overlap", type=float, default=1.0,
                   help="width = spacing * overlap when --rock-width is omitted")

    t.add_argument("--start-clear", type=float, default=0.3, metavar="M",
                   help="radius around the spawn platform kept free of rock columns [m] (0 disables)")
    t.add_argument("--start-clear-front", type=float, default=0.3, metavar="X",
                   help="forward extent of that clear zone past the platform x [m]")

    t.add_argument("--clearance", type=float, default=0.01,
                   help="how far the filler rock stays below the landing surface [m]")
    t.add_argument("--flush-radius", type=float, default=0.25,
                   help="rock cells within this distance of a stone match its height [m]")
    t.add_argument("--flush-blend", type=float, default=0.25,
                   help="transition band from flush height back to the clearance surface [m]")
    t.add_argument("--flush-drop", type=float, default=0.002,
                   help="tiny offset below the stone top in flush zones [m]")

    t.add_argument("--noise", type=float, default=0.03, help="max downward height jitter [m]")
    t.add_argument("--max-cell-yaw", type=float, default=1.79, help="max random yaw of each rock column [rad]")
    t.add_argument("--corridor", nargs=2, type=float, default=[0.2, 10.5], metavar=("W0", "W1"),
                   help="distance from the footstep path: full height inside W0, tapered to ground at W1 [m]")
    t.add_argument("--terrain-margin", type=float, default=2.0,
                   help="extent of the terrain grid beyond the footstep bounding box [m]")
    t.add_argument("--idw-power", type=float, default=2.0, help="IDW exponent for the surface")
    t.add_argument("--min-height", type=float, default=0.015, help="skip rock columns shorter than this [m]")
    t.add_argument("--seed", type=float, default=7.0, help="seed for deterministic per-cell noise")

    g = p.add_argument_group("platform & ground")
    g.add_argument("--plane-margin", type=float, default=1.0,
                   help="extra half-length around the footstep bounding box for the ground plane [m]")
    g.add_argument("--platform-size", nargs=2, type=float, default=list(DEFAULT_PLATFORM_SIZE),
                   metavar=("HX", "HY"), help="spawn platform horizontal half-extents [m]")
    g.add_argument("--platform-xy", nargs=2, type=float, default=list(DEFAULT_PLATFORM_XY),
                   metavar=("X", "Y"), help="spawn platform center XY [m]")
    g.add_argument("--platform-top", type=float, default=DEFAULT_PLATFORM_TOP,
                   help="spawn platform top face height [m] (default: z=0)")

    o = p.add_argument_group("offsets (cubes/rocks; mocap is still overwritten from CSV at runtime)")
    o.add_argument("--offset-x", type=float, default=0.03, help="stone XY offset in each foot's yaw frame, forward [m]")
    o.add_argument("--offset-y", type=float, default=0.0, help="stone XY offset in each foot's yaw frame, left [m]")
    o.add_argument("--offset-z", type=float, default=0.0, help="stone top height offset in world-up [m]")

    args = p.parse_args()
    if args.corridor[1] <= args.corridor[0]:
        p.error("--corridor W1 must be greater than W0")

    width = column_width(args)
    if width <= 0.0:
        p.error("rock column width must be positive (check --rock-width / --overlap)")
    need = required_width(args.rock_shape, args.layout, args.spacing)
    if width < need - 1e-9:
        hint = ""
        if args.layout == "grid":
            hex_need = required_width(args.rock_shape, "hex", args.spacing)
            hint = f", or --layout hex which only needs {1000 * hex_need:.0f} mm"
        print(
            f"warning: {args.rock_shape} columns {1000 * width:.0f} mm wide on a "
            f"{1000 * args.spacing:.0f} mm {args.layout} lattice leave "
            f"~{1000 * 0.5 * (need - width):.0f} mm gaps; use --rock-width {need:.3f}{hint}"
        )

    targets = read_global_csv(args.csv)
    rows = rows_from_targets(targets, args.start)
    off = (args.offset_x, args.offset_y, args.offset_z)
    inner, z_ground, z_min, n_rocks, n_cubes = build_worldbody(rows, off, args)

    out = os.path.realpath(args.xml)
    with open(out, "w") as f:
        f.write(build_xml(inner))

    ground_note = (
        f"at platform_top={args.platform_top:.4f} (min pos_z={z_min:.4f} >= platform)"
        if z_min >= args.platform_top
        else f"z_ground={z_ground:.4f}"
    )
    print(f"Rock columns: {args.rock_shape}, {1000 * width:.0f} mm wide, {1000 * args.spacing:.0f} mm apart, {args.layout} lattice")
    print(
        f"Wrote startterrain + ground plane + {n_rocks} {args.rock_shape} rock columns "
        f"({args.layout} lattice) + {n_cubes} mocap cubes into {out} ({ground_note})"
    )
