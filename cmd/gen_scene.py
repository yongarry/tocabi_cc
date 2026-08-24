#!/usr/bin/env python3
"""
Rebuild stair_simulation_scene.xml cube mocap bodies from global_command.csv
(or a step count). Runtime still overwrites mocap_pos from the CSV; baking
x/z/yaw into the XML keeps the scene consistent when opened alone.

Offsets (--offset-x/y/z) apply only to stepping stones / stair treads:
  stone XY = CSV target + R(yaw)*(ox, oy)   # foot yaw frame: +x fwd, +y left
  stone Z  = CSV pos_z + oz                 # world-up

When min(pos_z) is below the spawn platform top, startterrain spans down to
that level and footstep_ground sits there. If min(pos_z) is at/above platform
top, footstep_ground moves to the platform top and startterrain uses a fixed
depth.

--stair places static tread geoms along the L/R midline (not on each foot),
so a turning/spiral command looks like one staircase instead of a zigzag of
foot-sized blocks. Each tread is a box from the ground up to pos_z. The last
CSV step is a zero-length stop and is omitted (it would sit on the previous
tread). Foot-marker cubes are still baked so landings can be checked.

Usage:
  python3 gen_scene.py                  # use cmd/global_command.csv
  python3 gen_scene.py --csv path.csv
  python3 gen_scene.py --stair          # staircase treads at y=0
  python3 gen_scene.py --stair --size 0.125 0.5
  python3 gen_scene.py --offset-x 0.03 --offset-y 0.0
  python3 gen_scene.py 20               # N cubes at origin (legacy)
"""
import argparse
import csv
import math
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(_THIS_DIR, "global_command.csv")
SCENE_XML = os.path.join(
    _THIS_DIR,
    "../../dyros_tocabi_v2/tocabi_description/mujoco_model/stair_simulation_scene.xml",
)

# Foot-sized stepping stones placed at each global swing-foot target.
CUBE_HX = 0.125
CUBE_HY = 0.1
CUBE_HZ = 0.1
# --stair default half-extents. Z is from the ground to pos_z unless HZ is given.
# STAIR_SIZE = (0.125, 0.3, 3.7)
STAIR_SIZE = (0.125, 0.5)
# Spawn stance used when the first tread has no previous swing (match convert_to_global).
DEFAULT_INIT_LFOOT = (0.1, 0.1025, 0.0, 0.0)
DEFAULT_INIT_RFOOT = (0.1, -0.1025, 0.0, 0.0)
# Right = blue-ish, Left = red-ish (same convention as g1_controller markers)
# COLOR_R = "0.15 0.35 0.95 1"
# COLOR_L = "0.90 0.15 0.15 1"
COLOR_L = "0.2 0.2 0.2 1"
COLOR_R = "0.2 0.2 0.2 1"
COLOR_GROUND = "0.7 0.6 0.5 1"
COLOR_PLATFORM = "0.2 0.2 0.2 1"
# startterrain: top at z=0, centered at x=0.1 (spawn feet).
DEFAULT_PLATFORM_SIZE = (0.16, 0.30)
DEFAULT_PLATFORM_XY = (0.1, 0.0)
DEFAULT_PLATFORM_TOP = 0.0
DEFAULT_PLATFORM_HALF_HEIGHT = 0.5  # when min(pos_z) >= platform_top
DEFAULT_PLANE_MARGIN = 3.0


def read_global_csv(path):
    data = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or all(c.strip() == "" for c in row):
                continue
            key = row[0].strip()
            data[key] = [float(v) for v in row[1:] if v.strip() != ""]
    if "posx" not in data:
        raise ValueError(f"missing posx row in {path}")
    n = len(data["posx"])
    posy = data.get("posy", [0.0] * n)
    posz = data.get("posz", [0.0] * n)
    roty = data.get("roty", [0.0] * n)
    if len(posy) != n or len(posz) != n or len(roty) != n:
        raise ValueError(f"posy/posz/roty length mismatch in {path}")
    return [
        {"x": data["posx"][i], "y": posy[i], "z": posz[i], "yaw": roty[i]}
        for i in range(n)
    ]


def foot_for_step(i, start):
    """Even steps use `start` swing foot; odd steps use the other."""
    if i % 2 == 0:
        return start
    return "L" if start == "R" else "R"


def yaw_to_quat(yaw):
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def yaw_frame_offset(x, y, yaw, ox, oy):
    """Translate (x, y) by (ox, oy) in the foot yaw frame (+x fwd, +y left)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * ox - s * oy, y + s * ox + c * oy


def landing_poses(targets, off):
    """Stepping-stone landings: XY in each foot's yaw frame, Z world-up."""
    ox, oy, oz = off
    out = []
    for t in targets:
        sx, sy = yaw_frame_offset(t["x"], t["y"], t["yaw"], ox, oy)
        out.append({"x": sx, "y": sy, "z": t["z"] + oz, "yaw": t["yaw"]})
    return out


def cube_body(i, x, y, z, yaw, foot):
    # Center the box so its top face sits at landing height z (same as mujoco_ros:
    # mocap_pos.z = posz - CUBE_HZ).
    qw, qx, qy, qz = yaw_to_quat(yaw)
    rgba = COLOR_R if foot == "R" else COLOR_L
    return (
        # f'        <body name="cube_{i:02d}" mocap="true" '
        f'        <body name="cube_{i:02d}" '
        f'pos="{x:.4f} {y:.4f} {z - CUBE_HZ:.4f}" '
        f'quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}">'
        f'<geom type="box" size="{CUBE_HX:.4f} {CUBE_HY:.4f} {CUBE_HZ:.4f}" '
        f'rgba="{rgba}"/></body>'
    )


def prev_stance(targets, i, start):
    """World pose of the stance foot for swing target i (previous swing, or spawn)."""
    if i > 0:
        return targets[i - 1]
    init = DEFAULT_INIT_LFOOT if start == "R" else DEFAULT_INIT_RFOOT
    return {"x": init[0], "y": init[1], "z": init[2], "yaw": init[3]}


def _local_xy(px, py, ox, oy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = px - ox, py - oy
    return c * dx + s * dy, -s * dx + c * dy


def tread_pose(swing, stance, hx, hy_min):
    """Center the tread on the L/R midline in the swing yaw frame.

    A box is symmetric, so keeping the center on the swing foot zigzags with
    left/right. Shift to the stance–swing midpoint along local y; hx stays
    the requested half-depth. The last (stop) step is omitted by the caller
    because it lands beside the previous foot at the same height.
    """
    yaw = swing["yaw"]
    c, s = math.cos(yaw), math.sin(yaw)
    _, ly_st = _local_xy(stance["x"], stance["y"], swing["x"], swing["y"], yaw)
    hy = max(hy_min, 0.5 * abs(ly_st))
    yc = 0.5 * ly_st
    px = swing["x"] - s * yc
    py = swing["y"] + c * yc
    return px, py, yaw, hx, hy


def platform_ground_geoms(xs, ys, z_min, platform_size, platform_xy,
                          platform_top, plane_margin):
    """Spawn platform + ground plane at the lowest footstep (or platform top).

    If min(pos_z) >= platform_top, the plane sits at the platform top and the
    box uses a fixed half-height. Otherwise the plane is at z_min and the
    platform spans from that level up to platform_top.
    """
    if z_min >= platform_top:
        z_ground = platform_top
        plat_hz = DEFAULT_PLATFORM_HALF_HEIGHT
        plat_zc = platform_top - plat_hz
    else:
        z_ground = z_min
        plat_hz = (platform_top - z_min) * 0.5
        plat_zc = z_min + plat_hz

    plat_hx, plat_hy = platform_size
    plat_x, plat_y = platform_xy
    if xs:
        plane_cx = 0.5 * (min(xs) + max(xs))
        plane_cy = 0.5 * (min(ys) + max(ys))
        plane_hx = 0.5 * (max(xs) - min(xs)) + plane_margin
        plane_hy = 0.5 * (max(ys) - min(ys)) + plane_margin
    else:
        plane_cx, plane_cy = plat_x, plat_y
        plane_hx = plane_hy = plane_margin

    lines = [
        f'        <geom name="startterrain" type="box" group="3" '
        f'size="{plat_hx:.4f} {plat_hy:.4f} {plat_hz:.4f}" '
        f'pos="{plat_x:.4f} {plat_y:.4f} {plat_zc:.4f}" '
        f'rgba="{COLOR_PLATFORM}"/>',
        f'        <geom name="footstep_ground" type="plane" '
        f'pos="{plane_cx:.4f} {plane_cy:.4f} {z_ground:.4f}" '
        f'size="{plane_hx:.4f} {plane_hy:.4f} 0.1" '
        f'rgba="{COLOR_GROUND}" group="3"/>',
    ]
    return "\n".join(lines), z_ground


def stair_geom(i, x, y, z_top, yaw, foot, hx, hy, z_ground, fixed_hz=None):
    """Static tread. Top face at z_top; height from z_ground unless HZ is fixed."""
    if fixed_hz is not None:
        hz = fixed_hz
        zc = z_top - hz
    else:
        height = z_top - z_ground
        if height < 1e-4:
            return None
        hz = height * 0.5
        zc = z_ground + hz
    qw, qx, qy, qz = yaw_to_quat(yaw)
    rgba = COLOR_R if foot == "R" else COLOR_L
    return (
        f'        <geom name="stair_{i:02d}" type="box" group="3" '
        f'size="{hx:.4f} {hy:.4f} {hz:.4f}" '
        # f'pos="{x:.4f} 0.0000 {zc:.4f}" '
        f'pos="{x:.4f} {y:.4f} {zc:.4f}" '
        f'quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}" '
        f'rgba="0.8 0.4 0.1 1"/>'
    )


def build_xml(cubes_xml):
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
{cubes_xml}

    </worldbody>

    <visual>
        <quality shadowsize="2048" offsamples="16"/>
        <map stiffness="10" znear="0.05"/>
    </visual>
</mujoco>
"""


def main():
    p = argparse.ArgumentParser(description="Regenerate stair_simulation_scene.xml cubes.")
    p.add_argument("n", nargs="?", type=int, default=None,
                   help="cube count (legacy; preferred: --csv)")
    p.add_argument("--csv", default=None, help="global_command.csv path (default if n omitted)")
    p.add_argument("--xml", default=SCENE_XML, help="output scene XML path")
    p.add_argument("--start", choices=["R", "L"], default="R",
                   help="first swing foot (R=blue, L=red; must match convert_to_global)")
    p.add_argument("--stair", action="store_true",
                   help="place static treads on the L/R midline (covers both "
                        "feet; no mocap cubes). hy grows with step width")
    p.add_argument("--size", nargs="+", type=float, default=None, metavar="H",
                   help="with --stair: HX HY [HZ] box half-extents [m] "
                        f"(default {STAIR_SIZE[0]} {STAIR_SIZE[1]}, Z = ground→pos_z). "
                        "If HZ is given, vertical half-size is fixed (top at pos_z)")
    p.add_argument("--plane-margin", type=float, default=DEFAULT_PLANE_MARGIN,
                   help="extra half-length around the footstep bounding box for the ground plane [m]")
    p.add_argument("--platform-size", nargs=2, type=float,
                   default=list(DEFAULT_PLATFORM_SIZE), metavar=("HX", "HY"),
                   help="spawn platform horizontal half-extents [m]")
    p.add_argument("--platform-xy", nargs=2, type=float,
                   default=list(DEFAULT_PLATFORM_XY), metavar=("X", "Y"),
                   help="spawn platform center XY [m]")
    p.add_argument("--platform-top", type=float, default=DEFAULT_PLATFORM_TOP,
                   help="spawn platform top face height [m] (default: z=0)")
    p.add_argument("--offset-x", type=float, default=-0.0,
                   help="stone XY offset in each foot's yaw frame, forward [m]")
    p.add_argument("--offset-y", type=float, default=0.0,
                   help="stone XY offset in each foot's yaw frame, left [m]")
    p.add_argument("--offset-z", type=float, default=0.0,
                   help="stone top height offset in world-up [m]")
    args = p.parse_args()
    if args.size is not None and not args.stair:
        p.error("--size is only used with --stair")
    if args.stair:
        size = args.size if args.size is not None else list(STAIR_SIZE)
        if len(size) not in (2, 3):
            p.error("--size expects HX HY [HZ] (2 or 3 values)")
        args.size = size

    if args.n is not None:
        targets = [{"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0} for _ in range(args.n)]
        src = f"n={args.n}"
    else:
        csv_path = args.csv or DEFAULT_CSV
        targets = read_global_csv(csv_path)
        src = csv_path

    off = (args.offset_x, args.offset_y, args.offset_z)
    stones = landing_poses(targets, off)
    zs = [s["z"] for s in stones]
    z_min = min(zs) if zs else args.platform_top
    xs = [s["x"] for s in stones]
    ys = [s["y"] for s in stones]
    base, z_ground = platform_ground_geoms(
        xs, ys, z_min,
        tuple(args.platform_size), tuple(args.platform_xy),
        args.platform_top, args.plane_margin,
    )
    ox, oy, oz = off
    extra_off = f", offset=({ox}, {oy}, {oz})" if off != (0.0, 0.0, 0.0) else ""

    if args.stair:
        hx, hy = args.size[0], args.size[1]
        fixed_hz = args.size[2] if len(args.size) == 3 else None
        stair_lines = []
        for i, t in enumerate(targets):
            # last CSV row is a stop (step_x/z/yaw = 0); same XY as the previous
            # landing, so a full-height pillar here would bury that tread.
            if i == len(targets) - 1:
                continue
            # Midline from raw CSV feet so offset does not change tread width;
            # then shift the tread in the swing yaw frame.
            px, py, yaw, hx_i, hy_i = tread_pose(
                t, prev_stance(targets, i, args.start), hx, hy,
            )
            px, py = yaw_frame_offset(px, py, yaw, ox, oy)
            g = stair_geom(
                i + 1, px, py, t["z"] + oz, yaw, foot_for_step(i, args.start),
                hx_i, hy_i, z_ground, fixed_hz=fixed_hz,
            )
            if g is not None:
                stair_lines.append(g)
        cubes = base + "\n\n" + "\n".join(stair_lines)
        n_geom = len(stair_lines)
        kind = "stair"
        extra = (
            f" midline size=({hx}, {hy}, {fixed_hz if fixed_hz is not None else 'ground'})"
            + extra_off
        )

    else:
        cube_lines = "\n".join(
            cube_body(
                i + 1, s["x"], s["y"], s["z"], s["yaw"], foot_for_step(i, args.start)
            )
            for i, s in enumerate(stones)
        )
        cubes = base + "\n\n" + cube_lines
        n_geom = len(targets)
        kind = "cube"
        extra = extra_off

    out = os.path.realpath(args.xml)
    with open(out, "w") as f:
        f.write(build_xml(cubes))
    n_r = sum(1 for i in range(len(targets)) if foot_for_step(i, args.start) == "R")
    n_l = len(targets) - n_r
    ground_note = (
        f"at platform_top={args.platform_top:.4f} (min pos_z={z_min:.4f} >= platform)"
        if z_min >= args.platform_top
        else f"z_ground={z_ground:.4f}"
    )
    print(
        f"생성 완료: {out}  ({kind} 개수: {n_geom}, R={n_r}/blue L={n_l}/red, "
        f"start={args.start}{extra}, from {src}, {ground_note})"
    )


if __name__ == "__main__":
    main()
