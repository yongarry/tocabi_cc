#!/usr/bin/env python3
# Convert per-step LOCAL foot commands (command.csv) into absolute WORLD-frame
# swing-foot targets (global_command.csv), then regenerate stair_simulation_scene
# via gen_rocky_mountain.py (default) or gen_scene.py --stair (--stair).
# Mirrors g1_controller/cmd/convert_footcommand_2_global.py accumulation
# while keeping the tocabi row-oriented CSV layout.
#
# Local CSV (rows):
#   posx,posy,posz,rotr,rotp,roty,tssp,tdsp,foot
#     posy is a positive lateral magnitude; swing-side sign is applied from --start
#
# Global CSV (same rows): pos* / roty become absolute swing-foot landings.
#
# Usage:
#   python3 convert_to_global.py
#   python3 convert_to_global.py --start L
#   python3 convert_to_global.py --input command.csv --output global_command.csv
#   python3 convert_to_global.py --stair   # staircase via gen_scene.py --stair

import argparse
import csv
import math
import os
import subprocess

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(_THIS_DIR, "command.csv")
DEFAULT_OUTPUT = os.path.join(_THIS_DIR, "global_command.csv")

# Approx. spawn foot poses used when building nominal world targets (must line up
# with the controller's measured feet at walk start for cmd_mode=1).
DEFAULT_INIT_LFOOT = (0.1, 0.1025, 0.0, 0.0)
DEFAULT_INIT_RFOOT = (0.1, -0.1025, 0.0, 0.0)


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def read_command_csv(path):
    data = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue
            key = row[0].strip()
            data[key] = [float(v) for v in row[1:] if v.strip() != ""]
    required = ["posx", "posy", "posz", "rotr", "rotp", "roty", "tssp", "tdsp", "foot"]
    for c in required:
        if c not in data:
            raise ValueError(f"missing required row '{c}' in {path}")
    n = len(data["posx"])
    for c in required:
        if len(data[c]) != n:
            raise ValueError(f"row '{c}' length {len(data[c])} != posx length {n}")
    # Optional CoM height offset (pass-through; defaults to zeros)
    if "comz" in data:
        if len(data["comz"]) != n:
            raise ValueError(f"row 'comz' length {len(data['comz'])} != posx length {n}")
    elif "com_z" in data:
        data["comz"] = data["com_z"]
        if len(data["comz"]) != n:
            raise ValueError(f"row 'com_z' length {len(data['comz'])} != posx length {n}")
    else:
        data["comz"] = [0.0] * n
    return data


def to_global(data, init_stance, start_swing):
    """Accumulate local steps into absolute world swing-foot targets.

    stance[0] is the INITIAL STANCE FOOT world pose (the foot that is NOT swinging
    first). For each step k:
        step_y_signed = +|posy| for left swing, -|posy| for right swing
        swing = stance + R(stance.yaw) * (posx, step_y_signed, posz)
        swing.yaw = wrap(stance.yaw + roty)
        next stance = swing
    """
    n = len(data["posx"])
    swing_is_right = (start_swing == "R")
    stance = {
        "x": init_stance[0],
        "y": init_stance[1],
        "z": init_stance[2],
        "yaw": init_stance[3],
    }

    global_x, global_y, global_z, global_roty = [], [], [], []
    for k in range(n):
        # Alternate swing foot: start_swing on even steps.
        right = swing_is_right if (k % 2 == 0) else (not swing_is_right)
        sign = -1.0 if right else 1.0
        sy = sign * abs(data["posy"][k])
        c, s = math.cos(stance["yaw"]), math.sin(stance["yaw"])
        swing = {
            "x": stance["x"] + c * data["posx"][k] - s * sy,
            "y": stance["y"] + s * data["posx"][k] + c * sy,
            "z": stance["z"] + data["posz"][k],
            "yaw": wrap_to_pi(stance["yaw"] + data["roty"][k]),
        }
        global_x.append(swing["x"])
        global_y.append(swing["y"])
        global_z.append(swing["z"])
        global_roty.append(swing["yaw"])
        stance = swing

    return global_x, global_y, global_z, global_roty


def write_global_csv(path, global_x, global_y, global_z, global_roty, data):
    rows = [
        ["posx"] + [f"{v:.4f}" for v in global_x],
        ["posy"] + [f"{v:.4f}" for v in global_y],
        ["posz"] + [f"{v:.4f}" for v in global_z],
        ["rotr"] + [f"{v:.2f}" for v in data["rotr"]],
        ["rotp"] + [f"{v:.2f}" for v in data["rotp"]],
        ["roty"] + [f"{v:.4f}" for v in global_roty],
        ["tssp"] + [f"{v:.2f}" for v in data["tssp"]],
        ["tdsp"] + [f"{v:.2f}" for v in data["tdsp"]],
        ["foot"] + [f"{v:.3f}" for v in data["foot"]],
        ["comz"] + [f"{v:.3f}" for v in data["comz"]],
    ]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert local foot commands to absolute world-frame targets."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="local command.csv path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="global_command.csv path")
    parser.add_argument("--start", choices=["R", "L"], default="R",
                        help="first swing foot (default: R, matching is_right_stance_first=false)")
    parser.add_argument("--left", action="store_true",
                        help="alias for --start L (kept for backward compatibility)")
    parser.add_argument("--init-lfoot", nargs=4, type=float, default=list(DEFAULT_INIT_LFOOT),
                        metavar=("X", "Y", "Z", "YAW"), help="left foot world pose at spawn")
    parser.add_argument("--init-rfoot", nargs=4, type=float, default=list(DEFAULT_INIT_RFOOT),
                        metavar=("X", "Y", "Z", "YAW"), help="right foot world pose at spawn")
    parser.add_argument("--no-scene", action="store_true",
                        help="skip regenerating stair_simulation_scene.xml")
    parser.add_argument("--stair", action="store_true",
                        help="rebuild scene as a staircase via gen_scene.py --stair "
                             "(default: gen_rocky_mountain.py)")
    args = parser.parse_args()

    start = "L" if args.left else args.start
    # Initial stance foot = the foot that is NOT swinging first.
    init_foot = args.init_lfoot if start == "R" else args.init_rfoot

    data = read_command_csv(args.input)
    global_x, global_y, global_z, global_roty = to_global(data, init_foot, start)
    write_global_csv(args.output, global_x, global_y, global_z, global_roty, data)

    print(f"World-frame swing-foot targets (start swing={start}, stance[0]={init_foot}):")
    for i, (x, y, z, ry) in enumerate(zip(global_x, global_y, global_z, global_roty)):
        print(f"  step {i:2d}: x={x:.4f}  y={y:.4f}  z={z:.4f}  roty={math.degrees(ry):.2f}deg")

    if not args.no_scene:
        if args.stair:
            gen_scene = os.path.join(_THIS_DIR, "gen_scene.py")
            subprocess.run(
                ["python3", gen_scene, "--csv", args.output, "--start", start, "--stair"],
                check=True,
            )
        else:
            gen_scene = os.path.join(_THIS_DIR, "gen_rocky_mountain.py")
            subprocess.run(
                ["python3", gen_scene, "--csv", args.output, "--start", start],
                check=True,
            )
