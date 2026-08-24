#!/usr/bin/env python3
# Generate local command.csv by sampling step_x / step_y / step_z / step_yaw
# from ranges, then (by default) convert to global_command.csv and rebuild the
# MuJoCo stair scene cubes — same pipeline as g1_controller/cmd/gen_cmd.py.
#
# Usage:
#   python3 generate_command_csv.py 10
#   python3 generate_command_csv.py 20 --start L --seed 0
#   python3 generate_command_csv.py 12 --x -0.1 0.2 --yaw -0.1 0.1 --z 0.0 0.1
#   python3 generate_command_csv.py 8 --no-stop -o /tmp/command.csv
#   python3 generate_command_csv.py 10 --no-convert   # local CSV only
#   python3 generate_command_csv.py 10 --stair        # staircase scene (gen_scene.py --stair)

import argparse
import csv
import os
import random
import subprocess

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(_THIS_DIR, "command.csv")

# Defaults match the usual tocabi_cc command.csv / controller stop step (0.205).
RANGE_X = (0.25, 0.25)       # forward step length [m]
RANGE_Y = (0.205, 0.205)   # lateral step width (positive magnitude) [m]
RANGE_Z = (0.15, 0.15)       # per-step height change [m]
RANGE_YAW = (0.1, 0.1)     # per-step turn [rad]
RANGE_COM_Z = (0.0, 0.0)    # CoM height offset [m] (train com_z_command)
NOMINAL_Y = 0.205          # lateral width used for the final stop step [m]


def sample(lo, hi):
    return random.uniform(lo, hi)


def build_columns(n, rx, ry, rz, ryaw, rcomz, ssp, dsp, height, stop_last):
    """Return column lists for each command row (length n)."""
    posx, posy, posz, roty, comz = [], [], [], [], []
    for i in range(n):
        is_last_stop = stop_last and (i == n - 1)
        if is_last_stop:
            step_x, step_y, step_z, step_yaw = 0.0, NOMINAL_Y, 0.0, 0.0
            step_com_z = 0.0
        elif i == 0:
            step_x, step_y, step_z, step_yaw = 0.3, NOMINAL_Y, 0.0, 0.0
            step_com_z = 0.0
        else:
            step_x = sample(*rx)
            step_y = sample(*ry)
            step_z = sample(*rz)
            step_yaw = sample(*ryaw)
            step_com_z = sample(*rcomz)
        posx.append(f"{step_x:.3f}")
        posy.append(f"{step_y:.3f}")
        posz.append(f"{step_z:.3f}")
        roty.append(f"{step_yaw:.3f}")
        comz.append(f"{step_com_z:.3f}")

    return [
        ["posx", *posx],
        ["posy", *posy],
        ["posz", *posz],
        ["rotr", *[f"{0.0:.2f}" for _ in range(n)]],
        ["rotp", *[f"{0.0:.2f}" for _ in range(n)]],
        ["roty", *roty],
        ["tssp", *[f"{ssp:.2f}" for _ in range(n)]],
        ["tdsp", *[f"{dsp:.2f}" for _ in range(n)]],
        ["foot", *[f"{height:.3f}" for _ in range(n)]],
        ["comz", *comz],
    ]


def main():
    p = argparse.ArgumentParser(
        description="Generate tocabi_cc/cmd/command.csv by sampling step ranges."
    )
    p.add_argument("step", type=int, help="number of footsteps to generate")
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="output CSV path")
    p.add_argument("--start", choices=["R", "L"], default="R",
                   help="first swing foot (passed to convert_to_global)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument("--x", nargs=2, type=float, default=list(RANGE_X),
                   metavar=("MIN", "MAX"), help="posx / step_x range [m]")
    p.add_argument("--y", nargs=2, type=float, default=list(RANGE_Y),
                   metavar=("MIN", "MAX"), help="posy / step_y range [m]")
    p.add_argument("--z", nargs=2, type=float, default=list(RANGE_Z),
                   metavar=("MIN", "MAX"), help="posz / step_z range [m]")
    p.add_argument("--yaw", nargs=2, type=float, default=list(RANGE_YAW),
                   metavar=("MIN", "MAX"), help="roty / step_yaw range [rad]")
    p.add_argument("--comz", nargs=2, type=float, default=list(RANGE_COM_Z),
                   metavar=("MIN", "MAX"), help="comz / CoM height offset range [m]")
    p.add_argument("--ssp", type=float, default=0.7, help="single support time [s]")
    p.add_argument("--dsp", type=float, default=0.15, help="double support time [s]")
    p.add_argument("--height", type=float, default=0.1, help="swing apex height [m]")
    p.add_argument("--no-stop", action="store_true",
                   help="do not force the last step to be a stop step")
    p.add_argument("--no-convert", action="store_true",
                   help="only write local command.csv (skip convert_to_global / gen_scene)")
    p.add_argument("--stair", action="store_true", help="rebuild scene as a staircase via gen_scene.py --stair "
                        "(default: gen_rocky_mountain.py)")
    p.add_argument("--rocky", action="store_true", help="rebuild scene as a rocky mountain via gen_rocky_mountain.py")
    args = p.parse_args()

    if args.step <= 0:
        raise ValueError("step must be a positive integer")
    if args.seed is not None:
        random.seed(args.seed)

    rows = build_columns(
        args.step, args.x, args.y, args.z, args.yaw, args.comz,
        args.ssp, args.dsp, args.height, stop_last=not args.no_stop,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(
        f"Wrote {args.step} footsteps to {args.output} "
        f"(start={args.start}, x={tuple(args.x)}, y={tuple(args.y)}, "
        f"z={tuple(args.z)}, yaw={tuple(args.yaw)})"
    )

    if args.no_convert:
        return

    convert = os.path.join(_THIS_DIR, "convert_to_global.py")
    global_out = os.path.join(_THIS_DIR, "global_command.csv")
    subprocess.run(
        [
            "python3", convert,
            "--input", args.output,
            "--output", global_out,
            "--start", args.start,
            "--no-scene",
        ],
        check=True,
    )

    if args.stair:
        gen_scene = os.path.join(_THIS_DIR, "gen_scene.py")
        subprocess.run(
            ["python3", gen_scene, "--csv", global_out, "--start", args.start, "--stair"],
            check=True,
        )
    else:
        gen_scene = os.path.join(_THIS_DIR, "gen_scene.py")
        subprocess.run(
            ["python3", gen_scene, "--csv", global_out, "--start", args.start],
            check=True,
        )

    if args.rocky:
        gen_scene = os.path.join(_THIS_DIR, "gen_rocky_mountain.py")
        subprocess.run(
            ["python3", gen_scene, "--csv", global_out, "--start", args.start],
            check=True,
        )


if __name__ == "__main__":
    main()
