import argparse
import csv
import math
import os

INPUT_CSV = os.path.join(os.path.dirname(__file__), "command.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "global_command.csv")


def read_command_csv(path):
    data = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                key = row[0].strip()
                values = [float(v) for v in row[1:]]
                data[key] = values
    return data


def to_global(data):
    """Convert per-step local foot commands to global stance/swing foot positions.

    Mirrors the PyTorch footstep planning logic:
      - step 0 : swing = stance_0 + (step_x, step_y)  (no yaw rotation yet)
      - step k : stance_k = swing_{k-1}
                 swing_k  = stance_k + R(stance_k.yaw) * (step_x_k, step_y_k)
    where R(yaw) is the 2-D rotation matrix for the current stance yaw.
    """
    step_x   = data["posx"]
    step_y   = data["posy"]
    step_z   = data["posz"]
    step_yaw = data["roty"]

    n = len(step_x)

    # stance / swing arrays: [x, y, z, yaw]
    stance = [[0.0, 0.0, 0.0, 0.0]] * n
    swing  = [[0.0, 0.0, 0.0, 0.0]] * n

    # ── step 0 ──────────────────────────────────────────────────────────────
    # stance_0 is the origin (first stance foot = world origin)
    stance[0] = [0.0, 0.1025, 0.0, 0.0]

    swing[0] = [
        stance[0][0] + step_x[0],
        stance[0][1] + step_y[0],
        stance[0][2] + step_z[0],
        stance[0][3] + step_yaw[0],
    ]

    # ── step k (k >= 1) ─────────────────────────────────────────────────────
    for k in range(1, n):
        # new stance = previous swing
        stance[k] = swing[k - 1][:]

        yaw = stance[k][3]
        dx  =  math.cos(yaw) * step_x[k] - math.sin(yaw) * step_y[k]
        dy  =  math.sin(yaw) * step_x[k] + math.cos(yaw) * step_y[k]

        swing[k] = [
            stance[k][0] + dx,
            stance[k][1] + dy,
            stance[k][2] + step_z[k],
            stance[k][3] + step_yaw[k],
        ]

    global_x    = [sw[0] for sw in swing]
    global_y    = [sw[1] for sw in swing]
    global_z    = [sw[2] for sw in swing]
    global_roty = [sw[3] for sw in swing]

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
        ["foot"] + [f"{v:.2f}" for v in data["foot"]],
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert local foot commands to global frame.")
    parser.add_argument("--left", action="store_true", help="First swing foot is the left foot")
    args = parser.parse_args()

    is_left_swing_first: bool = args.left

    data = read_command_csv(INPUT_CSV)
    if not is_left_swing_first: # right foot swing first -> right foot swing command y, yaw should be negative (only for odd number of steps)
        for i in range(0, len(data["posx"])-1, 2):
            data["posy"][i] *= -1
            data["roty"][i] *= -1
    global_x, global_y, global_z, global_roty = to_global(data)
    write_global_csv(OUTPUT_CSV, global_x, global_y, global_z, global_roty, data)

    print("Global positions:")
    for i, (x, y, z, ry) in enumerate(zip(global_x, global_y, global_z, global_roty)):
        print(f"  step {i:2d}: x={x:.4f}  y={y:.4f}  z={z:.4f}  roty={math.degrees(ry):.2f}deg")
