import os
import random

# Number of commands (steps) to generate
NUM_COMMANDS = 50

# Command ranges (left swing foot)
Lx_pos = [-.4, .6]        # Workspace of swing foot is defined around stance foot.
Ly_pos = [.2, .5]         # In stance foot frame, polar coordinates of the next foothold. For the left swing foot, the range will be [-3.14, 0.]
Lyaw_angle = [-0.1, .8]   # Yaw orientation of the target foothold w.r.t. stance foot. For the left swing foot, the range will be [-1.57, 0.]
Lssp_time = [.5, 1.2]
Ldsp_time = [0.0, 0.2]    # nominal dsp 0.15. the dsp time is added to the start and end of each step
Lfoot_height = [0.05, 0.16]


def random_values(value_range, n):
    return [random.uniform(value_range[0], value_range[1]) for _ in range(n)]


def format_line(name, values):
    return name + " " + " ".join("{:.6f}".format(v) for v in values)


def main():
    foothold_x = random_values(Lx_pos, NUM_COMMANDS)
    foothold_y = random_values(Ly_pos, NUM_COMMANDS)
    foothold_yaw = random_values(Lyaw_angle, NUM_COMMANDS)
    t_dsp = random_values(Ldsp_time, NUM_COMMANDS)
    t_ssp = random_values(Lssp_time, NUM_COMMANDS)
    foot_height = random_values(Lfoot_height, NUM_COMMANDS)

    lines = [
        format_line("foothold_x_planned", foothold_x),
        format_line("foothold_y_planned", foothold_y),
        format_line("foothold_yaw_planned", foothold_yaw),
        format_line("t_dsp_planned", t_dsp),
        format_line("t_ssp_planned", t_ssp),
        format_line("foot_height_planned", foot_height),
    ]

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commands_2.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote {} commands to {}".format(NUM_COMMANDS, output_path))


if __name__ == "__main__":
    main()
