#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path


def repeated(value: str, count: int) -> list[str]:
    return [value] * count


def build_rows(step_count: int) -> list[list[str]]:
    posz_values = [f"{random.uniform(-0.1, 0.1):.4f}" for _ in range(step_count)]
    if step_count >= 1:
        posz_values[0] = "0.0"
    if step_count >= 2:
        posz_values[1] = "0.0"

    return [
        ["posx", *repeated("0.2", step_count)],
        ["posy", *repeated("0.205", step_count)],
        ["posz", *posz_values],
        ["rotr", *repeated("0", step_count)],
        ["rotp", *repeated("0", step_count)],
        ["roty", *repeated("0", step_count)],
        ["tssp", *repeated("0.9", step_count)],
        ["tdsp", *repeated("0.1", step_count)],
        ["foot", *repeated("0.08", step_count)],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tocabi_cc/cmd/command.csv from step count."
    )
    parser.add_argument("step", type=int, help="Number of steps(columns) to generate")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).with_name("command.csv"),
        help="Output CSV path (default: command.csv in this script directory)",
    )
    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("step must be a positive integer")

    rows = build_rows(args.step)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Generated {args.output} with {args.step} steps.")


if __name__ == "__main__":
    main()
