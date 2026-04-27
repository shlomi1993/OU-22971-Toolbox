"""
Reset script to stop Ray and clean up generated artifacts.
"""

import argparse
import shutil
import subprocess

from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "output"


def reset(output_dir: Path) -> None:
    """
    Stop Ray and remove output artifacts.

    Args:
        output_dir (Path): Project output directory to remove.
    """
    subprocess.run(["ray", "stop", "--force"])

    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed output directory: {output_dir}")
    else:
        print(f"Output directory {output_dir} does not exist.")

    print("Reset complete.")


def build_reset_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for reset command.

    Returns:
        argparse.ArgumentParser: Parser for reset command arguments.
    """
    parser = argparse.ArgumentParser(
        description="Stop Ray and clean up generated artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False  # Disable help when used as parent parser
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Project output directory to remove")
    return parser


if __name__ == "__main__":
    standalone_parser = argparse.ArgumentParser(parents=[build_reset_parser()])
    args = standalone_parser.parse_args()
    reset(output_dir=args.output_dir)
