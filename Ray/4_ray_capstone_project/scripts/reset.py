"""
Reset script to stop Ray and clean up generated artifacts.
"""

import argparse
import shutil
import subprocess

from pathlib import Path


def reset(prepared_dir: Path | None = None, output_dir: Path | None = None) -> None:
    """
    Stop Ray and remove output artifacts.

    Args:
        prepared_dir (Path | None): Optional prepared assets directory to remove.
        output_dir (Path | None): Optional output directory to remove. Defaults to project output/.
    """
    subprocess.run(["ray", "stop", "--force"])

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"

    dirs_to_remove = [output_dir]
    if prepared_dir is not None:
        dirs_to_remove.append(prepared_dir)

    for d in dirs_to_remove:
        if d.exists():
            shutil.rmtree(d)
            print(f"Removed directory: {d}")

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
    parser.add_argument("--prepared-dir", type=Path, default=None, help="Prepared assets directory to remove")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory to remove (defaults to project output/)")
    return parser


if __name__ == "__main__":
    standalone_parser = argparse.ArgumentParser(parents=[build_reset_parser()])
    args = standalone_parser.parse_args()
    reset(prepared_dir=args.prepared_dir, output_dir=args.output_dir)
