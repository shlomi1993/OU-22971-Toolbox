"""
A reset script to stop Ray and clean up generated artifacts.
"""

import argparse
import shutil
import subprocess

from pathlib import Path

from src.common import DEFAULT_OUTPUT_DIR, DEFAULT_PREPARED_DIR
from src.logger import logger


def reset_ray(prepared_dir: Path, output_dir: Path) -> None:
    """
    Stop Ray and remove generated artifacts.

    Args:
        prepared_dir (Path): Directory containing prepared assets.
        output_dir (Path): Directory containing output artifacts.
    """
    # Stop Ray
    logger.info("Stopping Ray...")
    try:
        subprocess.run(["ray", "stop", "--force"], capture_output=True, timeout=30)
        logger.info("Ray stopped successfully")
    except subprocess.TimeoutExpired:
        logger.warning("Ray stop command timed out, continuing with cleanup...")
    except FileNotFoundError:
        logger.warning("Ray command not found, skipping Ray shutdown")
    except Exception as e:
        logger.warning(f"Failed to stop Ray: {e}, continuing with cleanup...")

    # Remove prepared assets
    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)
        logger.info(f"Prepared assets removed: {prepared_dir}")
    else:
        logger.info("No prepared assets to remove")

    # Remove output artifacts
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info(f"Output artifacts removed: {output_dir}")
    else:
        logger.info("No output artifacts to remove")

    logger.info("Reset complete")


def build_reset_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for reset command.

    Returns:
        argparse.ArgumentParser: Parser for reset command arguments.
    """
    parser = argparse.ArgumentParser(
        description="Stop Ray and remove generated artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False  # Disable help when used as parent parser
    )
    parser.add_argument("--prepared-dir", type=Path, default=Path(DEFAULT_PREPARED_DIR), help="Directory with prepared assets")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR), help="Directory with output artifacts")
    return parser


def main():
    standalone_parser = argparse.ArgumentParser(parents=[build_reset_parser()])
    args = standalone_parser.parse_args()
    reset_ray(args.prepared_dir, args.output_dir)


if __name__ == "__main__":
    main()
