"""
Reset utilities for the Ray capstone project.

Provides functions to stop Ray and clean up generated artifacts.
"""

import logging
import shutil
import subprocess

from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def reset_ray(project_dir: Path) -> None:
    """
    Stop Ray and remove generated artifacts.

    Args:
        project_dir (Path): Project root directory.
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
    prepared_dir = project_dir / "prepared"
    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)
        logger.info("Prepared assets removed")
    else:
        logger.info("No prepared assets to remove")

    # Remove output artifacts
    output_dir = project_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info("Output artifacts removed")
    else:
        logger.info("No output artifacts to remove")

    logger.info("Reset complete")
