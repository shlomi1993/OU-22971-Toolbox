"""
Reset script to stop Ray and clean up generated artifacts.
"""

import shutil
import subprocess

from pathlib import Path


def reset_ray() -> None:
    """
    Stop Ray and remove output artifacts.
    """
    project_dir = Path(__file__).parent.parent
    output_dir = project_dir / "output"
    subprocess.run(["ray", "stop", "--force"])
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed output directory: {output_dir}")


if __name__ == "__main__":
    reset_ray()
