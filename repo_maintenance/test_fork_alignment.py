import subprocess
import unittest

from pathlib import Path


EXPECTED_NON_CAPSTONE_DIFF_PATHS = {
    "repo_maintenance/test_fork_alignment.py",
    ".gitignore",
}


class ForkAlignmentTest(unittest.TestCase):
    def test_fork_and_original_diff_alignment(self):
        repo_root = Path(__file__).resolve().parents[1]
        cmd = "git diff --name-status upstream/main -- . ':(exclude)**/*capstone_project/**'"
        result = subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True, text=True, shell=True)
        changed_paths = {line.split("\t")[-1] for line in result.stdout.splitlines() if line.strip()}
        self.assertEqual(changed_paths, EXPECTED_NON_CAPSTONE_DIFF_PATHS)


if __name__ == "__main__":
    unittest.main()
