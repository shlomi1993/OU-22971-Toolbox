#!/usr/bin/env bash
# Uninstall wrapper scripts from the active conda environment's bin directory
# Run this script with: bash scripts/bash/uninstall.sh

set -euo pipefail

# Check if conda environment is active
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Error: No conda environment is active."
    echo "Please activate your environment first: conda activate 22971-ray-capstone"
    exit 1
fi

CONDA_BIN="$CONDA_PREFIX/bin"

echo "Uninstalling command wrappers from: $CONDA_BIN"
echo ""

# Remove symlinks from conda environment's bin directory
for cmd in prepare run; do
    DEST="$CONDA_BIN/$cmd"

    if [ -L "$DEST" ]; then
        echo "  Removing: $cmd"
        rm "$DEST"
    elif [ -e "$DEST" ]; then
        echo "  Warning: $DEST exists but is not a symlink. Not removing."
    else
        echo "  Not found: $cmd"
    fi
done

echo ""
echo "Uninstallation complete!"
