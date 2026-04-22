#!/usr/bin/env bash
# Install wrapper scripts into the active conda environment's bin directory
# Run this script with: bash scripts/install.sh

set -euo pipefail

# Check if conda environment is active
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Error: No conda environment is active."
    echo "Please activate your environment first: conda activate 22971-ray-capstone"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$PROJECT_DIR/bin"
CONDA_BIN="$CONDA_PREFIX/bin"

echo "Installing command wrappers to: $CONDA_BIN"
echo ""

# Create symlinks in conda environment's bin directory
for cmd in prepare run; do
    SRC="$BIN_DIR/$cmd"
    DEST="$CONDA_BIN/$cmd"

    if [ -L "$DEST" ]; then
        echo "  Removing existing symlink: $cmd"
        rm "$DEST"
    elif [ -e "$DEST" ]; then
        echo "  Warning: $DEST exists and is not a symlink. Skipping."
        continue
    fi
    
    echo "  Creating symlink: $cmd -> $SRC"
    ln -s "$SRC" "$DEST"
done

echo ""
echo "Installation complete! You can now run:"
echo "  prepare --ref-parquet <file> --replay-parquet <file> --output-dir <dir>"
echo "  run --prepared-dir <dir> --output-dir <dir> --mode <blocking|async|stress>"
echo ""
echo "To uninstall, run: bash scripts/uninstall.sh"
