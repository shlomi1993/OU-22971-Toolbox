#!/usr/bin/env bash
# reset_ray.sh - Stop Ray and clean up generated artifacts.

set -euo pipefail

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${GREEN}Stopping Ray...${NC}"
if command -v ray &> /dev/null; then
    if timeout 30s ray stop --force &> /dev/null; then
        echo -e "${GREEN}Ray stopped successfully${NC}"
    else
        echo -e "${YELLOW}Warning: Ray stop command timed out or failed, continuing with cleanup...${NC}"
    fi
else
    echo -e "${YELLOW}Warning: Ray command not found, skipping Ray shutdown${NC}"
fi

echo -e "${GREEN}Removing generated artifacts...${NC}"
if [ -d "$PROJECT_DIR/output" ]; then
    rm -rf "$PROJECT_DIR/output"
    echo -e "${GREEN}Output artifacts removed: $PROJECT_DIR/output${NC}"
else
    echo "No output artifacts to remove"
fi

echo -e "${GREEN}Reset complete${NC}"
