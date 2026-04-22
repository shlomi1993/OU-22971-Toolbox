# Command Wrappers

This directory contains executable wrapper scripts that provide convenient shortcuts for running the main commands.

## Available Commands

- **prepare**: Wrapper for `python main.py prepare <args>`
- **run**: Wrapper for `python main.py run <args>`

## Installation

To install these commands in your conda environment (so you can run `prepare` and `run` directly):

```bash
# Activate your conda environment
conda activate 22971-ray-capstone

# Run the installation script
bash scripts/install.sh
```

This creates symlinks in your conda environment's `bin/` directory, making the commands available whenever the environment is activated.

## Usage Examples

After installation, you can use the shortened commands:

```bash
# Instead of: python main.py prepare --ref-parquet ... --replay-parquet ...
prepare --ref-parquet data/green_tripdata_2023-01.parquet --replay-parquet data/green_tripdata_2023-02.parquet --output-dir prepared/

# Instead of: python main.py run --prepared-dir ... --mode ...
run --prepared-dir prepared/ --output-dir output/ --mode async
```

## Uninstallation

To remove the command wrappers from your conda environment:

```bash
# Activate your conda environment
conda activate 22971-ray-capstone

# Run the uninstallation script
bash scripts/uninstall.sh
```

## How It Works

- The wrapper scripts are stored in `bin/` directory
- Installation creates symlinks in `$CONDA_PREFIX/bin/`
- The scripts resolve their actual location and execute `main.py` with the appropriate subcommand
- Works from any directory as long as the conda environment is activated
