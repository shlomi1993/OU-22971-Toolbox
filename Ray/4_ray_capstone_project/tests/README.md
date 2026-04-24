# Test Suite Guide

## Quick Start

```bash
# Quick mode (default): reduced params, max_ticks=3, ~30 seconds
pytest

# Full mode: all params, max_ticks=50, ~5 minutes
pytest --full

# Run only tests marked as full (must use with --full)
pytest -m full --full

# Run specific test file
pytest tests/test_data_validation.py

# Run specific test
pytest tests/test_integration.py::test_blocking_workflow
```

## Run Modes

### Quick Mode (Default)
- **Command:** `pytest` (no flags needed)
- **Duration:** ~30 seconds
- **Behavior:**
  - Tests marked `full` are skipped
  - Long tests use `--max-ticks 3` instead of 50
  - Parameterized tests run with reduced parameter sets
- **Use case:** Fast feedback during development

### Full Mode
- **Command:** `pytest --full`
- **Duration:** ~5 minutes
- **Behavior:**
  - All tests run, including those marked `full`
  - Long tests use `--max-ticks 50`
  - All parameterizations execute
- **Use case:** Pre-commit validation, CI/CD

## Markers

Only one marker is used:
- **`full`**: Tests that only run in full mode (skipped by default, run with `--full`)

Tests without the `full` marker run by default in both modes.

## Quick vs Full Comparison

| Aspect | Quick Mode (default) | Full Mode (`--full`) |
|--------|---------------------|-------------------------|
| **Duration** | ~30 seconds | ~5 minutes |
| **max_ticks** | 3 | 50 |
| **test_prepare_script** | 1 param (n_zones=3) | 4 params |
| **test_run_script** | 2 params (1 blocking, 1 async) | 8 params |
| **Workflow tests** | max_ticks=3 | max_ticks=50 |
| **Use case** | Development | Pre-commit, CI/CD |

## Default Configuration

The `pytest.ini` file sets default flags:
- `-v`: Verbose output (show test names)
- `-s`: Show print statements and live subprocess output
- `--strict-markers`: Fail if unknown markers are used

## Common Workflows

```bash
# Development: quick feedback (~30s)
pytest

# Pre-commit: comprehensive validation (~5min)
pytest --full

# Run only validation tests
pytest tests/test_data_validation.py

# Run only actor tests
pytest tests/test_zone_actor.py

# Run only integration tests
pytest tests/test_integration.py

# Debug single test with live output
pytest tests/test_integration.py::test_blocking_workflow

# Run only blocking/async comparison tests
pytest -k workflow --full

# Run only tests marked as full (with --full)
pytest -m full --full
```

## Test Organization

```
tests/
├── test_data_validation.py  # Data validation, scoring logic, artifact tests
├── test_zone_actor.py        # ZoneActor state management and fault tolerance
├── test_integration.py       # Script-level integration and e2e workflows
├── helpers.py                # Synthetic data generation, subprocess runner
├── conftest.py               # Pytest configuration, shared fixtures
└── README.md                 # This file
```

## Tips

- **Quick development loop:** `pytest` (30s)
- **Before commit:** `pytest --full` (5min)
- **Filter by name:** `pytest -k "blocking or async"`
- **See execution time:** `pytest --durations=10`
- **Run in parallel:** `pytest -n auto` (requires `pytest-xdist`)

