# Contributing

Contributions are welcome, especially reproducible graph passes, correctness tests, backend integrations, and controlled benchmark results.

## Development Setup

```bash
git clone https://github.com/Devrajsinh-Jhala/Custom-DL-Optimizer.git
cd Custom-DL-Optimizer
python -m venv .venv
python -m pip install -e ".[dev]"
```

## Before Opening a Pull Request

```bash
python -m ruff check .
python -m pytest
python -m build
python -m twine check dist/*
```

New optimization passes should include:

- A graph-pattern or kernel correctness test.
- A safe fallback for unsupported devices, dtypes, shapes, and layouts.
- Output-parity measurements against eager FP32.
- Benchmark evidence against the strongest relevant native baseline.
- Clear documentation of setup cost and steady-state assumptions.

Do not report speedup from a single unsynchronized GPU timing or present hardware-specific results as universal.
