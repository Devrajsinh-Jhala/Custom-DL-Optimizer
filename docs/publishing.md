# Publishing to PyPI

PyPI versions are immutable. Every correction after publication requires a new version.

## Trusted Publishing

Production releases use GitHub Actions OpenID Connect rather than a long-lived API token. Configure the existing PyPI project with:

```text
Owner: Devrajsinh-Jhala
Repository: Custom-DL-Optimizer
Workflow: publish.yml
Environment: pypi
```

After configuration, publish a GitHub release. The workflow builds, validates, and uploads the distributions with a short-lived credential.

## Local Release Checks

```bash
python -m pip install -e ".[dev]"
python -m ruff check custom_dl_optimizer tests examples
python -m pytest
python -m build
python -m twine check dist/*
```

Install the wheel in a clean environment before tagging:

```bash
python -m venv .release-venv
.release-venv/bin/python -m pip install dist/*.whl
.release-venv/bin/python -c "import custom_dl_optimizer; print(custom_dl_optimizer.__version__)"
```

On Windows, use `.release-venv\Scripts\python.exe`.

## TestPyPI

Manual TestPyPI uploads are optional:

```bash
python -m twine upload --repository testpypi dist/*
python -m pip install --extra-index-url https://test.pypi.org/simple/ custom-dl-optimizer
```

Never commit a PyPI token. If a token was ever exposed outside the ignored local file, revoke it before publishing.
