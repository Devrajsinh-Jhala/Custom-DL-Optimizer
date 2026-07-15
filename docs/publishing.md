# Publishing to PyPI

PyPI versions are immutable. Every correction after publication requires a new version.

## One-Time Trusted Publishing Setup

Production releases use GitHub Actions OpenID Connect rather than a long-lived API token. Do not add `token.txt`, a `.pypirc`, or a PyPI token to GitHub Secrets.

1. Open the repository's **Settings > Environments** page on GitHub.
2. Create an environment named `pypi`.
3. Add a required reviewer to that environment if your GitHub plan supports deployment protection rules.
4. Open the `custom-dl-optimizer` project on PyPI and select **Manage > Publishing**.
5. Add a GitHub Actions Trusted Publisher with these exact values:

```text
Owner: Devrajsinh-Jhala
Repository: Custom-DL-Optimizer
Workflow: publish.yml
Environment: pypi
```

The workflow filename and environment are case-sensitive identity claims. After this setup, PyPI issues a short-lived credential only to the matching publish job.

## Prepare a Release

Start from a clean, current `master` branch. Update the version in `pyproject.toml`, `custom_dl_optimizer/__init__.py`'s source-checkout fallback, `CHANGELOG.md`, and `CITATION.cff`. The version must not already exist on PyPI.

```bash
git switch master
git pull --ff-only
python -m pip install -e ".[dev]"
python -m ruff check custom_dl_optimizer tests examples tools
python -m pytest
python tools/validate_research_notebook.py
python -c "import shutil; [shutil.rmtree(path, ignore_errors=True) for path in ('build', 'dist', 'custom_dl_optimizer.egg-info')]"
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

## Publish Version 2.2.0

Once the Trusted Publisher exists and `master` is green, create a published GitHub Release. This command creates the `v2.2.0` tag from `master` and publishes the release:

```bash
gh release create v2.2.0 \
  --repo Devrajsinh-Jhala/Custom-DL-Optimizer \
  --target master \
  --title "v2.2.0" \
  --generate-notes
```

On PowerShell, place the command on one line or replace each `\` with a backtick. Publishing the GitHub Release triggers `.github/workflows/publish.yml`. The workflow rejects a release whose tag does not exactly match `v` plus the package version, builds fresh distributions, checks them with Twine, and uploads through Trusted Publishing.

Verify the release after the workflow succeeds:

```bash
python -m pip index versions custom-dl-optimizer
python -m pip install --upgrade custom-dl-optimizer==2.2.0
python -c "import custom_dl_optimizer as package; print(package.__version__)"
```

Do not rerun a failed upload blindly. PyPI does not permit replacing files for an existing version; inspect the workflow log and increment the version when any distribution was already accepted.

### Manual Twine Upload

When publishing manually, delete old build artifacts before creating the distributions, then upload only the new version:

```powershell
Remove-Item -Recurse -Force build, dist, custom_dl_optimizer.egg-info -ErrorAction SilentlyContinue
python -m build
python -m twine check dist/*
python -m twine upload --verbose dist/custom_dl_optimizer-2.2.0*
```

Use `__token__` as the username and the PyPI API token as the password. A `400 Bad Request` after a version has appeared on PyPI normally means that the filename already exists; increment the version instead of retrying the same file.

## TestPyPI

Manual TestPyPI uploads are optional:

```bash
python -m twine upload --repository testpypi dist/*
python -m pip install --extra-index-url https://test.pypi.org/simple/ custom-dl-optimizer
```

Never commit a PyPI token. If a token was ever exposed outside the ignored local file, revoke it before publishing.
