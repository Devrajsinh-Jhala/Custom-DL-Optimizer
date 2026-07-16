# v3.0.0 Release Checklist

Release date: 2026-07-17

## Required Gates

- [ ] Version is `3.0.0` in package metadata, source fallback, changelog, and citation metadata.
- [ ] Ruff passes for package, tests, and examples.
- [ ] All unit and provider-contract tests pass.
- [ ] Research notebook structure validation passes.
- [ ] Wheel and source distribution pass Twine metadata checks.
- [ ] The wheel installs into a clean virtual environment and imports as `3.0.0`.
- [ ] The installed wheel completes a CPU v3 smoke selection and writes a schema-v3 report.
- [ ] Landing page and guide render at desktop and mobile widths with no horizontal overflow.
- [ ] Repository contains no API token, local cache, build directory, or unrelated paper draft.
- [ ] CI is green on Python 3.10, 3.12, and 3.13.
- [ ] GitHub release tag exactly matches `v3.0.0`.
- [ ] PyPI lists both the v3 wheel and source distribution.
- [ ] GitHub Pages serves the v3 API and confidence-gate copy.

## Claim Boundary

- Do not claim state of the art without the completed literature review and reproduced baselines.
- Treat the historical Tesla T4 table as pilot evidence, not a v3 benchmark.
- Do not claim MediaTek NPU support until a provider is validated on a named SoC and SDK.
- Report regressions, failed providers, and retained baselines alongside selected wins.
