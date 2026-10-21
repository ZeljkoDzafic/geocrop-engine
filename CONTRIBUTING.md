# Contributing

Issues and pull requests are welcome.

1. Fork, create a branch, install with `pip install -e '.[dev]'`.
2. Keep functions file-in/file-out and configuration-driven; no study-specific constants in the package.
3. Add tests (`pytest`) for new behaviour; synthetic fixtures live in `geocrop.demo` and `tests/conftest.py`.
4. Run `ruff check src tests` and `pytest` before opening a pull request.
5. Document user-facing changes in `docs/` and `CHANGELOG.md`.

Never commit satellite products, parcel/LPIS data, field observations, credentials or model weights.
Scientific changes (metrics, sampling, validation) should cite their source in the docstring.
