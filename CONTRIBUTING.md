# Contributing to spectune

We value pull requests that are small, clear, and easy to verify.

## Pull request standard

A good pull request:

- solves one problem and avoids unrelated cleanup;
- explains the motivation and the chosen approach;
- includes tests for behavior changes, or explains why tests are not applicable;
- updates documentation when users or public APIs are affected;
- contains no generated files, debug output, credentials, or drive-by formatting changes;
- is reviewable commit by commit and uses a concise, conventional title.

Use one of these title prefixes:

`feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `chore`, or `revert`.

Examples:

```text
feat: add a configurable search timeout
fix(web-search): preserve upstream error details
docs: document sandbox credentials
```

## Before opening a pull request

Install development dependencies and run the same checks as CI:

```bash
python -m pip install -e '.[dev]'
ruff format --check .
ruff check .
pytest -v
```

If a change cannot be covered by an automated test, include reproducible manual
verification in the pull request.

## Review

Authors should self-review the diff before requesting review. Reviewers evaluate:

1. scope and necessity;
2. correctness and failure behavior;
3. tests and reproducibility;
4. API and naming clarity;
5. maintenance cost and documentation.

All review conversations should be resolved, required checks should pass, and at
least one approval should be present before merging.
