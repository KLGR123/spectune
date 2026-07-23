"""Validate pull request metadata without third-party dependencies."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

TITLE_PATTERN = re.compile(
    r"^(feat|fix|docs|test|refactor|perf|build|ci|chore|revert)"
    r"(\([a-z0-9][a-z0-9._/-]*\))?!?: \S.+$"
)
REQUIRED_SECTIONS = ("Why", "What changed", "Verification")
REQUIRED_CHECKS = (
    "This PR addresses one focused concern and contains no unrelated changes.",
    "I reviewed my own diff and removed debug output, generated files, and secrets.",
    "I added or updated tests, or explained above why tests are not applicable.",
    "I updated user-facing documentation when behavior or public APIs changed.",
)


def visible_markdown(text: str) -> str:
    """Remove template comments before checking for substantive content."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def section_content(body: str, heading: str) -> str | None:
    match = re.search(
        rf"^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s+|\Z)",
        body,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def validate(title: str, body: str) -> list[str]:
    errors: list[str] = []

    if len(title) > 80:
        errors.append("Keep the PR title at 80 characters or fewer.")
    if not TITLE_PATTERN.fullmatch(title):
        errors.append("Use a conventional title such as `feat: add timeout handling`.")

    body = visible_markdown(body)
    for heading in REQUIRED_SECTIONS:
        content = section_content(body, heading)
        if not content:
            errors.append(f"Fill in the `## {heading}` section.")

    for item in REQUIRED_CHECKS:
        if not re.search(rf"^-\s+\[[xX]\]\s+{re.escape(item)}\s*$", body, re.MULTILINE):
            errors.append(f"Complete the checklist item: {item}")

    return errors


def main() -> int:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        print("GITHUB_EVENT_PATH is not set.", file=sys.stderr)
        return 2

    event = json.loads(Path(event_path).read_text())
    pull_request = event["pull_request"]
    errors = validate(pull_request["title"], pull_request.get("body") or "")

    if errors:
        print("Pull request quality check failed:\n")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Pull request title and description meet the contribution standard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
