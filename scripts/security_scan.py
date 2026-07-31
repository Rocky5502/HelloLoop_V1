#!/usr/bin/env python3
"""Scan the public working tree and reachable Git blobs without printing matched values."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 5_000_000
EXCLUDED_GENERATED_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "generated",
    "runs",
}


def patterns() -> dict[str, re.Pattern[str]]:
    # Strings are split so this scanner does not report its own pattern declarations.
    return {
        "provider_key_shape": re.compile(r"(?:^|[^A-Za-z])" + "s" + r"k-[A-Za-z0-9_-]{16,}"),
        "github_token_shape": re.compile("gh" + r"[opusr]_[A-Za-z0-9_]{20,}"),
        "private_key": re.compile("BEGIN (?:RSA |EC |OPENSSH )?PRIVATE" + " KEY"),
        "absolute_home_path": re.compile("/(?:Us" + "ers|ho" + r"me)/[^/\s]+/"),
        "credential_assignment": re.compile(
            r"(?i)(?:api[_-]?key|password|authorization|bearer|access[_-]?token)[ \t]*[:=][ \t]*['\"][^'\"\r\n]{8,}"
        ),
        "private_ipv4": re.compile(
            r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)\d{1,3}\.\d{1,3}\b"
        ),
        "personal_email": re.compile(
            r"\b[A-Z0-9._%+-]+@(?!example\.(?:com|org|invalid)\b)(?!users\.noreply\.github\.com\b)[A-Z0-9.-]+\.[A-Z]{2,}\b",
            re.IGNORECASE,
        ),
        "confidential_venue_term": re.compile("AA" + "AI|Open" + "Review", re.IGNORECASE),
        "confidential_status_term": re.compile(r"submitted\s+to|under\s+review", re.IGNORECASE),
        "private_provider_host": re.compile("zhizeng" + "zeng", re.IGNORECASE),
    }


def inspect_text(label: str, content: str, findings: set[tuple[str, str]]) -> None:
    normalized = label.replace("\\", "/")
    if normalized.endswith("scripts/security_scan.py"):
        excluded = {"credential_assignment", "personal_email"}
    else:
        excluded = set()
    for category, pattern in patterns().items():
        if category not in excluded and pattern.search(content):
            findings.add((category, normalized))


def scan_worktree(findings: set[tuple[str, str]]) -> int:
    files = 0
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or any(part in EXCLUDED_GENERATED_DIRS for part in path.parts)
            or path.stat().st_size > MAX_TEXT_BYTES
        ):
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip"}:
            continue
        files += 1
        content = path.read_text(encoding="utf-8", errors="ignore")
        inspect_text(str(path.relative_to(ROOT)), content, findings)
    return files


def scan_history(findings: set[tuple[str, str]]) -> int:
    if not (ROOT / ".git").exists():
        return 0
    listing = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    objects = 0
    for entry in listing:
        object_id, _, name = entry.partition(" ")
        if not name or name.endswith((".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip")):
            continue
        size_result = subprocess.run(
            ["git", "cat-file", "-s", object_id],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if size_result.returncode or int(size_result.stdout.strip()) > MAX_TEXT_BYTES:
            continue
        blob = subprocess.run(
            ["git", "cat-file", "-p", object_id], cwd=ROOT, capture_output=True, check=True
        ).stdout.decode("utf-8", errors="ignore")
        objects += 1
        inspect_text(f"history:{object_id[:12]}:{name}", blob, findings)
    return objects


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree-only", action="store_true")
    args = parser.parse_args()
    findings: set[tuple[str, str]] = set()
    files = scan_worktree(findings)
    objects = 0 if args.worktree_only else scan_history(findings)
    print(f"security scan: {files} working-tree files, {objects} reachable Git objects")
    if findings:
        for category, location in sorted(findings):
            print(f"FAIL {category}: {location}")
        print(f"security scan failed with {len(findings)} location/category findings")
        return 1
    print("security scan passed: 0 findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
