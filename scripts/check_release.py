#!/usr/bin/env python3
"""Fail closed on accidental credentials, private data, weights or oversized files."""

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    re.compile(r'sk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{24,}'),
    re.compile(r'gh[pousr]_[A-Za-z0-9]{24,}'),
    re.compile(r'github_pat_[A-Za-z0-9_]{30,}'),
    re.compile(r'LTAI[A-Za-z0-9]{12,}'),
    re.compile(r'AKIA[A-Z0-9]{16}'),
    re.compile(r'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----'),
]


def check():
    failures = []
    inspected = 0
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or any(p in path.relative_to(ROOT).parts for p in ['.git', '__pycache__', '.venv', 'outputs']):
            continue
        relative = path.relative_to(ROOT)
        inspected += 1
        if path.is_symlink():
            failures.append(f'{relative}: symlink requires review')
        if path.stat().st_size > 10_000_000 or path.suffix in ['.pt', '.pth', '.safetensors', '.bin', '.pem', '.key', '.log']:
            failures.append(f'{relative}: large or forbidden artifact')
        if 'private' in relative.parts or path.name.startswith('.env') or '.local.' in path.name:
            failures.append(f'{relative}: private configuration')
        if path.suffix == '.jsonl' and relative.parts[0] != 'examples':
            failures.append(f'{relative}: non-synthetic JSONL requires explicit data-release approval')
        if path.suffix not in ['.py', '.txt', '.md', '.json', '.jsonl', '.csv', '.cff', '.yml', '.yaml']:
            continue
        content = path.read_text(errors='replace')
        if any(pattern.search(content) for pattern in PATTERNS):
            failures.append(f'{relative}: possible credential (value intentionally hidden)')
        if re.search(r'/(?:Users|home)/[A-Za-z]|/(?:pools|data/projects)/', content):
            failures.append(f'{relative}: machine-specific absolute path')
    manifest = json.loads((ROOT / 'provenance/release_manifest.json').read_text())
    import hashlib
    for record in manifest['artifacts']:
        path = ROOT / record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            failures.append(f'{record["path"]}: exported artifact hash changed')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'PASS: {inspected} files checked; no detected secrets/private text/weights; result hashes verified.')


if __name__ == '__main__':
    check()
