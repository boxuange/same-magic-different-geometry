#!/usr/bin/env python3
"""Verify SHA-256 manifest against the immutable public research inputs.

This self-contained checker works with GitHub-downloaded ZIPs and Git checkouts.
Only recognised source text is LF-normalised to survive Windows line endings;
binary input arrays are hashed exactly as stored.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'SHA256_MANIFEST.json'
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.json'}
TEXT_NAMES = {'.gitignore', '.gitattributes'}


def canonical_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES:
        return data.replace(b'\r\n', b'\n')
    return data


def main() -> int:
    expected = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if expected.get('format') != 'SHA-256':
        print('FAIL: unsupported manifest format')
        return 1
    failed = []
    seen = set()
    for item in expected['files']:
        rel = item['path']
        if rel in seen or rel.startswith('/') or '..' in Path(rel).parts:
            failed.append(f'invalid/duplicate path: {rel}')
            continue
        seen.add(rel)
        file = ROOT / rel
        if not file.is_file():
            failed.append(f'missing: {rel}')
            continue
        data = canonical_bytes(file)
        digest = hashlib.sha256(data).hexdigest()
        if digest != item['sha256'] or len(data) != item['size_bytes']:
            failed.append(f'hash/size mismatch: {rel}')
    if failed:
        for err in failed:
            print('FAIL:', err)
        return 1
    print(f'PASS: {len(seen)} fixed distributed files match SHA256_MANIFEST.json.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
