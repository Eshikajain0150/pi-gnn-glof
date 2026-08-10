#!/usr/bin/env python3
"""
Fix 2 of 2 for pi-gnn-glof.

Windows git rewrote the line endings of some committed evidence files, which
broke the per-file SHA-256 checksums. This restores the exact original bytes
from the archives still present in git history and adds a .gitattributes rule
so git never rewrites evidence bytes again.

Usage: put this file in the repository root and run

    python fix_lineendings.py
"""
from __future__ import annotations
import hashlib, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EV = ROOT / "evidence"

if not (ROOT / ".git").is_dir():
    sys.exit("ERROR: run this inside the pi-gnn-glof git repository root.")


def git(*args, **kw):
    return subprocess.run(("git",) + args, cwd=ROOT, capture_output=True, **kw)


print("1/5  writing .gitattributes so git stops rewriting evidence bytes")
attrs = ROOT / ".gitattributes"
rule = ("# evidence bytes are checksum-verified: never convert line endings\n"
        "evidence/** -text\n"
        "*.sha256 -text\n")
existing = attrs.read_text(encoding="utf-8") if attrs.exists() else ""
if "evidence/** -text" not in existing:
    attrs.write_text((existing.rstrip("\n") + "\n" + rule) if existing.strip() else rule,
                     encoding="utf-8")
    print("     .gitattributes written")
else:
    print("     already present")

print("2/5  recovering the original archives from git history")
tmp = Path(tempfile.mkdtemp())
found = 0
for commit in ("HEAD~1", "HEAD~2", "HEAD~3"):
    for name in ("causal30_v1_release.zip", "causal30_v1_calendar_release.zip"):
        out = tmp / name
        if out.exists():
            continue
        r = git("show", f"{commit}:evidence/{name}")
        if r.returncode == 0 and r.stdout[:2] == b"PK":
            out.write_bytes(r.stdout)
            print(f"     recovered {name} from {commit}")
            found += 1
if found == 0:
    sys.exit("ERROR: could not recover the archives from git history.")

print("3/5  restoring exact evidence bytes")
for name, folder in (("causal30_v1_release.zip", "causal30_v1"),
                     ("causal30_v1_calendar_release.zip", "causal30_v1_calendar")):
    z = tmp / name
    if not z.exists():
        continue
    target = EV / folder
    if target.exists():
        shutil.rmtree(target)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(EV)
    print(f"     restored evidence/{folder}/")
shutil.rmtree(tmp, ignore_errors=True)

print("4/5  verifying every checksum")
bad = []
for folder in ("causal30_v1", "causal30_v1_calendar"):
    manifest = EV / folder / "checksums.sha256"
    if not manifest.exists():
        continue
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        p = EV / folder / rel.strip().lstrip("./")
        if not p.exists():
            bad.append(f"{folder}/{rel} MISSING")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            bad.append(f"{folder}/{rel} MISMATCH")
if bad:
    print("     FAILED:", *bad, sep="\n       ")
    sys.exit(1)
print("     all checksums OK")

print("5/5  re-staging evidence with the corrected bytes")
git("rm", "-r", "-q", "--cached", "evidence")
git("add", "-A")
print("     staged")

print("""
DONE. Now run:

    python -m pytest -q            -> expect: 8 passed
    python scripts/quick_test.py   -> expect: QUICK TEST PASSED

then:

    git commit -m "Preserve exact evidence bytes and disable line-ending conversion"
    git push
""")
