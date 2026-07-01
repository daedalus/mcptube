#!/usr/bin/env bash
set -euo pipefail

# ── release.sh ──────────────────────────────────────────────────────────────
# Bump version → commit + tag → push → build → GitHub release.
# Usage:  ./tools/release.sh [patch|minor|major]   (default: patch)
# ────────────────────────────────────────────────────────────────────────────

PART="${1:-patch}"

# Ensure working tree is clean
if ! git diff --stat --exit-code; then
    echo "error: working tree has uncommitted changes; aborting." >&2
    exit 1
fi

# Ensure we are on master (or main)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "master" ] && [ "$BRANCH" != "main" ]; then
    echo "warning: releasing from branch '$BRANCH' (not master/main)"
fi

echo "==> bumpversion $PART"
bumpversion "$PART" --tag --verbose --commit

echo "==> push commit + tags"
git push
git push --tags

echo "==> build"
python -m build

echo "==> gh release"
TAG=$(git describe --tags --abbrev=0)
gh release create "$TAG" --generate-notes

echo "==> done: $TAG released"
