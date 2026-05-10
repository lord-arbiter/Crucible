#!/usr/bin/env bash
# Push the frontend (and bundled precaches) to a HuggingFace Space.
#
# Usage:
#   HF_USER=your-username SPACE_NAME=crucible ./scripts/deploy_space.sh
#
# Prereqs:
#   - `huggingface-cli login` already done with a write-scoped token, OR
#     export HF_TOKEN before running.
#   - The Space exists, or you accept that this script creates it.

set -euo pipefail

HF_USER="${HF_USER:?set HF_USER}"
SPACE_NAME="${SPACE_NAME:-crucible}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Staging Space contents in $WORK"
cp -r "$ROOT/frontend/." "$WORK/"
mkdir -p "$WORK/data/precached"
if compgen -G "$ROOT/data/precached/*.json" > /dev/null; then
    cp "$ROOT/data/precached/"*.json "$WORK/data/precached/"
fi
mkdir -p "$WORK/src/prompts"
cp "$ROOT/src/__init__.py" "$WORK/src/__init__.py"
cp -r "$ROOT/src/prompts/." "$WORK/src/prompts/"

if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli repo create "$HF_USER/$SPACE_NAME" --type space --space_sdk gradio -y || true
fi

cd "$WORK"
git init -q
git remote add origin "https://huggingface.co/spaces/$HF_USER/$SPACE_NAME"

if [[ -n "${HF_TOKEN:-}" ]]; then
    git remote set-url origin "https://$HF_USER:$HF_TOKEN@huggingface.co/spaces/$HF_USER/$SPACE_NAME"
fi

git add .
git -c user.email="crucible@local" -c user.name="$HF_USER" commit -m "Deploy Crucible frontend"
git branch -M main
git push -u origin main --force

echo "Pushed to https://huggingface.co/spaces/$HF_USER/$SPACE_NAME"
