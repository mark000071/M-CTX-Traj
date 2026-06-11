#!/usr/bin/env bash
# Initialise the M-CTX_Traj GitHub repository from this directory.
#
# This script does NOT push automatically — it stops just before
# `git push` so you can review what will be committed.
#
# Prerequisites:
#   1. You have created the empty GitHub repo
#      https://github.com/mark000071/M-CTX_Traj
#   2. You have configured `gh auth login` OR a Personal Access Token in
#      ~/.netrc / git credential helper.
#
# Usage:
#   bash scripts/init_github.sh                          # set up + commit + show
#   bash scripts/init_github.sh push                     # also push to origin
#
set -euo pipefail

REMOTE="git@github.com:mark000071/M-CTX_Traj.git"
HTTPS_REMOTE="https://github.com/mark000071/M-CTX_Traj.git"

cd "$(dirname "$0")/.."

if [ ! -d .git ]; then
    echo "[init] git init"
    git init -b main
fi

# Remote
if ! git remote get-url origin >/dev/null 2>&1; then
    echo "[init] adding remote origin → $REMOTE"
    git remote add origin "$REMOTE"
else
    echo "[init] remote origin already set"
fi

# Show what will be committed
echo
echo "=== files staged for first commit ==="
git add -A
git status --short

echo
echo "=== file count and size ==="
find . -type f -not -path "./.git/*" | wc -l
du -sh --exclude='.git' .

# First commit
if [ -z "$(git log --oneline 2>/dev/null)" ]; then
    git commit -m "Initial commit: M-CTX reference implementation, baselines, paper.

Source code, experiment harness, audit chain, case-study figures, and
LaTeX source of the ICDE 2027 submission *M-CTX: Exact and Scalable
Spatial Context Retrieval for Trajectory Analytics*.

* src/                core indices (BR-LZ, BR-LZ_opt, Flood, LMSFC, …)
* experiments/scripts/ unified benchmark harness (n=10 trials)
* analysis/            audit + figure generators
* figures/             12 case-study panels (300 DPI)
* paper/               IEEE-Conference LaTeX + 13-page submission PDF
* README.md            quick start
* REPRODUCING.md       per-table reproduction recipe
* LICENSE              MIT
* CITATION.cff         standard citation metadata"
    echo "[done] first commit created"
else
    echo "[skip] commit history exists; skipping initial commit"
fi

if [ "${1:-}" = "push" ]; then
    echo
    echo "=== pushing to origin/main ==="
    git push -u origin main || {
        echo "ssh push failed; retrying with HTTPS"
        git remote set-url origin "$HTTPS_REMOTE"
        git push -u origin main
    }
    echo "[done] pushed to $REMOTE"
else
    echo
    echo "=== READY TO PUSH ==="
    echo "Review with:  git log --oneline"
    echo "Push with:    bash scripts/init_github.sh push"
    echo "          or: git push -u origin main"
fi
