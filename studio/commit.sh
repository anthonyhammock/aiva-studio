#!/usr/bin/env bash
# Commit this job's output paths on top of the latest branch and push. Each job
# owns its persona's folders, so they replace whatever is there; other jobs'
# pushes in between are picked up by resetting to the remote first.
set -u
msg="$1"; shift
branch="${GITHUB_REF_NAME:-main}"
git config user.name "aiva-studio[bot]"
git config user.email "aiva-studio@users.noreply.github.com"
keep=$(mktemp -d)
for p in "$@"; do
  if [ -e "$p" ]; then mkdir -p "$keep/$(dirname "$p")" && cp -a "$p" "$keep/$p"; fi
done
for i in 1 2 3 4 5 6; do
  git fetch -q origin "$branch" && git reset -q --hard "origin/$branch" || { sleep $((i * 5)); continue; }
  for p in "$@"; do
    rm -rf "$p"
    if [ -e "$keep/$p" ]; then mkdir -p "$(dirname "$p")" && cp -a "$keep/$p" "$p"; fi
  done
  git add -A -- "$@"
  if git diff --cached --quiet; then echo "nothing to commit"; exit 0; fi
  git commit -q -m "$msg"
  if git push -q origin "HEAD:$branch"; then exit 0; fi
  sleep $((i * 7))
done
echo "push failed"; exit 1
