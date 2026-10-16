#!/usr/bin/env bash
# Commit the given paths and push, rebasing over the other personas' pushes.
set -u
msg="$1"; shift
git config user.name "aiva-studio[bot]"
git config user.email "aiva-studio@users.noreply.github.com"
git add -A -- "$@"
if git diff --cached --quiet; then echo "nothing to commit"; exit 0; fi
git commit -q -m "$msg"
for i in 1 2 3 4 5 6; do
  if git pull -q --rebase origin "${GITHUB_REF_NAME:-main}" && git push -q origin "HEAD:${GITHUB_REF_NAME:-main}"; then exit 0; fi
  sleep $((i * 7))
done
echo "push failed"; exit 1
