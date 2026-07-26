#!/usr/bin/env bash
set -euo pipefail

remote="${1:-origin}"
branch="${2:-main}"
retries="${GIT_PUSH_RETRIES:-5}"
sleep_s="${GIT_PUSH_RETRY_SLEEP_S:-5}"

for attempt in $(seq 1 "$retries"); do
  if git pull --rebase --autostash "$remote" "$branch"; then
    if git push "$remote" "HEAD:$branch"; then
      exit 0
    fi
  fi
  git rebase --abort >/dev/null 2>&1 || true
  if [ "$attempt" -lt "$retries" ]; then
    echo "Push attempt $attempt/$retries failed; retrying in ${sleep_s}s..." >&2
    sleep "$sleep_s"
  fi
done

echo "Failed to push after ${retries} attempts." >&2
exit 1
