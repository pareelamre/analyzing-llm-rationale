#!/usr/bin/env bash
set -euo pipefail

remote="${1:-origin}"
branch="${2:-main}"
retries="${GIT_PUSH_RETRIES:-5}"
sleep_s="${GIT_PUSH_RETRY_SLEEP_S:-5}"
keep_theirs_csv="${GIT_REBASE_KEEP_THEIRS:-}"

resolve_allowed_rebase_conflicts() {
  local conflicted file
  conflicted="$(git diff --name-only --diff-filter=U || true)"
  if [ -z "$conflicted" ] || [ -z "$keep_theirs_csv" ]; then
    return 1
  fi

  while IFS= read -r file; do
    if ! printf '%s\n' "$keep_theirs_csv" | tr ',' '\n' | grep -Fx -- "$file" >/dev/null; then
      return 1
    fi
  done <<< "$conflicted"

  while IFS= read -r file; do
    git checkout --theirs -- "$file"
    git add "$file"
  done <<< "$conflicted"

  GIT_EDITOR=true git rebase --continue
}

for attempt in $(seq 1 "$retries"); do
  if git pull --rebase --autostash "$remote" "$branch"; then
    if git push "$remote" "HEAD:$branch"; then
      exit 0
    fi
  else
    if resolve_allowed_rebase_conflicts; then
      if git push "$remote" "HEAD:$branch"; then
        exit 0
      fi
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
