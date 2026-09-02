#!/usr/bin/env bash
set -euo pipefail

remote="${1:-origin}"
branch="${2:-main}"
retries="${GIT_PUSH_RETRIES:-5}"
sleep_s="${GIT_PUSH_RETRY_SLEEP_S:-5}"
keep_theirs_csv="${GIT_REBASE_KEEP_THEIRS:-}"
publish_via_pr="${GIT_PUBLISH_VIA_PR:-1}"

slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//'
}

resolve_allowed_rebase_conflicts() {
  local conflicted file allowed
  conflicted="$(git diff --name-only --diff-filter=U || true)"
  if [ -z "$conflicted" ] || [ -z "$keep_theirs_csv" ]; then
    return 1
  fi

  while IFS= read -r file; do
    allowed=0
    while IFS= read -r keep_path; do
      [ -z "$keep_path" ] && continue
      if [[ "$keep_path" == "$file" || ( "$keep_path" == */ && "$file" == "$keep_path"* ) ]]; then
        allowed=1
        break
      fi
    done < <(printf '%s\n' "$keep_theirs_csv" | tr ',' '\n')
    if [ "$allowed" -ne 1 ]; then
      return 1
    fi
  done <<< "$conflicted"

  while IFS= read -r file; do
    echo "Keeping generated artifact from this publish: ${file}" >&2
    git checkout --theirs -- "$file"
    git add "$file"
  done <<< "$conflicted"

  GIT_EDITOR=true git rebase --continue
}

publish_through_pr() {
  if [ "$publish_via_pr" != "1" ] && [ "$publish_via_pr" != "true" ]; then
    return 1
  fi
  if ! command -v gh >/dev/null 2>&1; then
    echo "gh CLI is required for PR fallback publishing." >&2
    return 1
  fi
  if [ -z "${GH_TOKEN:-}" ] && [ -n "${GITHUB_TOKEN:-}" ]; then
    export GH_TOKEN="$GITHUB_TOKEN"
  fi
  if [ -z "${GH_TOKEN:-}" ]; then
    echo "GH_TOKEN is required for PR fallback publishing." >&2
    return 1
  fi

  local workflow run_id branch_slug update_branch title body pr_url
  workflow="$(slugify "${GITHUB_WORKFLOW:-scheduled-update}")"
  run_id="${GITHUB_RUN_ID:-$(date +%s)}"
  branch_slug="$(slugify "$branch")"
  update_branch="automation/${workflow}-${branch_slug}-${run_id}"
  title="${GIT_PUBLISH_PR_TITLE:-Automated ${GITHUB_WORKFLOW:-artifact} update}"
  body="${GIT_PUBLISH_PR_BODY:-Automated artifact update from ${GITHUB_WORKFLOW:-workflow} run ${GITHUB_RUN_ID:-local}.}"

  echo "Publishing through PR branch ${update_branch}." >&2
  git push --force-with-lease "$remote" "HEAD:refs/heads/${update_branch}"
  if ! pr_url="$(gh pr create --base "$branch" --head "$update_branch" --title "$title" --body "$body")"; then
    echo "Could not create fallback publish pull request." >&2
    return 1
  fi
  echo "Created ${pr_url}." >&2
  gh pr merge "$pr_url" --squash --delete-branch --subject "$title" --body "$body"
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

if publish_through_pr; then
  exit 0
fi

echo "Failed to push after ${retries} attempts." >&2
exit 1
