#!/bin/sh
# Deploy or roll back the Foresea stack on the box.
#
#   deploy.sh            # pull latest main, rebuild, health-check, auto-rollback
#   deploy.sh --rollback # return to the previous known-good image
#
# The image is built on the box (compose builds for the host architecture, so
# this works on ARM Ampere without buildx). Compose rebuilds reuse the same
# tag (foresea-app:latest), so the previous image is recorded by ID in
# .deployed-image and retagged back on rollback.
#
# Health gate: /ready returns 503 until the app finishes initializing (RAG
# embedder load), so the script polls /ready, not /health. The compose stack
# does not publish port 8000 (only Caddy's 80/443), so the probe runs inside
# the app container — the same check the compose healthcheck uses. On failure
# the previous image is restored and the failure is left in the logs.
set -eu

DEPLOY_DIR="${FORESEA_DEPLOY_DIR:-/opt/foresea/deploy/vps}"
INSTALL_DIR="$(cd "$DEPLOY_DIR/../.." && pwd)"
STATE_FILE="$DEPLOY_DIR/.deployed-image"
READY_TIMEOUT="${FORESEA_READY_TIMEOUT:-300}"

cd "$DEPLOY_DIR"

log() { echo "[deploy] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

# The image name compose built the app container from. Compose derives it from
# the project name (foresea-app:latest), so read it back from the container
# instead of guessing — a project rename then cannot break rollback.
current_app_image() {
	cid="$(docker compose ps -q app 2>/dev/null || true)"
	if [ -n "$cid" ]; then
		docker inspect --format '{{.Config.Image}}' "$cid"
	fi
}

ready_ok() {
	cid="$(docker compose ps -q app 2>/dev/null || true)"
	[ -n "$cid" ] || return 1
	docker exec "$cid" python -c \
		"import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/ready')" \
		>/dev/null 2>&1
}

restore_previous() {
	prev_id="$(cat "$STATE_FILE")"
	app_image="$(current_app_image)"
	[ -n "$app_image" ] || app_image="foresea-app:latest"
	log "restoring previous image ${prev_id#sha256:} as $app_image"
	docker tag "$prev_id" "$app_image"
	docker compose up -d --no-deps --force-recreate --no-build app
}

if [ "${1:-}" = "--rollback" ]; then
	if [ ! -f "$STATE_FILE" ]; then
		log "no previous image recorded in $STATE_FILE; nothing to roll back to"
		exit 1
	fi
	restore_previous
	log "rollback complete; verify with: docker compose logs --tail 50 app"
	exit 0
fi

# ── 1. Update the checkout. The live database lives on the foresea_state
# volume and .env is gitignored, so a hard reset cannot touch either.
git -C "$INSTALL_DIR" fetch origin main
git -C "$INSTALL_DIR" reset --hard origin/main
sha="$(git -C "$INSTALL_DIR" rev-parse --short HEAD)"
log "checkout at $sha"

# ── 2. Record the running image ID for rollback before touching anything.
app_cid="$(docker compose ps -q app || true)"
if [ -n "$app_cid" ]; then
	prev_id="$(docker inspect --format '{{.Image}}' "$app_cid")"
	printf '%s\n' "$prev_id" > "$STATE_FILE"
	log "previous image recorded: ${prev_id#sha256:}"
else
	log "no running app container; first deploy"
fi

# ── 3. Rebuild and restart.
log "building image (first build downloads torch; allow several minutes)"
docker compose build --pull app
docker compose up -d app

# ── 4. Health gate. /ready flips to 200 only after the app has initialized.
# Probed inside the container because port 8000 is not published to the host.
log "waiting up to ${READY_TIMEOUT}s for /ready (in-container probe)"
i=0
while [ "$i" -lt "$READY_TIMEOUT" ]; do
	if ready_ok; then
		log "deploy healthy at commit $sha"
		exit 0
	fi
	i=$((i + 5))
	sleep 5
done

log "ERROR: /ready did not pass within ${READY_TIMEOUT}s"
if [ -f "$STATE_FILE" ] && [ -n "${prev_id:-}" ]; then
	log "auto-rolling back"
	restore_previous
	log "rolled back; investigate with: docker compose logs --tail 200 app"
else
	log "no previous image recorded; leaving the failed stack up for inspection"
fi
exit 1