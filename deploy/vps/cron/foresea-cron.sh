#!/bin/sh
# Cron entry point for the Foresea box.
#
# Replaces the five Cloud Scheduler jobs. Three of them only dispatch GitHub
# Actions workflows -- those run on GitHub, not GCP, so they are plain curl
# calls here. The other one touches the twin runtime.
#
# Install with:
#   install -m 755 foresea-cron.sh /opt/foresea/bin/foresea-cron.sh
#   crontab /opt/foresea/deploy/vps/crontab
set -eu

ENV_FILE="${FORESEA_ENV_FILE:-/opt/foresea/deploy/vps/.env}"
# shellcheck disable=SC1090
. "$ENV_FILE"

REPO="pareelamre/analyzing-llm-rationale"
API="https://api.github.com/repos/${REPO}/actions/workflows"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

dispatch_workflow() {
	workflow="$1"
	log "dispatching ${workflow}"
	curl -fsS -X POST \
		-H "Authorization: Bearer ${GH_DISPATCH_TOKEN}" \
		-H "Accept: application/vnd.github+json" \
		-H "X-GitHub-Api-Version: 2022-11-28" \
		"${API}/${workflow}/dispatches" \
		-d '{"ref":"main"}' >/dev/null
}

case "${1:-}" in
track-record)
	dispatch_workflow track-record-tick.yml
	;;
forecast)
	dispatch_workflow track-record-forecast.yml
	;;
agent-trading)
	dispatch_workflow agent-trading-board-publish.yml
	;;
twin-due-work)
	# The twin maintenance service dispatches its own due jobs; this is the
	# same endpoint Cloud Scheduler called.
	log "triggering twin due work"
	curl -fsS -X POST \
		-H "Authorization: Bearer ${FORESEA_TWIN_DISPATCH_SECRET}" \
		-H "Content-Type: application/json" \
		"${FORESEA_TWIN_MAINTENANCE_URL}/internal/twin/dispatch" >/dev/null
	;;
metaculus-dispatch)
	# Replaces the Cloud Run job of the same name. It is the app image with a
	# different command, so it runs as a one-shot compose service.
	log "dispatching metaculus tournament workflow"
	cd "${FORESEA_DEPLOY_DIR:-/opt/foresea/deploy/vps}"
	docker compose --profile jobs run --rm metaculus-dispatch
	;;
*)
	echo "usage: $0 {track-record|forecast|agent-trading|twin-due-work|metaculus-dispatch}" >&2
	exit 2
	;;
esac
