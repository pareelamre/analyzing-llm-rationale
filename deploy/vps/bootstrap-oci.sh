#!/bin/sh
# One-time bootstrap for a fresh OCI Ampere instance (Ubuntu 22.04/24.04).
# Automates OCI.md sections 2-4 plus the cron install; idempotent, safe to
# re-run. Run as the login user (ubuntu); sudo is used where needed.
#
#   scp -i ~/.ssh/oci_foresea deploy/vps/bootstrap-oci.sh ubuntu@<instance-ip>:
#   ssh -i ~/.ssh/oci_foresea ubuntu@<instance-ip> 'sh bootstrap-oci.sh'
#
# It does NOT start the stack: fill in deploy/vps/.env and seed the database
# first (see the "next steps" printed at the end), then deploy by hand with
# deploy/vps/deploy.sh or by pushing to main with the CI workflow enabled.
#
# Note: the crontab hardcodes /opt/foresea paths, so a custom
# FORESEA_INSTALL_DIR also needs the crontab edited.
set -eu

INSTALL_DIR="${FORESEA_INSTALL_DIR:-/opt/foresea}"
REPO="${FORESEA_REPO:-https://github.com/pareelamre/analyzing-llm-rationale.git}"
DEPLOY_DIR="$INSTALL_DIR/deploy/vps"

log() { echo "[bootstrap] $*"; }

# ── 1. Instance firewall (OCI.md §2, layer 2). Oracle's Ubuntu images ship a
# REJECT-all ruleset that overrides the security list; insert the accept rules
# before it rather than at a hardcoded position.
for port in 80 443; do
	if sudo iptables -C INPUT -m state --state NEW -p tcp --dport "$port" -j ACCEPT 2>/dev/null; then
		log "iptables: port $port already open"
	else
		reject_line="$(sudo iptables -L INPUT --line-numbers -n | awk '/REJECT/ { print $1; exit }')"
		if [ -n "$reject_line" ]; then
			sudo iptables -I INPUT "$reject_line" -m state --state NEW -p tcp --dport "$port" -j ACCEPT
		else
			sudo iptables -A INPUT -m state --state NEW -p tcp --dport "$port" -j ACCEPT
		fi
		log "iptables: opened port $port"
	fi
done
if command -v netfilter-persistent >/dev/null 2>&1; then
	sudo netfilter-persistent save >/dev/null
	log "iptables rules saved"
fi

# ── 2. Docker + friends (OCI.md §3).
if ! command -v docker >/dev/null 2>&1; then
	sudo apt-get update -qq
	sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io docker-compose-v2 git cron >/dev/null
	sudo usermod -aG docker "$(id -un)"
	log "docker installed; group membership applies on the next login"
else
	log "docker already installed"
fi
sudo systemctl enable --now cron >/dev/null 2>&1 || true

# ── 3. Code (OCI.md §4).
if [ -d "$INSTALL_DIR/.git" ]; then
	git -C "$INSTALL_DIR" fetch origin main
	git -C "$INSTALL_DIR" reset --hard origin/main
	log "repo updated to $(git -C "$INSTALL_DIR" rev-parse --short HEAD)"
else
	sudo mkdir -p "$INSTALL_DIR"
	sudo chown "$(id -u):$(id -g)" "$INSTALL_DIR"
	git clone "$REPO" "$INSTALL_DIR"
	log "cloned $REPO"
fi

# ── 4. Environment template. Secrets are filled in by hand; the file is
# gitignored so re-runs never clobber it.
if [ ! -f "$DEPLOY_DIR/.env" ]; then
	cp "$DEPLOY_DIR/.env.example" "$DEPLOY_DIR/.env"
	chmod 600 "$DEPLOY_DIR/.env"
	log "created $DEPLOY_DIR/.env from the template -- fill it in before deploying"
else
	log ".env already present, left untouched"
fi

# ── 5. State volume, seeded from a database copied next to the checkout
# (OCI.md §5) if one is present.
sudo docker volume create foresea_state >/dev/null
seed="${FORESEA_DB_SEED:-$INSTALL_DIR/foresea.sqlite3}"
if [ -f "$seed" ]; then
	if ! sudo docker run --rm -v foresea_state:/data alpine test -f /data/foresea.sqlite3 >/dev/null 2>&1; then
		sudo docker run --rm -v foresea_state:/data -v "$INSTALL_DIR":/src:ro alpine \
			cp /src/foresea.sqlite3 /data/foresea.sqlite3
		log "seeded the foresea_state volume from $seed"
	else
		log "foresea_state volume already has a database, left untouched"
	fi
else
	log "no database at $seed -- copy one over before deploying (OCI.md §5)"
fi

# ── 6. Cron schedule (README §2b). The crontab appends to
# /var/log/foresea-cron.log, which only exists once created here -- cron runs
# as this user, not root, so a missing or root-owned log file fails every
# entry silently.
install -d -m 755 "$INSTALL_DIR/bin"
install -m 755 "$DEPLOY_DIR/cron/foresea-cron.sh" "$INSTALL_DIR/bin/"
sudo touch /var/log/foresea-cron.log
sudo chown "$(id -u):$(id -g)" /var/log/foresea-cron.log
crontab "$DEPLOY_DIR/crontab"
log "cron installed: $(crontab -l | grep -c . || true) lines"

log "next steps:"
log "  1. nano $DEPLOY_DIR/.env          # fill in the secrets"
log "  2. copy the database if not done above (OCI.md §5)"
log "  3. deploy: sh $DEPLOY_DIR/deploy.sh"
log "     (or set the OCI_* secrets + OCI_DEPLOY_ENABLED=true and push to main)"