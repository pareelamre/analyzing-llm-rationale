# Deploying Foresea on Oracle Cloud (OCI) Always Free

OCI's Always Free tier gives **2 ARM Ampere OCPUs and 12 GB RAM** (tenancy-wide,
not per instance), 200 GB block storage, and **10 TB/month egress**. The egress
allowance is what makes it viable — the GCP `e2-micro` free tier allows only
1 GB/month, which this app would burn through in roughly 280 page loads.

> **The limit was halved in June 2026.** It used to be 4 OCPU / 24 GB. Oracle
> made the change without a public announcement, and instances over the new cap
> were administratively disabled from 18 August 2026 and **deleted after 30
> days** unless the tenancy was brought within limits or upgraded to paid.
> Provisioning above 2 OCPU / 12 GB as an Always Free user is no longer
> possible, and a terminated over-limit instance cannot be recreated at the old
> size. **Stay at or below 2 OCPU / 12 GB total across all A1 instances.**

## Why ARM is fine here

Ampere instances are `aarch64`, and the app image was x86. This was verified
rather than assumed:

```
$ pip install --dry-run --platform linux/arm64 torch \
      --index-url https://download.pytorch.org/whl/cpu
resolved packages: 9
includes torch: True
```

PyTorch publishes aarch64 CPU wheels, so the dependency set resolves normally.
The compose file builds for the host architecture by default, so **no change is
needed** on an Ampere instance.

## 1. Create the instance

In the OCI console: **Compute → Instances → Create instance**.

| Field | Value |
|---|---|
| Image | Canonical Ubuntu 22.04 (or 24.04) |
| Shape | **VM.Standard.A1.Flex** (Ampere ARM) |
| OCPUs | **2** (this is the whole free allowance) |
| Memory | **12 GB** (this is the whole free allowance) |
| Boot volume | 50 GB (up to 200 free) |
| SSH key | paste `~/.ssh/oci_foresea.pub` |

Do not add a second A1 instance: the 2 OCPU / 12 GB is **tenancy-wide**, so a
second instance puts the tenancy over the cap and risks all of them being
disabled and deleted.

A dedicated key was generated for this at `~/.ssh/oci_foresea`:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICRdgB/qQplxzcwRsxjCyyzZhnjP/rk0vKo9+QfhvdWP foresea-oci
```

**Capacity is the known friction.** Ampere instances are often unavailable in
popular regions. If creation fails with "Out of host capacity", retry, try a
different availability domain, or try a less busy region (Mumbai `ap-mumbai-1`
is closest to you). This is the one step that may take persistence.

## 2. Open the firewall

OCI blocks everything by default, and there are **two** layers — missing the
second is the usual cause of "the server is up but unreachable".

**Layer 1 — Security List / NSG** (console): add ingress rules for
`0.0.0.0/0` on TCP **80** and **443**.

**Layer 2 — the instance's own iptables.** Oracle's Ubuntu images ship with a
default REJECT rule that overrides the security list:

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

## 3. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker "$USER"
newgrp docker
```

## 4. Get the code

The migration is merged to `main` (PR #669), so a plain clone has everything:

```bash
sudo mkdir -p /opt/foresea && sudo chown "$USER" /opt/foresea
git clone https://github.com/pareelamre/analyzing-llm-rationale.git /opt/foresea
cd /opt/foresea
```

## 5. Copy the data

From your workstation, with the migrated database:

```bash
scp -i ~/.ssh/oci_foresea foresea.sqlite3 ubuntu@<instance-ip>:/opt/foresea/
```

If you need to re-copy from Datastore (for example after the writers have moved
on), follow `deploy/vps/README.md` §1 — including the quiesce step, which is
required for `--verify` to report "All kinds match".

## 6. Configure and start

```bash
cd /opt/foresea/deploy/vps
cp .env.example .env && chmod 600 .env
# fill in the secrets; note server.py reads SCADS_AI_API_KEY, not SCADS_API_KEY
nano .env

docker volume create foresea_state
docker run --rm -v foresea_state:/data -v /opt/foresea:/src:ro alpine \
  cp /src/foresea.sqlite3 /data/foresea.sqlite3

docker compose up -d --build
docker compose logs -f app
```

The first build compiles nothing (wheels are prebuilt) but does download
PyTorch, so allow a few minutes. The first boot then downloads the RAG
embedding model into the `models` volume.

## 7. Verify

```bash
curl -fsS localhost:8000/health    # {"status":"ok"}
curl -fsS localhost:8000/ready     # {"ready":true,...}
```

`/ready` returns 503 with `provider_configured: false` if `SCADS_AI_API_KEY`
is unset — that is a missing LLM key, not a storage problem.

Then point `foresea.ink` at the instance's public IP. Caddy issues the TLS
certificate on first request.

## Resource sizing

Measured under a hard 1 GB cgroup: **146 MB idle, 611 MB peak** with the RAG
embedder loaded. The 12 GB Ampere instance is far beyond what this needs, which
leaves room for the twin runtime and marketd later.

## Cost

Always Free covers 2 OCPU / 12 GB / 200 GB / 10 TB egress. Staying inside those
limits costs **nothing**. The risks are:

- **Exceeding the A1 cap.** The 2 OCPU / 12 GB is tenancy-wide. A second A1
  instance puts the tenancy over it, and over-limit instances are disabled and
  then deleted after 30 days.
- **Exceeding 200 GB of block storage**, which is billed.
- **Leaving a non-A1 paid shape running** (for example an E4 or GPU instance),
  which is not covered by Always Free at all.
