# Shared application image for `worker` (APScheduler jobs) and `ais` (AIS
# WebSocket daemon) — both import the same clients/orchestration/analytics
# tree and the same requirements.txt, so one image, differentiated by
# docker-compose.yml's `command:`, not two Dockerfiles that would let
# their dependency installs drift apart. See CLAUDE.md § Deployment.
#
# Pinned to 3.12, not `python:3-slim`/`latest`: local development runs
# 3.14, but the deployed server's system Python — and the venv the live
# verifications actually ran under — is 3.12. `timescale/timescaledb
# :latest-pg16` already cost a debugging session to a moving base tag in
# Phase 2; this pin is deliberate, not an oversight.
FROM python:3.12-slim

WORKDIR /app

# Dependencies as their own layer, installed before application code is
# copied in: a source-only edit then only invalidates the COPY . . layer,
# not a full scientific-stack (pandas/scipy/statsmodels/scikit-learn)
# reinstall on a slow box.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Unprivileged by default: root inside a container isn't root on the
# host, but it's one container-escape away from it, and neither the
# worker nor the AIS daemon needs it.
RUN groupadd --system app \
    && useradd --system --gid app --no-create-home --shell /usr/sbin/nologin app
USER app

# Overridden per-service by docker-compose.yml's `command:`
# (`python -m worker.main` / `python -m ais.main`); this default just
# makes the image runnable standalone without one.
CMD ["python", "-m", "worker.main"]
