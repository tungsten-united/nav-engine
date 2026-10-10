#!/usr/bin/env bash
# nav-api on Cloud Run: the API + debugging frontend, on the team bucket. Idempotent; only creates or updates.
#
#   deploy/cloudrun.sh setup     once: APIs, service account + bucket access, empty secrets
#   deploy/cloudrun.sh [deploy]  build the Dockerfile here (docker), push it, deploy; rerun after every change
#   deploy/cloudrun.sh url       the service URL
#   deploy/cloudrun.sh token     the API token the frontends send
#   deploy/cloudrun.sh secrets   copy the secret values from this machine into Secret Manager (after a change)
#
# Secrets (API token, Cloudflare Access service token, Jev key) come from this machine and are never printed:
# ~/.cloudflared/access-service-token.env, ./typesafe.api, and $TOKEN_FILE (the API token, generated once).
# The service reads them from Secret Manager (setup grants nav-worker access; that needs a project owner).
# SECRETS_FROM=env passes them as plain env vars instead, readable by project viewers in the service config.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT=tungsten-united
REGION=europe-southwest1
SERVICE=nav-api
BUCKET=tungsten-united-nav-recordings
SA="nav-worker@$PROJECT.iam.gserviceaccount.com"
# env var = secret name
SECRETS=(NAV_API_TOKEN=nav-api-token CF_ACCESS_CLIENT_ID=cf-access-client-id
         CF_ACCESS_CLIENT_SECRET=cf-access-client-secret TYPESAFE_API_KEY=typesafe-api-key)
REPO="$REGION-docker.pkg.dev/$PROJECT/cloud-run-source-deploy"  # Artifact Registry
TOKEN_FILE="${NAV_API_TOKEN_FILE:-$HOME/.config/nav-engine/api-token}"
g() { gcloud --project "$PROJECT" --quiet "$@"; }

secret_values() {  # NAME=value lines, from this machine
    (set -a; . "$HOME/.cloudflared/access-service-token.env"
     printf 'CF_ACCESS_CLIENT_ID=%s\nCF_ACCESS_CLIENT_SECRET=%s\n' "$CF_ACCESS_CLIENT_ID" "$CF_ACCESS_CLIENT_SECRET")
    [[ -f typesafe.api ]] && printf 'TYPESAFE_API_KEY=%s\n' "$(tr -d '\n' < typesafe.api)"
    if [[ ! -s "$TOKEN_FILE" ]]; then
        mkdir -p "$(dirname "$TOKEN_FILE")"
        (umask 077; openssl rand -hex 24 | tr -d '\n' > "$TOKEN_FILE")
    fi
    printf 'NAV_API_TOKEN=%s\n' "$(cat "$TOKEN_FILE")"
}

setup() {
    g services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
    g artifacts repositories describe cloud-run-source-deploy --location "$REGION" >/dev/null 2>&1 ||
        g artifacts repositories create cloud-run-source-deploy --location "$REGION" --repository-format docker
    g iam service-accounts describe "$SA" >/dev/null 2>&1 ||
        g iam service-accounts create nav-worker --display-name "nav-api Cloud Run worker"
    gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member "serviceAccount:$SA" \
        --role roles/storage.objectAdmin >/dev/null
    for pair in "${SECRETS[@]}"; do
        local name="${pair#*=}"
        g secrets describe "$name" >/dev/null 2>&1 || g secrets create "$name" --replication-policy automatic
        g secrets add-iam-policy-binding "$name" --member "serviceAccount:$SA" \
            --role roles/secretmanager.secretAccessor >/dev/null 2>&1 ||
            echo "  $name: can't grant nav-worker access (needs a project owner; see the top of $0)"
    done
    echo "setup done; now: $0 deploy"
}

put() {  # put <secret> : value on stdin
    g secrets versions add "$1" --data-file=- >/dev/null && echo "  $1: new version"
}

secrets() {
    local line name
    while IFS= read -r line; do
        for pair in "${SECRETS[@]}"; do
            [[ "${line%%=*}" == "${pair%%=*}" ]] && printf %s "${line#*=}" | put "${pair#*=}"
        done
    done < <(secret_values)
}

token() { secret_values >/dev/null; cat "$TOKEN_FILE"; echo; }

image() {  # build the Dockerfile and push it; prints the image reference
    local tag
    tag="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- nav frontend pyproject.toml uv.lock Dockerfile || echo -dirty)"
    docker build -q -t "$REPO/$SERVICE:$tag" . >&2
    gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin "https://$REGION-docker.pkg.dev" >/dev/null 2>&1
    docker push -q "$REPO/$SERVICE:$tag" >&2
    echo "$REPO/$SERVICE:$tag"
}

ENV_FILE=""
deploy() {
    local img secret_flags=()
    img=$(image)
    ENV_FILE=$(mktemp --suffix .yaml)
    trap 'rm -f "$ENV_FILE"' EXIT
    local env_file="$ENV_FILE"
    chmod 600 "$env_file"
    cp deploy/cloudrun.env.yaml "$env_file"
    if [[ "${SECRETS_FROM:-manager}" == manager ]]; then
        secret_flags=(--set-secrets "$(IFS=,; for p in "${SECRETS[@]}"; do printf '%s,' "${p%%=*}=${p#*=}:latest"; done)")
        secret_flags[1]="${secret_flags[1]%,}"
    else
        secret_values | python3 -c 'import json, sys
for line in sys.stdin.read().splitlines():
    k, v = line.split("=", 1)
    print(f"{k}: {json.dumps(v)}")' >> "$env_file"
    fi
    g run deploy "$SERVICE" --image "$img" --region "$REGION" \
        --service-account "$SA" --execution-environment gen2 \
        --add-volume "name=data,type=cloud-storage,bucket=$BUCKET,mount-options=implicit-dirs" \
        --add-volume-mount volume=data,mount-path=/data \
        --cpu 2 --memory 4Gi --no-cpu-throttling --min-instances 1 --max-instances 1 \
        --concurrency 40 --timeout 3600 --allow-unauthenticated \
        --env-vars-file "$env_file" "${secret_flags[@]}"
    url
}

url() { g run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)'; }

"${1:-deploy}"
