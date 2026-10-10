#!/usr/bin/env bash
# The two Cloud Run services, one image (the Dockerfile), on the team bucket. Idempotent; only creates or updates.
#   map-api  recordings, map builds, review, map files + the debugging frontend; bucket read-write; runs as nav-worker
#   nav-api  the live API for Orient's orchestrator: published maps, routes, localization; bucket read-only; runs as nav-api
#
#   deploy/cloudrun.sh setup                     once: APIs, service accounts + bucket access, empty secrets
#   deploy/cloudrun.sh [deploy] [map-api|nav-api|all]   build the image here (docker), push it, deploy (default all)
#   deploy/cloudrun.sh url [map-api|nav-api]     the service URL (default map-api)
#   deploy/cloudrun.sh token [map-api|nav-api]   the API token that service wants (default map-api)
#   deploy/cloudrun.sh secrets                   copy the secret values from this machine into Secret Manager
#   deploy/cloudrun.sh ci-setup                  once: let GitHub Actions deploy (.github/workflows/deploy.yml)
#
# Secrets (API tokens, Cloudflare Access service token, Jev key) come from this machine and are never printed:
# ~/.cloudflared/access-service-token.env, ./typesafe.api, and the token files below (generated once). The services
# read them from Secret Manager (setup grants each service account its own; that needs a project owner).
# SECRETS_FROM=env passes them as plain env vars instead, readable by project viewers in the service config.
#
# Tokens: nav-api's is the API token from before the split ($NAV_TOKEN_FILE), so the app's map reads keep working.
# map-api's ($MAP_TOKEN_FILE) starts as a copy of it; to rotate one, write a new value to its file, then run
# `secrets` and `deploy <service>`.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT=tungsten-united
REGION=europe-southwest1
BUCKET=tungsten-united-nav-recordings
IMAGE=nav-engine
REPO="$REGION-docker.pkg.dev/$PROJECT/cloud-run-source-deploy"  # Artifact Registry
NAV_TOKEN_FILE="${NAV_API_TOKEN_FILE:-$HOME/.config/nav-engine/api-token}"
MAP_TOKEN_FILE="${MAP_API_TOKEN_FILE:-$HOME/.config/nav-engine/map-api-token}"
SERVICES=(map-api nav-api)
g() { gcloud --project "$PROJECT" --quiet "$@"; }

sa() {  # the service account a service runs as
    case "$1" in
        map-api) echo "nav-worker@$PROJECT.iam.gserviceaccount.com" ;;
        nav-api) echo "nav-api@$PROJECT.iam.gserviceaccount.com" ;;
        *) echo "unknown service: $1 (map-api or nav-api)" >&2; exit 1 ;;
    esac
}

secrets_of() {  # ENV_VAR=secret-name, per service
    case "$1" in
        map-api) echo NAV_API_TOKEN=map-api-token CF_ACCESS_CLIENT_ID=cf-access-client-id \
                      CF_ACCESS_CLIENT_SECRET=cf-access-client-secret TYPESAFE_API_KEY=typesafe-api-key ;;
        nav-api) echo NAV_API_TOKEN=nav-api-token CF_ACCESS_CLIENT_ID=cf-access-client-id \
                      CF_ACCESS_CLIENT_SECRET=cf-access-client-secret ;;
        *) echo "unknown service: $1" >&2; exit 1 ;;
    esac
}

token_file() {  # the token file of a service, created on first use
    local f
    case "$1" in
        nav-api) f="$NAV_TOKEN_FILE" ;;
        map-api) f="$MAP_TOKEN_FILE" ;;
        *) echo "unknown service: $1" >&2; exit 1 ;;
    esac
    if [[ ! -s "$NAV_TOKEN_FILE" ]]; then
        mkdir -p "$(dirname "$NAV_TOKEN_FILE")"
        (umask 077; openssl rand -hex 24 | tr -d '\n' > "$NAV_TOKEN_FILE")
    fi
    if [[ ! -s "$MAP_TOKEN_FILE" ]]; then  # starts as nav-api's: phones and teammates keep the token they have
        mkdir -p "$(dirname "$MAP_TOKEN_FILE")"
        (umask 077; cp "$NAV_TOKEN_FILE" "$MAP_TOKEN_FILE")
    fi
    echo "$f"
}

secret_values() {  # secret-name=value lines, from this machine
    (set -a; . "$HOME/.cloudflared/access-service-token.env"
     printf 'cf-access-client-id=%s\ncf-access-client-secret=%s\n' "$CF_ACCESS_CLIENT_ID" "$CF_ACCESS_CLIENT_SECRET")
    [[ -f typesafe.api ]] && printf 'typesafe-api-key=%s\n' "$(tr -d '\n' < typesafe.api)"
    printf 'nav-api-token=%s\n' "$(cat "$(token_file nav-api)")"
    printf 'map-api-token=%s\n' "$(cat "$(token_file map-api)")"
}

setup() {
    g services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
    g artifacts repositories describe cloud-run-source-deploy --location "$REGION" >/dev/null 2>&1 ||
        g artifacts repositories create cloud-run-source-deploy --location "$REGION" --repository-format docker
    g iam service-accounts describe "$(sa map-api)" >/dev/null 2>&1 ||
        g iam service-accounts create nav-worker --display-name "map-api Cloud Run worker"
    g iam service-accounts describe "$(sa nav-api)" >/dev/null 2>&1 ||
        g iam service-accounts create nav-api --display-name "nav-api Cloud Run (reads the bucket)"
    gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member "serviceAccount:$(sa map-api)" \
        --role roles/storage.objectAdmin >/dev/null
    gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member "serviceAccount:$(sa nav-api)" \
        --role roles/storage.objectViewer >/dev/null
    local svc pair name
    for svc in "${SERVICES[@]}"; do
        for pair in $(secrets_of "$svc"); do
            name="${pair#*=}"
            g secrets describe "$name" >/dev/null 2>&1 || g secrets create "$name" --replication-policy automatic
            g secrets add-iam-policy-binding "$name" --member "serviceAccount:$(sa "$svc")" \
                --role roles/secretmanager.secretAccessor >/dev/null 2>&1 ||
                echo "  $name: can't grant $(sa "$svc") access (needs a project owner; see the top of $0)"
        done
    done
    echo "setup done; now: $0 secrets, then $0 deploy"
}

# GitHub Actions: OIDC tokens of this repo -> the shared workload identity pool "github" -> nav-deployer, which may
# push images and deploy both services running as their accounts. Only adds: the repo to the provider's list, the
# account and its roles.
GITHUB_REPO=tungsten-united/nav-engine
WIF_POOL=github
WIF_PROVIDER=github-oidc
DEPLOYER="nav-deployer@$PROJECT.iam.gserviceaccount.com"
ci-setup() {
    local number cond svc
    number=$(g projects describe "$PROJECT" --format 'value(projectNumber)')
    cond=$(g iam workload-identity-pools providers describe "$WIF_PROVIDER" --workload-identity-pool "$WIF_POOL" \
        --location global --format 'value(attributeCondition)')
    if [[ "$cond" != *"'$GITHUB_REPO'"* ]]; then
        [[ "$cond" == "assertion.repository in ["*"]" ]] || { echo "unexpected provider condition: $cond" >&2; exit 1; }
        cond="${cond%]}, '$GITHUB_REPO']"
        g iam workload-identity-pools providers update-oidc "$WIF_PROVIDER" --workload-identity-pool "$WIF_POOL" \
            --location global --attribute-condition "$cond" >/dev/null
        echo "  provider $WIF_PROVIDER: now accepts $GITHUB_REPO"
    fi
    g iam service-accounts describe "$DEPLOYER" >/dev/null 2>&1 ||
        g iam service-accounts create nav-deployer --display-name "map-api / nav-api deploys from GitHub Actions"
    for role in roles/run.admin roles/artifactregistry.writer; do
        g projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$DEPLOYER" --role "$role" \
            --condition None >/dev/null
    done
    for svc in "${SERVICES[@]}"; do
        g iam service-accounts add-iam-policy-binding "$(sa "$svc")" --member "serviceAccount:$DEPLOYER" \
            --role roles/iam.serviceAccountUser >/dev/null
    done
    g iam service-accounts add-iam-policy-binding "$DEPLOYER" --role roles/iam.workloadIdentityUser \
        --member "principalSet://iam.googleapis.com/projects/$number/locations/global/workloadIdentityPools/$WIF_POOL/attribute.repository/$GITHUB_REPO" >/dev/null
    echo "ci-setup done: tag a commit of main (v*) to deploy"
}

put() {  # put <secret> : value on stdin
    g secrets versions add "$1" --data-file=- >/dev/null && echo "  $1: new version"
}

secrets() {
    local line
    while IFS= read -r line; do
        printf %s "${line#*=}" | put "${line%%=*}"
    done < <(secret_values)
}

token() { cat "$(token_file "${1:-map-api}")"; echo; }

image() {  # build the Dockerfile and push it; prints the image reference
    local tag
    tag="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- nav frontend pyproject.toml uv.lock Dockerfile || echo -dirty)"
    docker build -q -t "$REPO/$IMAGE:$tag" . >&2
    gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin "https://$REGION-docker.pkg.dev" >/dev/null 2>&1
    docker push -q "$REPO/$IMAGE:$tag" >&2
    echo "$REPO/$IMAGE:$tag"
}

# How each service runs. nav-api replaced the all-in-one service of the same name, and `gcloud run deploy` keeps
# whatever a flag doesn't set, so its flags also undo that one's: the read-write mount, no CPU throttling.
flags() {
    local volume="name=data,type=cloud-storage,bucket=$BUCKET,mount-options=implicit-dirs"
    case "$1" in
        map-api) echo --add-volume "$volume" --add-volume-mount volume=data,mount-path=/data \
                      --cpu 2 --memory 4Gi --no-cpu-throttling --min-instances 1 --max-instances 1 \
                      --concurrency 40 --timeout 3600 ;;
        nav-api) echo --clear-volumes --clear-volume-mounts --add-volume "$volume,readonly=true" \
                      --add-volume-mount volume=data,mount-path=/data \
                      --cpu 1 --memory 1Gi --cpu-throttling --min-instances 1 --max-instances 3 \
                      --concurrency 20 --timeout 60 ;;
    esac
}

ENV_FILES=()
cleanup() { [[ ${#ENV_FILES[@]} -eq 0 ]] || rm -f "${ENV_FILES[@]}"; }
trap cleanup EXIT

deploy_one() {  # deploy_one <service> <image>
    local svc="$1" img="$2" env_file pair secret_flags=() set=""
    env_file=$(mktemp --suffix .yaml)
    ENV_FILES+=("$env_file")
    chmod 600 "$env_file"
    cp "deploy/$svc.env.yaml" "$env_file"
    if [[ "${SECRETS_FROM:-manager}" == manager ]]; then
        for pair in $(secrets_of "$svc"); do set+="${pair%%=*}=${pair#*=}:latest,"; done
        secret_flags=(--set-secrets "${set%,}")
    else
        local values
        values=$(secret_values)
        for pair in $(secrets_of "$svc"); do
            printf '%s=%s\n' "${pair%%=*}" "$(grep "^${pair#*=}=" <<< "$values" | cut -d= -f2-)"
        done | python3 -c 'import json, sys
for line in sys.stdin.read().splitlines():
    k, v = line.split("=", 1)
    print(f"{k}: {json.dumps(v)}")' >> "$env_file"
        secret_flags=(--clear-secrets)
    fi
    # shellcheck disable=SC2046  # flags() prints separate words without spaces inside them
    g run deploy "$svc" --image "$img" --region "$REGION" --service-account "$(sa "$svc")" \
        --execution-environment gen2 $(flags "$svc") --allow-unauthenticated \
        --env-vars-file "$env_file" "${secret_flags[@]}" >&2
    url "$svc"
}

deploy() {
    local which="${1:-all}" img svc
    if [[ "$which" == all ]]; then
        set -- "${SERVICES[@]}"
    else
        sa "$which" >/dev/null
        set -- "$which"
    fi
    img=$(image)
    for svc in "$@"; do  # map-api first: the debugging frontend moves there before nav-api changes
        deploy_one "$svc" "$img"
    done
}

url() { g run services describe "${1:-map-api}" --region "$REGION" --format 'value(status.url)'; }

cmd="${1:-deploy}"
shift || true
"$cmd" "$@"
