#!/usr/bin/env bash
#
# run_local.sh --- boots the full retrieval-core stack locally with one command
#
# Contains:
#   ensure_env_file: copies .env.example to .env when .env is missing
#   wait_for_service: polls a health endpoint until ready, fail on timeout
#   main: builds and boots api + dashboard + qdrant + redis, prints endpoints
#
# Usage:
#   scripts/run_local.sh [--no-build] [--logs]
#
# Ports once up: api :8000, dashboard :5173, qdrant :6333, redis :6379.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker/docker-compose.yml"
API_HEALTH_URL="http://localhost:8000/healthz"
DASHBOARD_URL="http://localhost:5173"
STARTUP_TIMEOUT_SECONDS=120

ensure_env_file() {
    # Copies .env.example to .env when .env is missing.
    if [[ ! -f "${REPO_ROOT}/.env" ]]; then
        cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
        echo "created .env from .env.example"
    fi
}

wait_for_service() {
    # Polls a health endpoint until ready, fail on timeout.
    #
    # Args:
    #   name: human-readable service name for log lines
    #   url: health endpoint to poll
    local name="$1"
    local url="$2"
    local elapsed=0
    until curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [[ ${elapsed} -ge ${STARTUP_TIMEOUT_SECONDS} ]]; then
            echo "ERROR: ${name} did not become healthy within ${STARTUP_TIMEOUT_SECONDS}s" >&2
            echo "inspect with: docker compose -f ${COMPOSE_FILE} logs ${name}" >&2
            exit 1
        fi
    done
    echo "${name} is healthy"
}

main() {
    # Builds and boots the stack, then prints endpoints and a sample query.
    local build_flag="--build"
    local tail_logs=false
    for arg in "$@"; do
        case "${arg}" in
            --no-build) build_flag="" ;;
            --logs) tail_logs=true ;;
            *) echo "unknown flag: ${arg}" >&2; exit 2 ;;
        esac
    done

    ensure_env_file
    echo "starting retrieval-core stack..."
    # shellcheck disable=SC2086
    docker compose -f "${COMPOSE_FILE}" up -d ${build_flag}

    wait_for_service "api" "${API_HEALTH_URL}"
    wait_for_service "dashboard" "${DASHBOARD_URL}"

    cat <<EOF

retrieval-core is up:
  api        http://localhost:8000  (docs at /docs)
  dashboard  http://localhost:5173
  qdrant     http://localhost:6333
  redis      redis://localhost:6379

try a query:
  curl -s -X POST http://localhost:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"query": "Section 3.1 indemnification", "top_k": 3}' | python3 -m json.tool

ingest the sample corpus:
  docker compose -f ${COMPOSE_FILE} exec api python -m src.ingest.loader --corpus data/docs/

stop everything:
  docker compose -f ${COMPOSE_FILE} down
EOF

    if [[ "${tail_logs}" == true ]]; then
        docker compose -f "${COMPOSE_FILE}" logs -f api
    fi
}

main "$@"
