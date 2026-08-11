#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_DIR="$({
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
})"
readonly UPSTREAM_ROOT="${UPSTREAM_ROOT:-/home/oliver/Code/upstream/airhockey-memory-distillation}"
readonly CHALLENGE_SOURCE="${UPSTREAM_ROOT}/air_hockey_challenge"
readonly DRL_SOURCE="${UPSTREAM_ROOT}/drl_air_hockey"

readonly CHALLENGE_COMMIT="34729081d4327141dcf560c0fb8274cd4882e82c"
readonly DRL_COMMIT="a41081c4c3860877d7553c7eb9e84c1b7f5f7da0"
readonly CHALLENGE_IMAGE="marvin/air-hockey-challenge:2025-34729081d432-rebuilt"
readonly DRL_IMAGE="marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt"

for command_name in docker git tar; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        printf 'Required command is missing: %s\n' "${command_name}" >&2
        exit 1
    fi
done

for source_dir in "${CHALLENGE_SOURCE}" "${DRL_SOURCE}"; do
    if [[ ! -d "${source_dir}/.git" ]]; then
        printf 'Pinned upstream checkout is missing: %s\n' "${source_dir}" >&2
        exit 1
    fi
done

if [[ "$(git -C "${CHALLENGE_SOURCE}" rev-parse HEAD)" != "${CHALLENGE_COMMIT}" ]]; then
    printf 'Challenge checkout is not at the pinned commit.\n' >&2
    exit 1
fi
if [[ "$(git -C "${DRL_SOURCE}" rev-parse HEAD)" != "${DRL_COMMIT}" ]]; then
    printf 'DRL checkout is not at the pinned commit.\n' >&2
    exit 1
fi
if [[ -n "$(git -C "${CHALLENGE_SOURCE}" status --porcelain)" ]]; then
    printf 'Challenge checkout is dirty.\n' >&2
    exit 1
fi
if [[ -n "$(git -C "${DRL_SOURCE}" status --porcelain)" ]]; then
    printf 'DRL checkout is dirty.\n' >&2
    exit 1
fi

staging_dir="$(mktemp -d)"
cleanup() {
    rm -rf "${staging_dir}"
}
trap cleanup EXIT INT TERM

mkdir -p "${staging_dir}/challenge" "${staging_dir}/drl"
git -C "${CHALLENGE_SOURCE}" archive "${CHALLENGE_COMMIT}" \
    | tar -xf - -C "${staging_dir}/challenge"
git -C "${DRL_SOURCE}" archive "${DRL_COMMIT}" \
    | tar -xf - -C "${staging_dir}/drl"

(
    cd "${staging_dir}/drl"
    git apply --unidiff-zero \
        "${PROJECT_DIR}/patches/drl_air_hockey_marvin.patch"
)

docker build \
    --file "${PROJECT_DIR}/containers/upstream/Dockerfile.challenge-marvin" \
    --build-arg "SOURCE_COMMIT=${CHALLENGE_COMMIT}" \
    --provenance=false \
    --tag "${CHALLENGE_IMAGE}" \
    "${staging_dir}/challenge"

docker build \
    --file "${PROJECT_DIR}/containers/upstream/Dockerfile.drl-marvin" \
    --build-arg "PARENT_IMAGE=${CHALLENGE_IMAGE}" \
    --build-arg "SOURCE_COMMIT=${DRL_COMMIT}" \
    --provenance=false \
    --tag "${DRL_IMAGE}" \
    "${staging_dir}/drl"

docker image inspect "${CHALLENGE_IMAGE}" "${DRL_IMAGE}" \
    --format '{{.RepoTags}} {{.Id}} {{json .Config.Labels}}'
