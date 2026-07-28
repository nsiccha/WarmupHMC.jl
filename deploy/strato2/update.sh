#!/usr/bin/env bash
set -euo pipefail

mirror=${WHMC_MIRROR:-/home/n/.local/state/kb-agents/git-mirrors/WarmupHMC.jl.git}
release=${WHMC_RELEASE:-/home/n/services/WarmupHMC}
revision=${WHMC_REVISION:-refs/remotes/origin/kb-approved}
state_root=${WHMC_STATE_ROOT:-/home/n/.local/share/WarmupHMC}
env_dir=${WHMC_ENV_DIR:-${state_root}/env}
package_base=${WHMC_PACKAGE_BASE:-${state_root}/package-sets}
package_root=${WHMC_PACKAGE_ROOT:-}
package_mirror_root=${WHMC_PACKAGE_MIRROR_ROOT:-/home/n/.local/state/kb-agents/git-mirrors}
julia=${WHMC_JULIA:-/home/n/.local/bin/julia}
unit=warmuphmc-web.service
port=${WHMC_PORT:-8128}

[[ -d ${release}/.git ]] || { echo "no service checkout at ${release}; run install.sh first" >&2; exit 1; }
[[ -x ${julia} ]] || { echo "missing Julia executable: ${julia}" >&2; exit 1; }
[[ -z $(git -C "${release}" status --porcelain --untracked-files=no) ]] || {
    echo "refusing dirty service checkout: ${release}" >&2
    exit 1
}
[[ $(git -C "${release}" remote get-url origin) == "${mirror}" ]] || {
    echo "refusing service checkout with unexpected origin" >&2
    exit 1
}

git -C "${release}" fetch --force origin '+refs/heads/*:refs/remotes/origin/*'
resolved=$(git -C "${release}" rev-parse --verify "${revision}^{commit}")
current=$(git -C "${release}" rev-parse HEAD)
[[ -n ${package_root} ]] || package_root=${package_base}/${resolved}

sync_stack() {
    local lock repository want source checkout origin dirty have fresh synced=0
    lock=$(git -C "${release}" show "${resolved}:deploy/strato2/stack.lock")
    mkdir -p "${package_root}"
    while read -r repository want _; do
        [[ -n ${repository:-} && ${repository:0:1} != '#' ]] || continue
        [[ ${want:-} =~ ^[0-9a-f]{40}$ ]] || { echo "bad stack lock row: ${repository} ${want:-}" >&2; return 1; }
        source=${package_mirror_root}/${repository}.git
        checkout=${package_root}/${repository}
        [[ -d ${source} ]] && git --git-dir="${source}" cat-file -e "${want}^{commit}" 2>/dev/null || {
            echo "missing ${repository} pin ${want} in ${source}" >&2
            return 1
        }
        fresh=0
        if [[ ! -d ${checkout}/.git ]]; then
            [[ ! -e ${checkout} ]] || { echo "refusing non-Git path ${checkout}" >&2; return 1; }
            git clone --quiet --no-checkout "${source}" "${checkout}"
            fresh=1
        fi
        origin=$(git -C "${checkout}" remote get-url origin)
        [[ ${origin} == "${source}" ]] || { echo "unexpected origin for ${checkout}: ${origin}" >&2; return 1; }
        if (( fresh == 0 )); then
            dirty=$(git -C "${checkout}" status --porcelain --untracked-files=no)
            [[ -z ${dirty} ]] || { echo "refusing dirty package checkout ${checkout}" >&2; return 1; }
        fi
        git -C "${checkout}" fetch --quiet --force origin "${want}"
        git -C "${checkout}" checkout --quiet --detach "${want}"
        have=$(git -C "${checkout}" rev-parse HEAD)
        [[ ${have} == "${want}" ]] || { echo "materialized ${have}, expected ${want}" >&2; return 1; }
        synced=$((synced + 1))
    done <<<"${lock}"
    (( synced > 0 )) || { echo "stack.lock selected no packages" >&2; return 1; }
    echo "stack: materialized ${synced} exact dependencies for ${resolved}"
}

sync_stack
if [[ ${WHMC_PREFLIGHT_ONLY:-0} == 1 ]]; then
    echo "WHMC_PREFLIGHT_ONLY=1; service untouched"
    exit 0
fi

if ! systemctl stop "${unit}"; then
    echo "could not stop ${unit}; install deploy/strato2/50-warmuphmc-web.rules" >&2
    exit 1
fi

started=0
recover_if_down() {
    status=$?
    if (( status != 0 && started == 0 )); then
        echo "deploy failed while ${unit} was stopped; attempting recovery" >&2
        systemctl start "${unit}" || echo "recovery failed; ${unit} is DOWN" >&2
    fi
}
trap recover_if_down EXIT

[[ ${resolved} == "${current}" ]] || git -C "${release}" checkout --detach "${resolved}"
JULIA_DEPOT_PATH=/home/n/.julia \
BRIDGESTAN=/home/n/.bridgestan/bridgestan-2.9.0 \
STANC_PATH=/home/n/.bridgestan/bridgestan-2.9.0/bin/stanc \
WHMC_ENV_DIR="${env_dir}" \
WHMC_PACKAGE_ROOT="${package_root}" \
    "${julia}" --startup-file=no "${release}/deploy/strato2/setup_env.jl" "${release}"

systemctl start "${unit}"
started=1

for _ in $(seq 1 "${WHMC_HEALTH_TRIES:-90}"); do
    state=$(systemctl is-active "${unit}" 2>/dev/null || true)
    if [[ ${state} != active && ${state} != activating ]]; then
        echo "${unit} entered ${state}; recent log follows" >&2
        journalctl -u "${unit}" -n 80 --no-pager >&2 || true
        exit 1
    fi
    if curl -fsS --max-time 10 "http://127.0.0.1:${port}/" >/dev/null 2>&1; then
        echo "healthy at ${resolved}: http://127.0.0.1:${port}/ served 200"
        exit 0
    fi
    sleep 2
done

echo "${unit} stayed active but never served 127.0.0.1:${port}; recent log follows" >&2
journalctl -u "${unit}" -n 80 --no-pager >&2 || true
exit 1
