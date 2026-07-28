#!/usr/bin/env bash
set -euo pipefail

[[ ${EUID} -eq 0 ]] || { echo "install.sh must run as root on strato2" >&2; exit 1; }

service_user=n
service_group=n
service_home=/home/n
mirror=${WHMC_MIRROR:-/home/n/.local/state/kb-agents/git-mirrors/WarmupHMC.jl.git}
release=${WHMC_RELEASE:-/home/n/services/WarmupHMC}
revision=${WHMC_REVISION:-refs/remotes/origin/kb-approved}
state_root=${WHMC_STATE_ROOT:-/home/n/.local/share/WarmupHMC}
env_dir=${WHMC_ENV_DIR:-${state_root}/env}
package_base=${WHMC_PACKAGE_BASE:-${state_root}/package-sets}
julia=${WHMC_JULIA:-/home/n/.local/bin/julia}
unit=warmuphmc-web.service

[[ -d ${mirror} ]] || { echo "missing Git mirror ${mirror}" >&2; exit 1; }
[[ -x ${julia} ]] || { echo "missing Julia ${julia}" >&2; exit 1; }
[[ -d /home/n/.bridgestan/bridgestan-2.9.0 ]] || { echo "missing BridgeStan 2.9.0" >&2; exit 1; }

install -d -o "${service_user}" -g "${service_group}" "$(dirname "${release}")"
install -d -o "${service_user}" -g "${service_group}" "${state_root}" "${state_root}/cache" "${state_root}/htmxo_errors"

as_user() { runuser -u "${service_user}" -- env HOME="${service_home}" "$@"; }

if [[ ! -e ${release}/.git ]]; then
    [[ ! -e ${release} ]] || { echo "release path exists but is not Git: ${release}" >&2; exit 1; }
    as_user git clone "${mirror}" "${release}"
fi
[[ $(as_user git -C "${release}" remote get-url origin) == "${mirror}" ]] || {
    echo "unexpected release origin" >&2
    exit 1
}
[[ -z $(as_user git -C "${release}" status --porcelain --untracked-files=no) ]] || {
    echo "refusing dirty release checkout" >&2
    exit 1
}

as_user git -C "${release}" fetch --force origin '+refs/heads/*:refs/remotes/origin/*'
resolved=$(as_user git -C "${release}" rev-parse --verify "${revision}^{commit}")
current=$(as_user git -C "${release}" rev-parse HEAD 2>/dev/null || echo none)
if [[ ${resolved} != "${current}" ]] && systemctl is-active --quiet "${unit}"; then
    echo "refusing to advance a live checkout; run update.sh first" >&2
    exit 1
fi
[[ ${resolved} == "${current}" ]] || as_user git -C "${release}" checkout --detach "${resolved}"

package_root=${package_base}/${resolved}
as_user env WHMC_PREFLIGHT_ONLY=1 WHMC_RELEASE="${release}" WHMC_REVISION="${revision}" \
    WHMC_STATE_ROOT="${state_root}" WHMC_PACKAGE_ROOT="${package_root}" WHMC_JULIA="${julia}" \
    bash "${release}/deploy/strato2/update.sh"

as_user env JULIA_DEPOT_PATH=/home/n/.julia \
    BRIDGESTAN=/home/n/.bridgestan/bridgestan-2.9.0 \
    STANC_PATH=/home/n/.bridgestan/bridgestan-2.9.0/bin/stanc \
    WHMC_ENV_DIR="${env_dir}" WHMC_PACKAGE_ROOT="${package_root}" \
    "${julia}" --startup-file=no "${release}/deploy/strato2/setup_env.jl" "${release}"

systemd-analyze verify "${release}/deploy/strato2/${unit}"
install -m 0644 "${release}/deploy/strato2/${unit}" "/etc/systemd/system/${unit}"
install -m 0644 "${release}/deploy/strato2/50-warmuphmc-web.rules" \
    /etc/polkit-1/rules.d/50-warmuphmc-web.rules
systemctl daemon-reload
systemctl enable "${unit}"
echo "installed ${unit} at ${resolved}; start it with systemctl start ${unit}"
