#!/usr/bin/env bash
set -euo pipefail

work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT
mkdir -p "${work}/src/deploy/strato2" "${work}/dep" "${work}/bin" "${work}/mirrors"

git -C "${work}/dep" init -q -b main
git -C "${work}/dep" config user.email test@example.invalid
git -C "${work}/dep" config user.name test
echo dep >"${work}/dep/file"
git -C "${work}/dep" add file
git -C "${work}/dep" commit -qm dep
dep_sha=$(git -C "${work}/dep" rev-parse HEAD)
git clone -q --bare "${work}/dep" "${work}/mirrors/Dummy.jl.git"

git -C "${work}/src" init -q -b kb-approved
git -C "${work}/src" config user.email test@example.invalid
git -C "${work}/src" config user.name test
echo old >"${work}/src/version"
git -C "${work}/src" add version
git -C "${work}/src" commit -qm old
old_sha=$(git -C "${work}/src" rev-parse HEAD)
echo new >"${work}/src/version"
printf 'Dummy.jl %s\n' "${dep_sha}" >"${work}/src/deploy/strato2/stack.lock"
touch "${work}/src/deploy/strato2/setup_env.jl"
git -C "${work}/src" add -A
git -C "${work}/src" commit -qm new
new_sha=$(git -C "${work}/src" rev-parse HEAD)
git clone -q --bare "${work}/src" "${work}/mirror"
git clone -q "${work}/mirror" "${work}/release"
git -C "${work}/release" checkout -q --detach "${old_sha}"

cat >"${work}/bin/systemctl" <<'STUB'
#!/usr/bin/env bash
case "$1" in
  stop)
    echo "stop $(git -C "${WHMC_RELEASE}" rev-parse HEAD)" >>"${TEST_LOG}"
    exit "${STOP_RC:-0}"
    ;;
  start)
    echo "start $(git -C "${WHMC_RELEASE}" rev-parse HEAD)" >>"${TEST_LOG}"
    ;;
  is-active) echo active ;;
esac
STUB
cat >"${work}/bin/julia" <<'STUB'
#!/usr/bin/env bash
echo "julia $(git -C "${WHMC_RELEASE}" rev-parse HEAD)" >>"${TEST_LOG}"
STUB
cat >"${work}/bin/curl" <<'STUB'
#!/usr/bin/env bash
echo curl >>"${TEST_LOG}"
STUB
cat >"${work}/bin/journalctl" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB
chmod +x "${work}/bin/"*

run_case() {
    : >"${work}/log"
    set +e
    env PATH="${work}/bin:${PATH}" TEST_LOG="${work}/log" \
        WHMC_MIRROR="${work}/mirror" WHMC_RELEASE="${work}/release" \
        WHMC_STATE_ROOT="${work}/state" WHMC_PACKAGE_MIRROR_ROOT="${work}/mirrors" \
        WHMC_JULIA="${work}/bin/julia" WHMC_HEALTH_TRIES=1 "${EXTRA_ENV[@]}" \
        bash "$(dirname "$0")/update.sh" >"${work}/out" 2>&1
    rc=$?
    set -e
}

EXTRA_ENV=()
run_case
[[ ${rc} -eq 0 ]]
[[ $(git -C "${work}/release" rev-parse HEAD) == "${new_sha}" ]]
[[ $(cat "${work}/log") == $'stop '"${old_sha}"$'\njulia '"${new_sha}"$'\nstart '"${new_sha}"$'\ncurl' ]]

git -C "${work}/release" checkout -q --detach "${old_sha}"
EXTRA_ENV=(STOP_RC=1)
run_case
[[ ${rc} -ne 0 ]]
[[ $(git -C "${work}/release" rev-parse HEAD) == "${old_sha}" ]]
! grep -q '^julia ' "${work}/log"

echo "update.sh ordering tests passed"
