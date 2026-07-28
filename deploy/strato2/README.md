# Persistent Strato2 WarmupHMC app

The system service runs the dashboard on Strato2 loopback port `8128`. A
desktop user service forwards the existing WarmupHMC registry port `8090` to
that remote port, so `/p/WarmupHMC/` and its Google SSO boundary do not change.

- Release checkout: `/home/n/services/WarmupHMC`, detached at the mirror's
  `kb-approved` revision.
- Persistent environment/cache: `/home/n/.local/share/WarmupHMC/`.
- Service: `warmuphmc-web.service`, running as user `n`.
- Exact private dependencies: `stack.lock`, materialized from Strato2's
  credential-free mirrors before the service is stopped.

Bootstrap after the reviewed revision reaches the mirror:

```bash
ssh rstrato2 'bash -s' < deploy/strato2/install.sh
ssh rstrato2 'systemctl start warmuphmc-web.service'
systemctl --user link "$PWD/deploy/strato2/ssh-tunnel-warmuphmc.service"
systemctl --user enable --now ssh-tunnel-warmuphmc.service
```

Routine code update (the scoped polkit rule makes this rootless):

```bash
ssh strato2 'bash /home/n/services/WarmupHMC/deploy/strato2/update.sh'
```

`update.sh` materializes every lock pin while the old process is still healthy,
then stops the unit before moving the checkout, refreshes the Julia environment,
starts the unit, and health-checks it. If startup exits, it prints the journal
instead of waiting out a blind port-poll timeout.

Verify the three boundaries:

```bash
ssh strato2 'systemctl is-active warmuphmc-web.service; curl -fsS http://127.0.0.1:8128/'
curl -fsS http://127.0.0.1:8090/
curl -fsS http://localhost:4200/p/WarmupHMC/
```
