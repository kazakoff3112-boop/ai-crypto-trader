# Public portfolio history

The first public release removes private workstation paths, references to an
unrelated deployed service and a concrete shadow-state snapshot from historical
documentation, not only from the current checkout. Application code, strategies,
tests and systemd units are unchanged by this sanitization.

Rewritten documentation changes commit hashes. A pre-existing private deployment
may therefore have a different Git ancestry, even when its application code is
identical. **Do not automatically pull this public history into such a VPS or
force-reset production.** A future deployment requires a separate, reviewed
code/history reconciliation plan and the usual state/backup checks. Publishing
this repository does not update production.

The maintainer retains a private pre-publication Git bundle outside the tracked
tree. Never publish private backups, internal refs, operator scripts, environment
files or databases. Push only the audited public `main`, not all local refs.
