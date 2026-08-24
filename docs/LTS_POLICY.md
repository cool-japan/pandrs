# PandRS Long-Term Support (LTS) Policy

## Current status: no LTS program

PandRS is **pre-1.0** (currently `0.4.x`). There is no LTS release, no
v1.0.0 date commitment, and no paid/commercial support tier. This document
states the honest, current policy — not an aspirational roadmap.

## Versioning

PandRS follows [Semantic Versioning](https://semver.org/) with the pre-1.0
caveat SemVer itself defines: for `0.y.z` releases, a bump in `y` (minor) may
contain breaking changes, same as a major bump would post-1.0. `0.4.1` itself
has shipped a breaking change in a patch release before (see CHANGELOG.md) —
until 1.0, treat every upgrade as a potential breaking change and read the
changelog.

## Support window

Only the **latest published `0.x.y` release** is supported. Older releases do
not receive backported fixes. There is no fixed release cadence.

## Security

Report vulnerabilities via [GitHub Security Advisories](https://github.com/cool-japan/pandrs/security/advisories/new)
as described in [`SECURITY.md`](../SECURITY.md). There is no CVSS-tiered SLA,
no dedicated security email alias, and no formal third-party security audit
has been completed — `SECURITY_FIX_REPORT.md` in this directory is an
internal engineering log of specific dependency-vulnerability fixes, not an
audit report.

## Minimum Supported Rust Version (MSRV)

The MSRV is whatever `rust-version` says in the workspace [`Cargo.toml`](../Cargo.toml)
at any given release; it may change in any `0.x` release pre-1.0.

## What changes when 1.0 ships

A real LTS policy (support windows, deprecation process, MSRV notice period)
will be written once there is a 1.0 release to anchor it to. Until then,
please don't build production plans around this document promising more
than it does above.
