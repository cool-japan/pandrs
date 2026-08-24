# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities using **[GitHub Security Advisories](https://github.com/cool-japan/pandrs/security/advisories/new)**
for this repository. This lets us discuss and prepare a fix privately before
any public disclosure.

If you cannot use GitHub Security Advisories, open a regular
[GitHub Issue](https://github.com/cool-japan/pandrs/issues) with minimal
detail and ask for a private channel — do not post exploit details in a
public issue.

There is no dedicated security email address, published SLA, or paid support
tier for this project. PandRS is maintained by a small team (COOLJAPAN OU /
Team Kitasan) on a best-effort basis.

## Supported Versions

PandRS is **pre-1.0** (currently `0.4.x`). Per [Semantic Versioning](https://semver.org/)
for `0.y.z` releases, any `0.x` bump may include breaking changes, and only
the **latest published `0.x.y` release** receives fixes. There is no
long-term-support (LTS) branch and no guarantee of backporting fixes to older
`0.x` releases — see [docs/LTS_POLICY.md](docs/LTS_POLICY.md).

| Version | Supported |
|---------|-----------|
| Latest `0.4.x` | Yes |
| Older `0.x` releases | No (upgrade to latest) |

## What to Expect

- We will acknowledge reports on a best-effort basis; there is no contractual
  response-time guarantee (no CVSS-tiered SLA).
- Fixes ship as a normal patch/minor release once available; there is no
  guaranteed release cadence for security fixes.
- Confirmed vulnerabilities are published as GitHub Security Advisories
  against this repository, with credit to the reporter unless they ask to
  stay anonymous.
- `cargo-deny` (see `deny.toml`) is run against dependencies as part of
  development to catch known-vulnerable crates (RUSTSEC advisories); this is
  a development-time check, not a continuously running scanning service.

## Minimum Supported Rust Version

The current MSRV is tracked in [`Cargo.toml`](Cargo.toml) (`rust-version`).
There is no formal MSRV-bump notice period pre-1.0; MSRV may move with any
release.

## Scope

This policy covers the `pandrs` Rust crate and its `py_bindings` Python
extension in this repository. It does not cover third-party forks or
redistributions.
