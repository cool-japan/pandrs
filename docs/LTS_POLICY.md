# PandRS Long-Term Support (LTS) Policy

## Overview

PandRS follows a predictable release and support model to ensure stability for production deployments. This policy provides clear guidance on version support timelines, compatibility guarantees, and upgrade paths.

## Release Cycle

### Regular Releases
- **Frequency**: Every 3-4 months
- **Support Duration**: 6 months
- **Purpose**: New features, improvements, and refinements
- **Release Type**: Feature releases with performance enhancements

### LTS Releases
- **Frequency**: Every 12 months
- **Support Duration**: 24 months (2 years)
- **Purpose**: Stability and long-term maintenance with minimal breaking changes
- **Release Type**: Major version releases with extended support commitment

## LTS Versions

### v1.0.0 LTS (2026-2028)
- **Release Date**: Q2 2026
- **End of Life**: Q2 2028
- **Support Type**: Full support
- **Status**: Planned
- **Key Focus**: Production-grade stability, comprehensive documentation

### Future LTS Schedule
- v2.0.0 LTS: Q2 2027 - Q2 2029
- v3.0.0 LTS: Q2 2028 - Q2 2030

## Support Phases

### Active Support (First 18 months)
- **Features**: Full support tier
- **Bug Fixes**: All bugs addressed
- **Security Patches**: All vulnerabilities patched
- **Performance Improvements**: Continuous optimization
- **Minor Feature Additions**: Backward-compatible enhancements
- **Full Compatibility Commitment**: No breaking changes
- **Documentation**: Comprehensive and regularly updated
- **Community**: Priority community support

### Maintenance Mode (Last 6 months)
- **Security Patches**: Security vulnerabilities patched immediately
- **Critical Bug Fixes**: Only critical bugs affecting production use
- **No New Features**: Feature development halted
- **Preparation**: Focus on migration guides to next LTS
- **Status**: Pre-EOL maintenance
- **Update Cadence**: As-needed for security and critical issues

### End of Life (EOL)
- **No Further Updates**: Version enters unsupported status
- **Migration**: Users encouraged to migrate to newer LTS
- **Community Support**: Community-driven assistance only
- **Archive**: Version moved to archive repositories
- **Recommendation**: Production users should migrate to latest LTS

## Versioning Policy

Following Semantic Versioning (SemVer 2.0):

```
MAJOR.MINOR.PATCH

1.2.3
│ │ │
│ │ └─ PATCH: Bug fixes (backward compatible)
│ └─── MINOR: New features (backward compatible)
└───── MAJOR: Breaking changes
```

### LTS Version Updates

**Example for v1.0.0 LTS**:
- 1.0.0 → 1.0.1: Security patch
- 1.0.1 → 1.0.2: Bug fix
- 1.0.2 → 1.1.0: Minor feature (opt-in, backward compatible)
- **No 2.0.0 during LTS period**: Breaking changes reserved for next LTS cycle

**Guarantee**: No breaking changes during LTS lifecycle (MAJOR version remains constant).

## Compatibility Guarantees

### API Stability
- **Public API**: Remains stable across entire LTS lifecycle
- **Breaking Changes**: None allowed in MAJOR version
- **Deprecations**: Announced 6 months in advance
- **Migration Guides**: Provided for all deprecations
- **Backward Compatibility**: Maintained throughout support period
- **Binary Compatibility**: Preserved for dependent libraries

### Data Format Compatibility
- **Parquet Files**: Readable across all LTS versions and minor upgrades
- **CSV Format**: Unchanged from release to release
- **Database Schema**: Migrations provided with clear documentation
- **Data Serialization**: Consistent across versions
- **Import/Export**: Guaranteed compatibility for data interchange

### Dependency Policy
- **Minimum Supported Rust Version (MSRV)**: Defined per release
- **For v1.0.0 LTS**: Rust 1.75+
- **MSRV Updates**: Only in MINOR releases with 6-month notice
- **Dependency Stability**: Core dependencies frozen after release
- **Patch Updates**: Security-critical dependencies may be updated

## Migration Path

### Between LTS Versions

```
v1.0.0 LTS (2026-2028)
    ↓ 6-month overlap
v2.0.0 LTS (2027-2029)
    ↓ 6-month overlap
v3.0.0 LTS (2028-2030)
```

**Overlap Period**: 6 months when two LTS versions are both fully supported, enabling gradual migration.

### Migration Support
- **Detailed Migration Guides**: Step-by-step upgrade instructions
- **Deprecation Warnings**: Clear messages during transition
- **Compatibility Checkers**: Tools to identify breaking changes in your code
- **Professional Services**: Consulting available for complex migrations
- **Community Resources**: Migration discussions on GitHub
- **Testing Support**: Guidance on validating upgrades

## Security Policy

### Security Updates

**Critical** (CVSS 9.0-10.0)
- **Response Time**: Patch within 24 hours
- **Release**: Emergency release if needed
- **Notification**: Immediate to all subscribers

**High** (CVSS 7.0-8.9)
- **Response Time**: Patch within 1 week
- **Release**: Coordinated with regular schedule or emergency
- **Notification**: Priority notification to subscribers

**Medium** (CVSS 4.0-6.9)
- **Response Time**: Patch within 1 month
- **Release**: Next scheduled patch release
- **Notification**: Standard security advisory

**Low** (CVSS 0.1-3.9)
- **Response Time**: Next scheduled release
- **Release**: Regular release cycle
- **Notification**: Release notes documentation

### Vulnerability Disclosure
- **Security Contact**: security@pandrs.org
- **Response Guarantee**: Within 48 hours
- **Public Disclosure**: After patch is available and deployed
- **Coordinated Disclosure**: Responsible disclosure policy
- **Credit**: Researchers credited in advisories (if desired)

### Security Advisories
- **Published On**: GitHub Security Advisories
- **Distribution**: RSS feed available
- **Email Notifications**: For Professional+ subscription tiers
- **CVE Tracking**: All critical/high vulnerabilities tracked
- **Archive**: Historical advisories maintained

## Deprecation Policy

### Process

**Phase 1: Announcement**
- Duration: 6 months before removal
- Action: Public announcement in release notes
- Communication: Email to subscribers

**Phase 2: Marking**
- Implementation: `#[deprecated]` attribute with alternatives
- Compiler Warnings: Clear deprecation messages
- Documentation: Updated in API reference

**Phase 3: Documentation**
- Migration Guide: Detailed upgrade instructions
- Examples: Before/after code samples
- Alternatives: Clear recommendations for replacements

**Phase 4: Removal**
- Timeline: Next MAJOR version only
- Break: Breaking change formally documented
- Support: Community help available during transition

### Example Timeline
```
v1.2.0: Announce deprecation of feature X
  └─ Release notes: "Feature X deprecated, will be removed in v2.0.0"

v1.3.0 - v1.9.0: Deprecation warnings
  └─ Compiler: Generates `deprecated` warnings
  └─ Docs: Migration guide available

v2.0.0: Feature X removed
  └─ Breaking change: Feature no longer available
  └─ Support: Migration resources archived for reference
```

## Minimum Supported Rust Version (MSRV)

### Policy
- **MSRV Definition**: Specified per release in documentation
- **Update Frequency**: Updated only in MINOR releases
- **Notice Period**: 6-month advance warning before MSRV increase
- **Stability**: MSRV fixed for PATCH releases
- **Documentation**: Always documented in release notes and README

### Current MSRV
- **v0.3.0/v1.0.0**: Rust 1.75+
- **Tested On**:
  - stable (latest)
  - beta (latest)
  - nightly (latest)
- **Platform Support**: Linux, macOS, Windows

### MSRV Bump Timeline
```
v1.0.0 (Rust 1.75+): Released Q2 2026
v1.4.0 (Rust 1.78+): Released Q2 2027 - 6 month notice given
v2.0.0 (Rust 1.80+): Released Q2 2027
```

## Testing and CI

### LTS Quality Standards
- **Test Coverage**: 95%+ code coverage required
- **Known Issues**: Zero known critical bugs at release
- **Performance**: Benchmarks validated against baseline
- **Security**: Formal audit completed pre-release
- **Platform Support**: Tested on Linux, macOS, Windows
- **Regression Testing**: Continuous nightly test runs

### Continuous Integration
- **On Every Commit**: Full test suite execution
- **Nightly Builds**: Extended regression testing
- **Performance Benchmarking**: Automated performance tracking
- **Security Scanning**: Dependency vulnerability checks
- **Cross-Platform**: Multi-platform build validation
- **Documentation**: Build and validation checks

### Quality Gates for LTS Release
- All tests pass on all supported platforms
- No compiler warnings in release mode
- Benchmark performance meets or exceeds targets
- Security audit completed with no critical issues
- Documentation reviewed and complete
- Changelog comprehensive and accurate

## Commercial Support

### Included with LTS
- **Community Support**: GitHub Issues and Discussions
- **Public Documentation**: Complete API and user guides
- **Security Advisories**: Published security updates
- **Release Notes**: Detailed changelog for each update
- **Best Practices**: Documentation on production use

### Professional Tiers
- **Priority Bug Fixes**: Critical issues prioritized
- **Extended Support Options**: Support beyond standard LTS
- **Custom SLAs**: Tailored service level agreements
- **Technical Consulting**: Expert guidance available
- **Custom Builds**: Specialized features on request
- **See**: ENTERPRISE_SUPPORT.md for details

## FAQ

**Q: Can I stay on LTS forever?**

A: Each LTS is supported for 24 months. After EOL, you should migrate to the latest LTS to ensure continued security and support. However, if you have critical business needs, professional extended support may be available.

**Q: What if I find a bug in LTS?**

A: Report it through GitHub Issues. We'll fix critical bugs throughout the LTS lifecycle. For security vulnerabilities, please use the private security disclosure process.

**Q: Can I use non-LTS in production?**

A: Yes, regular releases (every 3-4 months) are production-ready. However, LTS versions are recommended for long-term stability and reduced maintenance burden.

**Q: How do I know which version is LTS?**

A: LTS versions are clearly marked in release notes and documentation (e.g., v1.0.0 LTS). Regular releases use standard versioning without the LTS designation.

**Q: What's the difference between LTS and regular releases?**

A: LTS versions receive 24 months of support vs 6 months for regular releases, have stricter compatibility guarantees, and focus on stability. Regular releases include new features and improvements more frequently.

**Q: How do I migrate from one LTS to the next?**

A: Detailed migration guides are published with each new LTS release. Start by reading the migration guide, then test thoroughly in a staging environment before production deployment.

**Q: Are there costs associated with LTS?**

A: Basic LTS support is free. Professional support options are available for enterprises requiring priority bug fixes and extended support.

**Q: What about security updates after EOL?**

A: After EOL, no updates are provided. Migrating to the current LTS is essential to maintain security compliance.

**Q: Can I request features during LTS?**

A: LTS focuses on stability. Feature requests are accepted for future LTS cycles. Feature-rich regular releases are available for new functionality.

**Q: How do I report a security vulnerability?**

A: Please email security@pandrs.org with details. Do not publicly disclose until we've had a chance to patch and release an update.

## Support Timeline Overview

```
        Active Support          Maintenance         EOL
           (18 months)          (6 months)
             ════════════════════  ══════════   ─────
v1.0.0 LTS: 2026-Q2 ──────────── 2027-Q4 ──── 2028-Q2
            │                     │            │
         Full Support        Security Only   Archived
         Bug Fixes           Critical Fixes
         Performance         No Features
         Improvements

v2.0.0 LTS: 2027-Q2 ──────────── 2028-Q4 ──── 2029-Q2
            └─────────── 6-month Overlap ──────────┘
```

## Related Documents

- [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute to PandRS
- [CHANGELOG.md](../CHANGELOG.md) - Detailed version history
- [PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md) - Production deployment guide
- [ENTERPRISE_SUPPORT.md](ENTERPRISE_SUPPORT.md) - Commercial support options (when available)

---

**Last Updated**: February 2026

**Version**: 1.0

**Maintainer**: PandRS Core Team

For questions about this policy, please open an issue on GitHub or contact us at security@pandrs.org
