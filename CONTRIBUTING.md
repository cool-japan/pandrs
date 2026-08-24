# Contributing to PandRS

Thank you for your interest in contributing to PandRS! This document provides guidelines and best practices for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Quality Standards](#code-quality-standards)
- [Error Handling Guidelines](#error-handling-guidelines)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

- Be respectful and constructive
- Focus on what is best for the project
- Show empathy towards other contributors
- Accept constructive criticism gracefully

## Getting Started

### Prerequisites

- Rust **1.88** for the default feature set and `distributed`; **1.89** for
  `cloud-storage`/`all-safe`/`stable` (a higher floor pulled in via
  `object_store`'s AWS backend). The workspace `rust-version` in
  [`Cargo.toml`](Cargo.toml) is pinned at the 1.88 floor — cargo enforces
  the higher per-dependency requirement automatically once a
  1.89-requiring feature is on. **The `all-safe` bundle used throughout
  this guide needs 1.89.**
- Cargo
- Git

### Setup

```bash
git clone https://github.com/cool-japan/pandrs.git
cd pandrs

# Use the `all-safe` feature bundle for local dev — it covers everything
# except CUDA/WASM/distributed, which need external toolchains you likely
# don't have installed (`--all-features` will try to build the `cuda`
# feature too, which needs the CUDA toolkit).
cargo build --features all-safe
cargo nextest run --features all-safe
```

### Install Pre-commit Hook

```bash
chmod +x scripts/pre-commit-unwrap-check.sh
ln -s ../../scripts/pre-commit-unwrap-check.sh .git/hooks/pre-commit
```

## Development Workflow

### Branch Policy

- `master` receives only release commits (tagged `Availability of X.Y.Z`).
- All development happens on the current `0.x.y` release branch (e.g. `0.4.1`
  right now — check `git branch` / the repo's default branch pointer if
  unsure which one is active).
- Base your feature branch off the current `0.x.y` branch, not `master`.

### Workflow

1. Fork the repository
2. Create a feature branch off the current `0.x.y` branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `cargo nextest run --features all-safe` (see the note on `--all-features` under Setup above)
5. Check for clippy warnings: `cargo clippy --features all-safe -- -D warnings` (the enforced, library-scoped gate — see the policy note below for why this isn't `--all-targets`)
6. Format code: `cargo fmt`
7. Commit changes (pre-commit hook will run)
8. Push to your fork
9. Create a Pull Request

## Code Quality Standards

### No Warnings Policy

**All code must compile without warnings:**
- Run `cargo clippy --features all-safe -- -D warnings` (library-scoped — this is the enforced gate; see the policy note below)
- Fix all clippy warnings before submitting PR
- Use `#[allow(clippy::...)]` sparingly and only with justification

> **Clippy policy.** PandRS allows the high-volume, stylistic
> `clippy::style` and `clippy::complexity` groups crate-wide (in
> `src/lib.rs`) and enforces `clippy::correctness` (deny), `clippy::suspicious`,
> and `clippy::perf` (warn). The former blanket `#![allow(clippy::all)]` is
> **gone — do not reintroduce it**: it would silence those three enforced
> groups too and make "zero clippy warnings" unfalsifiable. Enforcement is
> verified on the library with `cargo clippy --features all-safe -- -D
> warnings` (clean). A few narrowly-scoped, individually-justified allows
> remain, each documented at its site:
> - `result_large_err` crate-wide — the public `Error` enum is 232 bytes on
>   64-bit (over clippy's 128-byte threshold), dominated by the `Enhanced`
>   variant's `ErrorContext` payload. Shrinking it (boxing the payload or
>   splitting the enum) is a breaking public-API change, deferred to 0.5.0.
> - Six `mut_from_ref` in `storage::arena` — the canonical bump-arena
>   pattern; a `&mut self` signature on `alloc` would defeat the arena's
>   core invariant of holding several live allocations at once.
> - One `arc_with_non_send_sync` in the JIT window cache
>   (`dataframe::jit_window`) — the cache legitimately needs
>   `Arc<Mutex<..>>`, but can't be `Send+Sync` because `JitFunction`
>   transitively holds a `!Send` JIT context; a thread-safe JIT-state
>   redesign is deferred to 0.5.0.
> - Module-scoped `approx_constant` on three test modules whose
>   fixture/expected float literals merely resemble π/√2 (e.g. arena tests
>   allocating a `3.14` payload, an asserted `std([1..=5]) == 1.4142…`) —
>   not a genuine π/e computation anywhere.
>
> New code must pass the enforced groups (`correctness`/`suspicious`/`perf`);
> use `#[allow(clippy::…)]` only for a specific lint at a specific site,
> with a written reason — never a blanket re-allow. Integration tests,
> examples, and benches are separate crates that the crate-level allows in
> `src/lib.rs` don't reach, so a broader `cargo clippy --all-targets
> --features all-safe` run will still surface stylistic and
> `result_large_err` warnings there by design — the enforced gate is the
> library-scoped command above, not an `--all-targets -D warnings` run.

### No Unwrap Policy ⚠️ CRITICAL

**Never use `.unwrap()` in production code.** This is checked by a local
pre-commit hook (`scripts/pre-commit-unwrap-check.sh`, see Setup above) and
by reviewers — there is currently no CI workflow that runs on pull requests
(see the Review Process note near the end of this document), so please run
the hook and `cargo clippy`/`cargo nextest` locally before opening a PR.

#### ❌ Bad - Don't Do This

```rust
let value = map.get(&key).unwrap();
let first = vec.first().unwrap();
let result = operation().unwrap();
```

#### ✅ Good - Do This Instead

```rust
// Use ? operator for error propagation
let value = map.get(&key)
    .ok_or_else(|| Error::KeyNotFound(key.clone()))?;

// Use .ok_or_else() with descriptive errors
let first = vec.first()
    .ok_or_else(|| Error::InsufficientData("vector is empty".into()))?;

// Propagate errors properly
let result = operation()?;
```

### Error Handling Patterns

PandRS has established 8 standard error handling patterns. **Always use these patterns** instead of `.unwrap()`:

#### 1. Float Comparisons (NaN Handling)

```rust
// Safe NaN handling
a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
```

#### 2. Collection Access

```rust
// Empty collection check
values.first()
    .ok_or_else(|| Error::InsufficientData("collection is empty".into()))?

// Index access
values.get(index)
    .ok_or_else(|| Error::IndexOutOfBounds { index, size: values.len() })?

// Last element
values.last()
    .ok_or_else(|| Error::InsufficientData("collection is empty".into()))?
```

#### 3. HashMap Lookups

```rust
// Key lookup
map.get(&key)
    .ok_or_else(|| Error::ColumnNotFound(key.clone()))?

// Mutable lookup (use if-let pattern)
if let Some(value) = map.get_mut(&key) {
    *value += 1;
} else {
    return Err(Error::ColumnNotFound(key.clone()));
}
```

#### 4. Model State Validation

```rust
// ML model state
self.coefficients
    .ok_or_else(|| Error::InvalidOperation(
        "Model not fitted. Call fit() first.".into()
    ))?

// Optional fields
self.config
    .ok_or_else(|| Error::InvalidOperation(
        "Configuration not set".into()
    ))?
```

#### 5. Lock Operations

```rust
// Use the lock_safe! macro
use crate::core::sync_helpers::lock_safe;

let data = lock_safe!(self.cache, "cache access")?;
let guard = read_lock_safe!(self.buffer, "buffer read")?;

// Or manually handle poison errors
let data = self.cache.lock()
    .map_err(|_| Error::LockPoisoned {
        context: "cache lock".into()
    })?;
```

#### 6. GPU Operations

```rust
// Memory layout validation
matrix.data.as_slice()
    .ok_or_else(|| GpuError::InvalidData(
        "Matrix not contiguous in memory".into()
    ))?

// Device allocation
device.allocate(size)
    .map_err(|e| GpuError::AllocationFailed(e.to_string()))?
```

#### 7. Iterator Access

```rust
// Iterator exhaustion
iter.next()
    .ok_or_else(|| Error::InsufficientData(
        "iterator exhausted".into()
    ))?

// Iterator minimum/maximum
values.iter()
    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
    .ok_or_else(|| Error::InsufficientData("empty iterator".into()))?
```

#### 8. Type Conversions

```rust
// Column type conversion
column.as_float64()
    .ok_or_else(|| Error::TypeMismatch(format!(
        "Expected Float64Column, got {}",
        column.dtype()
    )))?

// String parsing
s.parse::<i64>()
    .map_err(|e| Error::ParseError {
        value: s.to_string(),
        target_type: "i64".into(),
        error_msg: e.to_string(),
    })?
// Or use the equivalent helper: Error::parse_error(s, "i64", e)
```

### When to Use `.expect()`

`.expect()` is acceptable **only** for impossible failures with clear comments:

```rust
// ✅ OK - Compile-time constant is always valid
let time = NaiveTime::from_hms_opt(0, 0, 0)
    .expect("00:00:00 is always valid");

// ✅ OK - Schema guarantees this exists
let column = data.get_mut(&name)
    .expect("column exists in schema per constructor invariant");

// ✅ OK - Previous validation ensures this succeeds
let value = validated_option
    .expect("value validated in previous step");
```

**Always include a comment explaining WHY the failure is impossible.**

### Test Code Exception

Unwraps are **acceptable in test code**:

```rust
#[cfg(test)]
mod tests {
    use pandrs::{DataFrame, Series};

    #[test]
    fn test_something() {
        let mut df = DataFrame::new();
        df.add_column("a".to_string(), Series::new(vec![1, 2, 3], None).unwrap())
            .unwrap(); // ✅ OK in tests
        assert_eq!(df.row_count(), 3); // ✅ OK in tests
    }
}
```

## Testing Requirements

### Running Tests

```bash
# Most tests (excludes CUDA/WASM/distributed, which need external toolchains)
cargo nextest run --features all-safe --no-fail-fast

# Specific module
cargo nextest run --features all-safe dataframe

# With output
cargo nextest run --features all-safe --nocapture
```

### Test Coverage

- All new features must include tests
- Test both success and error paths
- Include edge cases and boundary conditions
- There is no coverage-measurement tool (tarpaulin/llvm-cov/codecov) wired
  into this repo today, so there is no enforced coverage percentage — use
  judgment: cover the success path, the error path, and the boundary/empty
  cases for anything you touch

### Test Organization

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_case() {
        // Test the happy path
    }

    #[test]
    fn test_error_case() {
        // Test error handling
        let result = function_that_fails();
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_case() {
        // Test boundary conditions
    }
}
```

## Code Style

### Formatting

- Use `cargo fmt` to format code
- Follow Rust naming conventions:
  - `snake_case` for variables, functions, modules
  - `PascalCase` for types, traits, enums
  - `SCREAMING_SNAKE_CASE` for constants

### Documentation

```rust
/// Brief description of function
///
/// More detailed description if needed.
///
/// # Arguments
///
/// * `param1` - Description of param1
/// * `param2` - Description of param2
///
/// # Returns
///
/// Description of return value
///
/// # Errors
///
/// * `Error::KeyNotFound` - When key doesn't exist
/// * `Error::InvalidInput` - When input is invalid
///
/// # Examples
///
/// ```
/// use pandrs::DataFrame;
/// let df = DataFrame::new();
/// // ...
/// ```
pub fn function_name(param1: Type1, param2: Type2) -> Result<ReturnType> {
    // Implementation
}
```

## Pull Request Process

### Before Submitting

- [ ] All tests pass: `cargo nextest run --features all-safe`
- [ ] No clippy warnings: `cargo clippy --features all-safe -- -D warnings`
- [ ] Code formatted: `cargo fmt`
- [ ] No unwraps in production code
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for notable changes)

### PR Description

Include:
- **What**: Brief description of changes
- **Why**: Motivation for changes
- **How**: Technical approach
- **Testing**: How you tested the changes
- **Breaking Changes**: Any API changes (avoid if possible)

### Example PR Template

```markdown
## Summary

Brief description of what this PR does.

## Motivation

Why are these changes needed?

## Changes

- Change 1
- Change 2
- Change 3

## Testing

- [ ] Added unit tests
- [ ] Added integration tests
- [ ] Tested with all features
- [ ] Verified no performance regression

## Checklist

- [ ] No unwraps in production code
- [ ] All tests pass
- [ ] No clippy warnings
- [ ] Code formatted
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. Code review by maintainers — **note:** there is currently no CI workflow
   that runs tests/clippy/fmt on pull requests (the only workflow in
   `.github/workflows/` builds and publishes release wheels on version
   tags), so please run the build/test/clippy/fmt commands above locally
   before requesting review
2. Address feedback
3. Approval and merge

## Common Pitfalls

### ❌ Don't

```rust
// Don't use unwrap
let value = option.unwrap();

// Don't ignore errors
let _ = operation();

// Don't use expect without explanation
let value = option.expect("failed");

// Don't return generic errors
return Err(Error::InvalidInput("bad input".into()));
```

### ✅ Do

```rust
// Use ? for error propagation
let value = option.ok_or_else(|| Error::specific_error())?;

// Handle or propagate errors
let result = operation()?;

// Use expect with clear explanation
let value = option.expect("value guaranteed by schema invariant");

// Return specific errors with context
return Err(Error::InvalidInput(format!(
    "Expected positive value, got {}",
    value
)));
```

## Getting Help

- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the API documentation

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Recognition

Contributors are recognized in:
- CHANGELOG.md (for significant contributions)
- GitHub contributors page
- Release notes

Thank you for contributing to PandRS! 🎉
