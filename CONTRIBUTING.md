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

- Rust 1.70.0 or later (MSRV)
- Cargo
- Git

### Setup

```bash
git clone https://github.com/cool-japan/pandrs.git
cd pandrs
cargo build --all-features
cargo test --all-features
```

### Install Pre-commit Hook

```bash
chmod +x scripts/pre-commit-unwrap-check.sh
ln -s ../../scripts/pre-commit-unwrap-check.sh .git/hooks/pre-commit
```

## Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `cargo nextest run --all-features`
5. Check for clippy warnings: `cargo clippy --all-targets --all-features -- -D warnings`
6. Format code: `cargo fmt`
7. Commit changes (pre-commit hook will run)
8. Push to your fork
9. Create a Pull Request

## Code Quality Standards

### No Warnings Policy

**All code must compile without warnings:**
- Run `cargo clippy --all-targets --all-features -- -D warnings`
- Fix all clippy warnings before submitting PR
- Use `#[allow(clippy::...)]` sparingly and only with justification

### No Unwrap Policy ⚠️ CRITICAL

**Never use `.unwrap()` in production code.** This is strictly enforced by CI.

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
        details: e.to_string(),
    })?
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
    #[test]
    fn test_something() {
        let df = DataFrame::new();
        df.add_column("a", vec![1, 2, 3]).unwrap(); // ✅ OK in tests
        assert_eq!(df.get_column("a").unwrap().len(), 3); // ✅ OK in tests
    }
}
```

## Testing Requirements

### Running Tests

```bash
# All tests
cargo nextest run --all-features --no-fail-fast

# Specific module
cargo nextest run -p pandrs dataframe

# With output
cargo nextest run --all-features --nocapture
```

### Test Coverage

- All new features must include tests
- Aim for 80%+ test coverage
- Test both success and error paths
- Include edge cases and boundary conditions

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

- [ ] All tests pass: `cargo nextest run --all-features`
- [ ] No clippy warnings: `cargo clippy -- -D warnings`
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

1. Automated checks run (CI)
2. Code review by maintainers
3. Address feedback
4. Approval and merge

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
