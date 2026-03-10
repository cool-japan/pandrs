# PandRS Testing Policy

## Unwrap Usage in Tests

### Policy
**Test code unwrap() is ACCEPTABLE and ENCOURAGED** for test setup and assertions.

### Rationale
1. **Fail-Fast Behavior**: Tests should fail immediately on unexpected errors
2. **Readability**: `unwrap()` makes test code more concise and readable
3. **Framework Safety**: Test panics are caught by the test framework
4. **Clear Failures**: When tests panic, the error message clearly indicates the problem

### Examples

#### ✅ GOOD - Using unwrap() in tests
```rust
#[test]
fn test_dataframe_operations() {
    let mut df = DataFrame::new();
    df.add_column("id".to_string(), Series::new(vec![1, 2, 3], Some("id".to_string())).unwrap()).unwrap();
    df.add_column("name".to_string(), Series::new(vec!["a", "b", "c"], Some("name".to_string())).unwrap()).unwrap();

    assert_eq!(df.row_count(), 3);
    assert_eq!(df.column_count(), 2);
}
```

#### ❌ BAD - Over-engineering error handling in tests
```rust
#[test]
fn test_dataframe_operations() -> Result<()> {
    let mut df = DataFrame::new();
    df.add_column("id".to_string(), Series::new(vec![1, 2, 3], Some("id".to_string()))?)?;
    df.add_column("name".to_string(), Series::new(vec!["a", "b", "c"], Some("name".to_string()))?)?;

    assert_eq!(df.row_count(), 3);
    assert_eq!(df.column_count(), 2);
    Ok(())
}
```

### Production Code Policy
**Production code (src/) MUST NOT use unwrap()** except for documented safe patterns:
- `partial_cmp().unwrap_or(Ordering::Equal)` - has fallback
- Static regex with `expect()` instead of `unwrap()`

All other unwraps must be replaced with proper error handling using `?` operator or error conversion.

### Verification
To check for production code unwraps:
```bash
grep -r "\.unwrap()" src/ --include="*.rs" | grep -v test | grep -v "unwrap_or"
```

Target: 0 unwraps in production code (excluding safe patterns)

### See Also
- Safe patterns documentation: `/tmp/pandrs_safe_unwrap_patterns.md`
- Error handling helpers: `src/core/sync_helpers.rs`, `src/core/error.rs`
