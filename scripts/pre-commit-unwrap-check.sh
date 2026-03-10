#!/bin/bash
# Pre-commit hook to check for unwraps in staged files
# Install: ln -s ../../scripts/pre-commit-unwrap-check.sh .git/hooks/pre-commit

echo "🔍 Pre-commit: Checking for unwraps in staged files..."

# Get staged Rust files in src/
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep "^src/.*\.rs$" || true)

if [ -z "$STAGED_FILES" ]; then
  echo "✅ No Rust source files staged"
  exit 0
fi

# Check each staged file
UNWRAPS_FOUND=0
UNWRAP_DETAILS=""

for file in $STAGED_FILES; do
  # Get the staged content (not working directory version)
  CONTENT=$(git show ":$file")

  # Check for unwraps (excluding safe patterns)
  UNWRAPS=$(echo "$CONTENT" | grep -n "\.unwrap()" | \
            grep -v "unwrap_or" | \
            grep -v "unwrap_or_else" | \
            grep -v "unwrap_or_default" | \
            grep -v "//" | \
            grep -v "expect(" || true)

  if [ ! -z "$UNWRAPS" ]; then
    # Check if in test module
    if ! echo "$CONTENT" | grep -q "#\[cfg(test)\]"; then
      UNWRAPS_FOUND=1
      UNWRAP_DETAILS="${UNWRAP_DETAILS}\n${file}:\n${UNWRAPS}\n"
    fi
  fi
done

if [ $UNWRAPS_FOUND -eq 1 ]; then
  echo ""
  echo "❌ ERROR: Found .unwrap() in staged production code!"
  echo -e "$UNWRAP_DETAILS"
  echo ""
  echo "Please replace unwraps with proper error handling:"
  echo "  - Use ? operator for error propagation"
  echo "  - Use .ok_or_else() with descriptive errors"
  echo "  - Use .expect() only for impossible failures (with comment)"
  echo ""
  echo "To bypass this check (not recommended): git commit --no-verify"
  exit 1
else
  echo "✅ No unwraps found in staged files"
  exit 0
fi
