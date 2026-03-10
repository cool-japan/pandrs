#!/bin/bash
# Check for unwraps in production code
# Exit code 0 = success (no unwraps), 1 = failure (unwraps found)

set -e

echo "🔍 Checking for .unwrap() in production code..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Find all Rust files in src/
UNWRAP_COUNT=0
PRODUCTION_UNWRAPS=""

for file in $(find src/ -name "*.rs" -type f); do
  # Skip if file starts with test module
  if head -20 "$file" | grep -q "^#\[cfg(test)\]"; then
    continue
  fi

  # Count unwraps that are NOT:
  # - In comments (//)
  # - unwrap_or variants
  # - In doc comment examples (highly indented)
  # - .expect() calls (acceptable with good messages)

  while IFS= read -r line; do
    LINE_NUM=$(echo "$line" | cut -d: -f1)
    LINE_CONTENT=$(echo "$line" | cut -d: -f2-)

    # Skip if in test module (check context)
    CONTEXT=$(sed -n "$((LINE_NUM-50)),$((LINE_NUM))p" "$file" 2>/dev/null || echo "")
    if echo "$CONTEXT" | grep -q "#\[cfg(test)\]"; then
      continue
    fi

    # Skip doc examples (highly indented)
    if echo "$LINE_CONTENT" | grep -qE "^[[:space:]]{8,}"; then
      continue
    fi

    # This is a production unwrap
    UNWRAP_COUNT=$((UNWRAP_COUNT + 1))
    PRODUCTION_UNWRAPS="${PRODUCTION_UNWRAPS}\n${file}:${line}"
  done < <(grep -n "\.unwrap()" "$file" 2>/dev/null | \
           grep -v "unwrap_or" | \
           grep -v "unwrap_or_else" | \
           grep -v "unwrap_or_default" | \
           grep -v "//" | \
           grep -v "expect(" || true)
done

# Report results
echo ""
if [ $UNWRAP_COUNT -eq 0 ]; then
  echo -e "${GREEN}✅ Success! No production unwraps found${NC}"
  echo ""
  echo "Production code is unwrap-free!"
  exit 0
else
  echo -e "${RED}❌ ERROR: Found $UNWRAP_COUNT unwrap(s) in production code${NC}"
  echo ""
  echo "Unwraps found in:"
  echo -e "$PRODUCTION_UNWRAPS"
  echo ""
  echo -e "${YELLOW}📚 Error Handling Guidelines:${NC}"
  echo "  1. Use ? operator for error propagation"
  echo "  2. Use .ok_or_else() with descriptive errors"
  echo "  3. Use .expect() only for impossible failures (with comment)"
  echo "  4. Test code unwraps are acceptable"
  echo ""
  echo "Common patterns:"
  echo "  ❌ map.get(key).unwrap()"
  echo "  ✅ map.get(key).ok_or_else(|| Error::KeyNotFound(key.clone()))?"
  echo ""
  echo "  ❌ values.first().unwrap()"
  echo "  ✅ values.first().ok_or_else(|| Error::InsufficientData(\"empty\".into()))?"
  echo ""
  echo "See CONTRIBUTING.md for complete error handling patterns"
  exit 1
fi
