#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# tools/download_fixture.sh
#
# Downloads the test fixture Qwen3-0.6B-Q4_K_M.gguf to testdata/ with SHA256
# verification. Idempotent: exits 0 immediately if valid file is already present.
#
# IMPORTANT REPO GROUND TRUTH:
# - huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF returns HTTP 404 (broken xet pointer).
# - huggingface.co/Qwen/Qwen3-0.6B-GGUF (official) only provides Q8_0, not Q4_K_M.
# - unsloth/Qwen3-0.6B-GGUF is verified and working; this script defaults to it.
# - A custom mirror URL can be supplied via the FIXTURE_URL environment variable.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/testdata"
TARGET_FILENAME="Qwen3-0.6B-Q4_K_M.gguf"
TARGET_FILE="${TARGET_DIR}/${TARGET_FILENAME}"
CHECKSUMS_FILE="${TARGET_DIR}/CHECKSUMS.md"

EXPECTED_SHA256="ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a"
EXPECTED_SIZE=396705472
DEFAULT_URL="https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"
URL="${FIXTURE_URL:-$DEFAULT_URL}"

compute_sha256() {
    local file="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$file" | awk '{print $1}'
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c '
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    while True:
        chunk = f.read(65536)
        if not chunk:
            break
        h.update(chunk)
print(h.hexdigest())
' "$file"
    elif command -v python >/dev/null 2>&1; then
        python -c '
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    while True:
        chunk = f.read(65536)
        if not chunk:
            break
        h.update(chunk)
print(h.hexdigest())
' "$file"
    else
        echo "FAIL: No sha256 utility found (sha256sum, shasum, or python required)" >&2
        return 1
    fi
}

get_file_size() {
    local file="$1"
    if command -v stat >/dev/null 2>&1; then
        stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || wc -c < "$file" | tr -d '[:space:]'
    else
        wc -c < "$file" | tr -d '[:space:]'
    fi
}

write_checksums_md() {
    local filename="$1"
    local size="$2"
    local sha="$3"

    cat <<EOF > "$CHECKSUMS_FILE"
# Test Fixture Checksums

| File | Size (bytes) | SHA256 |
| --- | --- | --- |
| ${filename} | ${size} | ${sha} |

## Mirror & Provenance Notes

- **Primary Source:** [unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf)
- **Repository Notes:**
  - \`huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF\` returns HTTP 404 (broken xet object pointer / "Entry not found").
  - \`huggingface.co/Qwen/Qwen3-0.6B-GGUF\` (official) only serves Q8_0, not Q4_K_M.
  - \`unsloth/Qwen3-0.6B-GGUF\` is the verified and pinned mirror source for this fixture.
EOF
}

mkdir -p "$TARGET_DIR"

if [ -f "$TARGET_FILE" ]; then
    echo "Checking existing fixture: ${TARGET_FILE}..."
    ACTUAL_SHA256="$(compute_sha256 "$TARGET_FILE" | tr '[:upper:]' '[:lower:]')"
    if [ "$ACTUAL_SHA256" = "$EXPECTED_SHA256" ]; then
        echo "PASS: already present, checksum OK (${ACTUAL_SHA256})"
        write_checksums_md "$TARGET_FILENAME" "$EXPECTED_SIZE" "$EXPECTED_SHA256"
        exit 0
    else
        echo "WARNING: Existing fixture checksum mismatch (got ${ACTUAL_SHA256}, expected ${EXPECTED_SHA256}). Re-downloading..."
    fi
fi

echo "Downloading fixture from ${URL}..."
TEMP_FILE="${TARGET_FILE}.tmp.$$"

if ! curl -L --fail --progress-bar -o "$TEMP_FILE" "$URL"; then
    echo "FAIL: curl download failed from ${URL}" >&2
    rm -f "$TEMP_FILE"
    exit 1
fi

ACTUAL_SIZE="$(get_file_size "$TEMP_FILE")"
if [ "$ACTUAL_SIZE" -ne "$EXPECTED_SIZE" ]; then
    echo "FAIL: Size mismatch for downloaded file. Expected: ${EXPECTED_SIZE}, got: ${ACTUAL_SIZE}" >&2
    rm -f "$TEMP_FILE"
    exit 1
fi

ACTUAL_SHA256="$(compute_sha256 "$TEMP_FILE" | tr '[:upper:]' '[:lower:]')"
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "FAIL: Checksum mismatch. Expected: ${EXPECTED_SHA256}, got: ${ACTUAL_SHA256}" >&2
    rm -f "$TEMP_FILE"
    exit 1
fi

mv -f "$TEMP_FILE" "$TARGET_FILE"
write_checksums_md "$TARGET_FILENAME" "$ACTUAL_SIZE" "$ACTUAL_SHA256"

echo "PASS: Download verified successfully."
echo "File:     ${TARGET_FILE}"
echo "Size:     ${ACTUAL_SIZE} bytes"
echo "SHA256:   ${ACTUAL_SHA256}"
