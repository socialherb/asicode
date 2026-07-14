#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

echo "========================================"
echo "  asicode installer"
echo "========================================"

# 1. Check Python >= 3.10
command -v python3 &>/dev/null || { echo "❌ python3 is required."; exit 1; }
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo "  ✓ Python $PY_VER"
else
    echo "❌ Python 3.10+ required (found: $PY_VER)"
    exit 1
fi

# 2. Create venv
if [ -d "$VENV_DIR" ]; then
    echo "  ✓ venv already exists: $VENV_DIR"
else
    echo "  → creating venv ..."
    python3 -m venv "$VENV_DIR"
    echo "  ✓ venv created"
fi

# 3. pip install (auto-fallback to --break-system-packages)
echo "  → installing dependencies ..."
"$VENV_DIR/bin/python3" -m pip install --quiet -e "$REPO_DIR" 2>/dev/null \
    || "$VENV_DIR/bin/python3" -m pip install --quiet --break-system-packages -e "$REPO_DIR" \
    || { echo "❌ pip install failed"; exit 1; }
echo "  ✓ dependencies installed"

# 4. Symlink — ~/.local/bin (almost always on PATH)
TARGET_DIR="$HOME/.local/bin"
mkdir -p "$TARGET_DIR"
ln -sf "$VENV_DIR/bin/asi" "$TARGET_DIR/asi"
ln -sf "$VENV_DIR/bin/asicode" "$TARGET_DIR/asicode"
echo "  ✓ symlinked: $TARGET_DIR/asi, $TARGET_DIR/asicode"

# 5. Check PATH
if [[ ":$PATH:" != *":$TARGET_DIR:"* ]]; then
    echo ""
    echo "  ⚠  $TARGET_DIR is not on your PATH."
    echo "     Add it to your shell config:"
    echo "       echo 'export PATH=\"\$PATH:$TARGET_DIR\"' >> ~/.zshrc"
fi

echo ""
echo "========================================"
echo "  ✅ Install complete! Run 'asi' in your terminal"
echo "========================================"
