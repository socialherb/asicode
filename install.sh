#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

echo "========================================"
echo "  asicode 설치 스크립트"
echo "========================================"

# 1. Python >= 3.10 확인
command -v python3 &>/dev/null || { echo "❌ python3 가 필요합니다."; exit 1; }
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo "  ✓ Python $PY_VER"
else
    echo "❌ Python 3.10+ 필요합니다 (현재: $PY_VER)"
    exit 1
fi

# 2. venv 생성
if [ -d "$VENV_DIR" ]; then
    echo "  ✓ 가상환경 있음: $VENV_DIR"
else
    echo "  → 가상환경 생성 중 ..."
    python3 -m venv "$VENV_DIR"
    echo "  ✓ 가상환경 생성 완료"
fi

# 3. pip install (--break-system-packages 자동 대응)
echo "  → 의존성 설치 중 ..."
"$VENV_DIR/bin/python3" -m pip install --quiet -e "$REPO_DIR" 2>/dev/null \
    || "$VENV_DIR/bin/python3" -m pip install --quiet --break-system-packages -e "$REPO_DIR" \
    || { echo "❌ pip install 실패"; exit 1; }
echo "  ✓ 의존성 설치 완료"

# 4. 심볼릭 링크 — ~/.local/bin (PATH 에 거의 항상 있음)
TARGET_DIR="$HOME/.local/bin"
mkdir -p "$TARGET_DIR"
ln -sf "$VENV_DIR/bin/asi" "$TARGET_DIR/asi"
echo "  ✓ 심볼릭 링크: $TARGET_DIR/asi"

# 5. PATH 확인
if [[ ":$PATH:" != *":$TARGET_DIR:"* ]]; then
    echo ""
    echo "  ⚠  $TARGET_DIR 이 PATH 에 없습니다."
    echo "     셸 설정 파일에 추가:"
    echo "       echo 'export PATH=\"\$PATH:$TARGET_DIR\"' >> ~/.zshrc"
fi

echo ""
echo "========================================"
echo "  ✅ 설치 완료! 터미널에서 'asi' 입력"
echo "========================================"
