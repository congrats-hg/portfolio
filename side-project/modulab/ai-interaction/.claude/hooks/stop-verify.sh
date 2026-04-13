#!/bin/bash
INPUT=$(cat)

# 무한 루프 방지
if [ "$(echo "$INPUT" | jq -r '.stop_hook_active')" = "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"
ERRORS=""

# 프론트엔드 타입 체크
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
  cd frontend
  if ! npx tsc --noEmit 2>/dev/null; then
    ERRORS="${ERRORS}\n- TypeScript 타입 에러 존재"
  fi
  cd "$CLAUDE_PROJECT_DIR"
fi

# 프론트엔드 테스트
if [ -d "frontend" ] && grep -q '"test"' frontend/package.json 2>/dev/null; then
  cd frontend
  if ! npm test -- --run 2>/dev/null; then
    ERRORS="${ERRORS}\n- 프론트엔드 테스트 실패"
  fi
  cd "$CLAUDE_PROJECT_DIR"
fi

# 백엔드 테스트 (conda 환경)
if [ -d "backend" ] && [ -f "backend/pyproject.toml" ]; then
  if ! conda run -n interaction pytest backend --tb=short -q 2>/dev/null; then
    ERRORS="${ERRORS}\n- 백엔드 테스트 실패"
  fi
fi

# claude-progress.md 업데이트 여부
if ! git diff --name-only HEAD 2>/dev/null | grep -q "claude-progress.md"; then
  if ! git diff --name-only 2>/dev/null | grep -q "claude-progress.md"; then
    ERRORS="${ERRORS}\n- claude-progress.md가 업데이트되지 않음"
  fi
fi

if [ -n "$ERRORS" ]; then
  echo "{\"decision\": \"block\", \"reason\": \"종료 전 해결 필요:${ERRORS}\"}"
  exit 0
fi

exit 0
