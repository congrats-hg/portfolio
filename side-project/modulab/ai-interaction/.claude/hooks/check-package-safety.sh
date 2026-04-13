#!/bin/bash
set -euo pipefail

INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // ""')

# 패키지 설치 명령인지 확인
if ! echo "$CMD" | grep -Eq '(npm\s+install|npm\s+i\s|pip\s+install|pip3\s+install)'; then
  exit 0
fi

# 패키지명 추출 (플래그 제거)
PACKAGES=$(echo "$CMD" | \
  sed -E 's/^.*(npm\s+(install|i)|pip3?\s+install)\s+//' | \
  sed -E 's/--[a-zA-Z-]+(=\S+)?//g' | \
  sed -E 's/-[a-zA-Z]\s*\S*//g' | \
  tr -s ' ' | xargs)

# 패키지명이 비어있으면 (lockfile 기반 설치) 통과
if [ -z "$PACKAGES" ]; then
  exit 0
fi

SUSPICIOUS=""
for pkg in $PACKAGES; do
  # 2글자 이하 → typosquat 의심
  if [ ${#pkg} -le 2 ]; then
    SUSPICIOUS="${SUSPICIOUS}\n- '${pkg}': 이름이 너무 짧아 typosquatting 의심"
  fi

  # 악성 패키지 접두어 패턴
  if echo "$pkg" | grep -Eiq '^(node-|python-|py-)(hide|steal|hack|exfil|keylog)'; then
    SUSPICIOUS="${SUSPICIOUS}\n- '${pkg}': 알려진 악성 패키지 패턴"
  fi

  # 유명 패키지 typosquat
  KNOWN_TYPOS="expresss|reqeusts|lodassh|axois|reacr|djagno|flaask|beautifulsoup5"
  if echo "$pkg" | grep -Eiq "^(${KNOWN_TYPOS})$"; then
    SUSPICIOUS="${SUSPICIOUS}\n- '${pkg}': 유명 패키지 typosquat 의심"
  fi
done

if [ -n "$SUSPICIOUS" ]; then
  cat <<EOF
{
  "decision": "deny",
  "reason": "패키지 안전성 경고:${SUSPICIOUS}\n\n사용자에게 이 패키지가 안전한지 확인을 요청하세요."
}
EOF
  exit 0
fi

exit 0
