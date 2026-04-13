#!/bin/bash
set -euo pipefail

INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // ""')

# Python 관련 명령인지 확인
if ! echo "$CMD" | grep -Eq '(^|\s|&&\s*|;\s*)(pip\s|pip3\s|python\s|python3\s|pytest|uvicorn|vllm)'; then
  exit 0
fi

# conda activate interaction 또는 conda run -n interaction 포함 시 통과
if echo "$CMD" | grep -q 'conda activate interaction'; then
  exit 0
fi
if echo "$CMD" | grep -q 'conda run -n interaction'; then
  exit 0
fi

# docker 내부 실행은 예외
if echo "$CMD" | grep -Eq '(docker\s+(exec|run|compose))'; then
  exit 0
fi

SUGGESTED=$(echo "$CMD" | sed 's/^/conda activate interaction \&\& /')

cat <<EOF
{
  "decision": "deny",
  "reason": "Python 명령은 반드시 conda 가상환경에서 실행해야 합니다.\n차단: ${CMD}\n수정: ${SUGGESTED}\nconda activate interaction을 앞에 붙여서 다시 실행하세요."
}
EOF
exit 0
