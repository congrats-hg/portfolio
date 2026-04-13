#!/bin/bash
set -euo pipefail

FILE=$(cat | jq -r '.tool_input.file_path // empty')

if [ -n "$FILE" ] && echo "$FILE" | grep -qE '(\.env$|\.env\.local$|credentials|secret|\.pem$|\.key$)'; then
  echo '{"decision": "deny", "reason": "민감 파일 수정 차단: '"$FILE"'"}'
  exit 0
fi

exit 0
