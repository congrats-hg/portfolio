---
name: test-prompt
description: 한국어 문장을 vLLM에 보내 분석 결과를 확인합니다.
argument-hint: "[korean-sentence]"
allowed-tools: "Bash(curl *) Bash(conda *) Bash(python3 *) Read"
---

# LLM 프롬프트 테스트

$ARGUMENTS = 테스트할 한국어 문장

1. backend/app/services/prompt_builder.py를 읽어 현재 시스템 프롬프트 확인
2. API 요청:
   ```
   conda activate interaction && curl -s http://localhost:8080/api/analyze \
     -H "Content-Type: application/json" \
     -d '{"text": "$ARGUMENTS"}' | python3 -m json.tool
   ```
3. 응답 분석: keyword 적절성, template 선택, params 범위, 다의어 meanings
4. 문제가 있으면 prompt_builder.py 개선 제안

vLLM 서버가 없으면 백엔드만으로 테스트 불가함을 안내.
