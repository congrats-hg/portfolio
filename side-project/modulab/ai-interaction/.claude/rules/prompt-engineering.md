---
paths:
  - "backend/app/services/prompt_builder.py"
  - "backend/app/templates/**"
---
- 시스템 프롬프트는 한국어로 작성
- 추출 우선순위: 의성어 > 의태어 > 다의어 > 자연현상 > 일반 단어
- vLLM guided_json으로 JSON 스키마 강제 (shared/templates.schema.json과 동기화)
- few-shot 예시 포함 권장 (최소 3개)
- temperature: 의미 분석 0.3, 파라미터 생성 0.7
