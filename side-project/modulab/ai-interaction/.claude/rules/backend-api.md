---
paths:
  - "backend/**/*.py"
---
- FastAPI async 엔드포인트 사용
- Pydantic v2 모델로 요청/응답 검증
- vLLM 호출은 OpenAI SDK의 AsyncOpenAI 클라이언트 사용
- LLM 응답 실패 시 generic 템플릿 + 기본 파라미터로 폴백
- 다의어 선택은 백엔드에서 random.choice()로 수행 (LLM에게 맡기지 않음)
