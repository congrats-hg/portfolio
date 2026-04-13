---
name: add-effect
description: 새로운 인터랙션 효과를 프로젝트에 추가합니다.
argument-hint: "[name] [description]"
allowed-tools: "Read Edit Write Bash(npm *)"
---

# 새 인터랙션 효과 추가

$0 = 효과 이름 (영어, camelCase)
$1 = 효과 설명

## 필수 작업 체크리스트

1. **Effect 클래스 생성**: frontend/src/effects/${0}Effect.ts
   - BaseEffect를 읽고 인터페이스 확인
   - 기존 효과 중 가장 유사한 것을 참고
   - 필수 메서드: init, update, onHandCollision, onFaceUpdate, destroy

2. **EffectRegistry 등록**: frontend/src/effects/EffectRegistry.ts

3. **타입 확장**: frontend/src/types/analysis.ts — TemplateName union에 추가

4. **JSON Schema 업데이트**: shared/templates.schema.json — template enum에 추가

5. **Pydantic 모델 업데이트**: backend/app/models/response.py — template Literal에 추가

6. **프롬프트 업데이트**: backend/app/services/prompt_builder.py — 사용 가능 템플릿 목록에 추가 + 한국어 예시

7. **테스트**: /test-effect $0 로 동작 확인
