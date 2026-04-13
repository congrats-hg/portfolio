# ai-interaction

한국어 단어/표현의 의미 기반 인터랙티브 시각화.
사용자 입력(음성/텍스트) → EXAONE LLM 분석 → 텍스트 파티클 인터랙션 + 웹캠 트래킹.

## 세션 시작 루틴
1. claude-progress.md를 읽고 현재 상태 파악
2. tasks.json에서 다음 미완료 작업 확인
3. `git log --oneline -10`으로 최근 변경 확인
4. PLAN.md를 참조하여 현재 Phase의 완료 기준 확인
5. 작업 1개만 선택하여 구현 시작

## 환경
- **Python 가상환경**: 모든 pip install, pytest, python, uvicorn, vllm 실행 시 반드시 `conda activate interaction`을 먼저 실행
  - 예: `conda activate interaction && pip install ...`
  - 예: `conda activate interaction && cd backend && pytest`
  - 절대로 시스템 Python이나 다른 환경에서 실행하지 않는다
- **패키지 설치 주의**: npm install, pip install 시 반드시 패키지의 안전성을 웹서치로 확인한 후 진행. 의심스러운 경우 설치를 중단하고 사용자에게 확인을 요청한다.

## 빌드 & 테스트
- 프론트: `cd frontend && npm run dev` (localhost:5173)
- 백엔드: `conda activate interaction && cd backend && uvicorn app.main:app --reload --port 8080`
- vLLM: `conda activate interaction && vllm serve LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --port 8000`
- 전체: `docker compose up`
- 프론트 테스트: `cd frontend && npm test`
- 백엔드 테스트: `conda activate interaction && cd backend && pytest`
- 타입: `cd frontend && npx tsc --noEmit`
- 린트: `cd frontend && npm run lint`

## 아키텍처 핵심
- 렌더링 엔진은 명령적: React는 UI 셸만. PixiJS ticker 내 Engine.ts 직접 제어.
- 트래킹 데이터는 getState()로 읽기. useStore() 구독 금지.
- 텍스처 캐싱 필수: 동일 키워드 PIXI.Text → 텍스처 1회 생성 후 Sprite 공유.
- Matter.Body는 풀링 안 함. PIXI.Sprite만 풀링.
- Three.js는 EyeEffect 전용. lazy load. 다른 효과 사용 금지.

## 코드 컨벤션
- 프론트: TypeScript strict, 2-space, named export만
- 백엔드: Python 3.11+, Pydantic v2, async/await
- 새 Effect → 반드시 EffectRegistry 등록
- 한국어 주석 OK, 변수/함수명 영어

## 세션 종료 루틴
1. 작업 결과를 git commit (서술적 메시지)
2. tasks.json에서 완료한 항목 상태를 "done"으로 변경
3. claude-progress.md 업데이트: 완료 내용, 발견된 이슈, 다음 작업 힌트
4. 테스트/빌드/린트가 모두 통과하는지 확인

## 주요 참조
- **구현 계획 (필독)**: @PLAN.md — Phase별 완료 기준, 아키텍처 설계, 효과별 사양
- 스키마: @shared/templates.schema.json
- 효과 기반 클래스: @frontend/src/effects/BaseEffect.ts
- 프롬프트: @backend/app/services/prompt_builder.py
- 경로별 상세 규칙: @.claude/rules/
