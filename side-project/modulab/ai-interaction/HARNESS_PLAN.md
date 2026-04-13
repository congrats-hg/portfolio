# 하네스 구축 계획

이 문서는 ai-interaction 프로젝트의 Claude Code 하네스 전체 구축 명세다.
이 파일 하나만 읽고 모든 하네스 파일을 생성할 수 있어야 한다.

## 프로젝트 요약

한국어 단어/표현의 의미 기반 인터랙티브 시각화.
사용자 입력(음성/텍스트) → EXAONE LLM 분석 → 텍스트 파티클 인터랙션 + 웹캠 트래킹.
기술: React+Vite+TS, PixiJS, Matter.js, Three.js(EyeEffect 전용), MediaPipe, FastAPI, vLLM.

---

## 생성할 파일 전체 구조

```
ai-interaction/
├── CLAUDE.md
├── PLAN.md                            # 이미 존재 (구현 계획서)
├── claude-progress.md
├── tasks.json
├── .claude/
│   ├── settings.json
│   ├── settings.local.json            # .gitignore 대상
│   ├── agents/
│   │   └── reviewer.md
│   ├── hooks/
│   │   ├── stop-verify.sh
│   │   ├── protect-sensitive.sh
│   │   ├── enforce-conda.sh
│   │   └── check-package-safety.sh
│   ├── rules/
│   │   ├── frontend-engine.md
│   │   ├── frontend-effects.md
│   │   ├── frontend-components.md
│   │   ├── backend-api.md
│   │   └── prompt-engineering.md
│   └── skills/
│       ├── dev/SKILL.md
│       ├── test-effect/SKILL.md
│       ├── test-prompt/SKILL.md
│       ├── add-effect/SKILL.md
│       └── docker-up/SKILL.md
```

---

## 1. CLAUDE.md

100줄 이내. 목차 역할. 훅으로 강제할 수 있는 것은 적지 않는다.

```markdown
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
```

---

## 2. claude-progress.md (초기 상태)

```markdown
# Progress

## 현재 상태
- Phase: 1 (스캐폴딩 + 정적 렌더링)
- 마지막 완료 작업: 없음 (프로젝트 시작)
- 빌드 상태: 미확인
- 테스트 상태: 미확인

## 다음 작업
- tasks.json의 첫 번째 todo 항목 착수

## 알려진 이슈
- (없음)

## 아키텍처 결정 기록
- (세션 중 결정사항을 여기에 누적 기록)
```

---

## 3. tasks.json (Phase 1 초기)

항목 제거/재정렬 금지. 상태만 "todo" → "done"으로 변경. JSON 형식은 Markdown보다 모델 유도 손상에 강하다.

```json
{
  "phase": 1,
  "tasks": [
    {
      "id": "task-001",
      "title": "프로젝트 스캐폴딩 (Vite React TS + FastAPI)",
      "status": "todo",
      "commit": null
    },
    {
      "id": "task-002",
      "title": "PixiJS v8 설정 + 한국어 텍스트 1개 렌더링",
      "status": "todo",
      "commit": null
    },
    {
      "id": "task-003",
      "title": "Matter.js 연동 + TextParticle 구현",
      "status": "todo",
      "commit": null
    },
    {
      "id": "task-004",
      "title": "TextParticlePool + 화면 경계 벽",
      "status": "todo",
      "commit": null
    }
  ]
}
```

---

## 4. .claude/settings.json

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "WebSearch",
      "Bash(npm:*)",
      "Bash(npx:*)",
      "Bash(node:*)",
      "Bash(pip:*)",
      "Bash(uv:*)",
      "Bash(uvicorn:*)",
      "Bash(pytest:*)",
      "Bash(python:*)",
      "Bash(python3:*)",
      "Bash(conda:*)",
      "Bash(docker compose:*)",
      "Bash(docker:*)",
      "Bash(git:*)",
      "Bash(curl:*)",
      "Bash(ls:*)",
      "Bash(mkdir:*)",
      "Bash(cat:*)",
      "Bash(cd:*)"
    ],
    "deny": [
      "Bash(rm -rf /)",
      "Bash(git push --force:*)",
      "Edit(.env)",
      "Edit(.env.local)"
    ]
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/enforce-conda.sh"
          }
        ]
      },
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/check-package-safety.sh"
          }
        ]
      },
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "prompt",
            "prompt": "다음 명령이 패키지를 설치하려 합니다: $TOOL_INPUT. 이 패키지들이 안전한지 웹서치로 확인하세요. 1) 패키지가 공식적이고 널리 사용되는지 2) 최근 보안 이슈가 보고되었는지 3) typosquatting 가능성이 있는지. 위험하다고 판단되면 {\"ok\": false, \"reason\": \"위험 사유\"}로 응답하세요. 안전하면 {\"ok\": true}로 응답하세요.",
            "if": "tool_input.command matches '(npm install|pip install)'"
          }
        ]
      },
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/protect-sensitive.sh"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "FILE=$(cat | jq -r '.tool_input.file_path // empty'); if [ -n \"$FILE\" ]; then case \"$FILE\" in *.ts|*.tsx) cd $CLAUDE_PROJECT_DIR/frontend && npx prettier --write \"$CLAUDE_PROJECT_DIR/$FILE\" 2>/dev/null ;; *.py) conda run -n interaction python3 -m black --quiet \"$CLAUDE_PROJECT_DIR/$FILE\" 2>/dev/null ;; esac; fi; exit 0"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/stop-verify.sh",
            "timeout": 60
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "echo '세션 시작: claude-progress.md, tasks.json, PLAN.md를 먼저 읽으세요.'"
          }
        ]
      }
    ]
  }
}
```

---

## 5. .claude/hooks/

### enforce-conda.sh

pip/python/pytest/uvicorn/vllm 명령이 conda activate interaction 없이 실행되면 차단.

```bash
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
```

### check-package-safety.sh

npm install / pip install 시 패키지명을 추출하여 정적 위험 패턴 검사. 의심 시 차단하고 사용자 확인 유도.

```bash
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
```

### stop-verify.sh

테스트/빌드/린트가 모두 통과하고 claude-progress.md가 업데이트되어야 종료 허용.

```bash
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
```

### protect-sensitive.sh

```bash
#!/bin/bash
set -euo pipefail

FILE=$(cat | jq -r '.tool_input.file_path // empty')

if [ -n "$FILE" ] && echo "$FILE" | grep -qE '(\.env$|\.env\.local$|credentials|secret|\.pem$|\.key$)'; then
  echo '{"decision": "deny", "reason": "민감 파일 수정 차단: '"$FILE"'"}'
  exit 0
fi

exit 0
```

모든 .sh 파일은 `chmod +x`로 실행 권한을 부여해야 한다.

---

## 6. .claude/agents/reviewer.md

만든 에이전트가 평가하면 미흡해도 칭찬하는 편향이 있다. 별도 서브에이전트가 회의적으로 평가한다.

```markdown
---
name: reviewer
description: 구현 결과물을 독립적으로 평가하는 QA 에이전트
allowed-tools:
  - Read
  - Glob
  - Grep
  - Bash(npm test:*)
  - Bash(npx tsc:*)
  - Bash(conda run -n interaction pytest:*)
  - Bash(git diff:*)
model: sonnet
---

당신은 ai-interaction 프로젝트의 독립 코드 리뷰어입니다.

## 절대 규칙
- 실패하면 실패다. 합리화하지 말고, 축소하지 말고, 변명하지 마라.
- "나쁘지 않다", "대체로 괜찮다" 금지. 통과/실패를 이진 판정.

## 평가 체크리스트

### 정확성
- [ ] 타입 에러 없음 (`npx tsc --noEmit`)
- [ ] 프론트 테스트 통과 (`npm test`)
- [ ] 백엔드 테스트 통과 (`conda run -n interaction pytest backend`)
- [ ] 린트 통과 (`npm run lint`)

### 아키텍처 준수
- [ ] React 컴포넌트에서 PixiJS ticker 직접 접근 안 함
- [ ] useTrackingStore를 컴포넌트에서 구독하지 않음
- [ ] 새 Effect가 있으면 BaseEffect 상속 + EffectRegistry 등록됨
- [ ] Matter.Body 풀링하지 않음

### 완성도
- [ ] tasks.json 해당 항목 "done" 업데이트됨
- [ ] claude-progress.md 업데이트됨
- [ ] 서술적 커밋 메시지로 커밋됨

## 출력 형식
```
## 평가 결과: [PASS / FAIL]
### 통과 항목
- (목록)
### 실패 항목
- (항목): (구체적 이유)
### 수정 지시
- (구체적 행동)
```

실패 항목이 1개라도 있으면 FAIL.
```

---

## 7. .claude/rules/

### frontend-engine.md
```yaml
---
paths:
  - "frontend/src/engine/**/*.ts"
---
```
- Engine.ts의 update() 루프 순서: effect.update → physics.step → pool.sync → collision.check
- TextParticle은 반드시 Matter.Body + PIXI.Sprite 쌍으로 구성
- CollisionManager는 공간 해시 그리드 사용 (O(n*m) 전수 검사 금지)
- PhysicsWorld.step()에서 Matter.Engine.update() 호출 시 delta 전달 필수
- sprite.anchor.set(0.5, 0.5)로 Matter.js body center와 동기화

### frontend-effects.md
```yaml
---
paths:
  - "frontend/src/effects/**/*.ts"
---
```
- 모든 Effect는 BaseEffect를 상속
- 필수 구현 메서드: init, update, onHandCollision, onFaceUpdate, destroy
- destroy()에서 반드시 모든 파티클을 풀에 반환
- EffectRegistry에 등록하지 않으면 사용 불가
- 파라미터 범위: speed(1-10), density(1-100), intensity(1-10)
- EyeEffect만 Three.js 사용 가능, 나머지는 PixiJS만 사용

### frontend-components.md
```yaml
---
paths:
  - "frontend/src/components/**/*.tsx"
---
```
- React 컴포넌트에서 PixiJS ticker에 직접 접근하지 않음 (Engine을 통해서만)
- useTrackingStore는 컴포넌트에서 구독 금지 (useAppStore만 구독)
- WebcamProvider는 Context로 video element 제공
- PixiCanvas는 Application mount/unmount만 담당

### backend-api.md
```yaml
---
paths:
  - "backend/**/*.py"
---
```
- FastAPI async 엔드포인트 사용
- Pydantic v2 모델로 요청/응답 검증
- vLLM 호출은 OpenAI SDK의 AsyncOpenAI 클라이언트 사용
- LLM 응답 실패 시 generic 템플릿 + 기본 파라미터로 폴백
- 다의어 선택은 백엔드에서 random.choice()로 수행 (LLM에게 맡기지 않음)

### prompt-engineering.md
```yaml
---
paths:
  - "backend/app/services/prompt_builder.py"
  - "backend/app/templates/**"
---
```
- 시스템 프롬프트는 한국어로 작성
- 추출 우선순위: 의성어 > 의태어 > 다의어 > 자연현상 > 일반 단어
- vLLM guided_json으로 JSON 스키마 강제 (shared/templates.schema.json과 동기화)
- few-shot 예시 포함 권장 (최소 3개)
- temperature: 의미 분석 0.3, 파라미터 생성 0.7

---

## 8. .claude/skills/

### dev/SKILL.md

```yaml
---
name: dev
description: 프론트엔드와 백엔드 개발 서버를 동시에 실행합니다.
allowed-tools: "Bash(npm *) Bash(conda *) Bash(uvicorn *) Bash(cd *) Bash(kill *) Bash(lsof *)"
---

# 개발 서버 실행

1. 기존에 실행 중인 dev 서버가 있는지 확인 (포트 5173, 8080)
2. 있으면 종료
3. 백엔드: `conda activate interaction && cd backend && uvicorn app.main:app --reload --port 8080 &`
4. 프론트엔드: `cd frontend && npm run dev`

$ARGUMENTS가 "backend"이면 백엔드만, "frontend"이면 프론트엔드만 실행.
```

### test-effect/SKILL.md

```yaml
---
name: test-effect
description: 특정 인터랙션 효과를 하드코딩된 파라미터로 테스트합니다.
argument-hint: "[effect-name]"
allowed-tools: "Bash(npm *) Bash(npx *) Read Edit"
---

# 효과 테스트

$0 = 테스트할 효과 이름 (rain, wind, waterfall, impact, eye, snow, generic)

1. frontend/src/App.tsx를 읽고 현재 상태 확인
2. 해당 효과의 테스트용 하드코딩 파라미터 설정:
   - rain: { speed: 3, density: 50, size: "medium", color: "#4A90D9" }
   - wind: { speed: 2, density: 30, size: "medium", color: "#7EC8E3" }
   - waterfall: { speed: 5, density: 80, size: "medium", color: "#2E86C1" }
   - impact: { speed: 8, density: 1, size: "huge", color: "#E74C3C" }
   - eye: { speed: 1, density: 1, size: "large", color: "#2C3E50" }
   - snow: { speed: 1, density: 40, size: "small", color: "#ECF0F1" }
3. 해당 효과를 즉시 트리거하는 테스트 코드를 임시로 적용
4. dev 서버가 실행 중인지 확인
5. 테스트 후 임시 코드 롤백 방법 안내
```

### test-prompt/SKILL.md

```yaml
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
```

### add-effect/SKILL.md

```yaml
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
```

### docker-up/SKILL.md

```yaml
---
name: docker-up
description: Docker Compose로 전체 서비스를 빌드하고 실행합니다.
allowed-tools: "Bash(docker *) Bash(docker compose *) Bash(curl *)"
---

# Docker Compose 실행

1. docker compose 설정 파일 확인
2. 이미지 빌드: `docker compose build`
3. 서비스 실행: `docker compose up -d`
4. 헬스체크:
   - vLLM: `curl -s http://localhost:8000/v1/models`
   - 백엔드: `curl -s http://localhost:8080/health`
   - 프론트엔드: `curl -s http://localhost:5173`
5. 로그 확인: `docker compose logs -f --tail=50`

$ARGUMENTS가 "down"이면 `docker compose down` 실행.
$ARGUMENTS가 "logs"이면 `docker compose logs -f` 실행.
```

---

## 9. 구축 순서

### Step 1: 자율 실행 기반 (최우선)
- [ ] CLAUDE.md 생성
- [ ] PLAN.md가 프로젝트 루트에 있는지 확인
- [ ] claude-progress.md 생성
- [ ] tasks.json 생성
- [ ] .claude/settings.json 생성
- [ ] .claude/hooks/ 4개 파일 생성 + 모두 `chmod +x`
- [ ] .claude/agents/reviewer.md 생성
- [ ] .gitignore에 `.claude/settings.local.json` 추가
- [ ] `conda env list | grep interaction`으로 환경 존재 확인

### Step 2: 규칙 파일
- [ ] .claude/rules/ 5개 파일 생성

### Step 3: 개발 스킬 (Phase 1~2)
- [ ] /dev, /test-effect 스킬 생성

### Step 4: LLM 스킬 (Phase 4)
- [ ] /test-prompt, /add-effect 스킬 생성

### Step 5: 배포 스킬 (Phase 8)
- [ ] /docker-up 스킬 생성

---

## 10. 자율 실행 (AFK 모드)

Ralph Wiggum 패턴. Claude Code에서:

```bash
# 플러그인 설치 (1회)
claude plugin install ralph-wiggum

# 자율 실행 (10회 반복 상한)
/ralph-loop "tasks.json의 다음 미완료 작업을 구현하라.
각 반복마다:
1. claude-progress.md를 읽고 현재 상태 파악
2. tasks.json에서 status=todo인 첫 번째 작업 선택
3. PLAN.md에서 해당 Phase의 완료 기준 확인
4. 구현 완료 후 테스트/타입체크/린트 통과 확인
5. git commit + tasks.json 업데이트 + claude-progress.md 업데이트
6. 모든 tasks가 done이면 <promise>PHASE_COMPLETE</promise> 출력" \
  --max-iterations 10 \
  --completion-promise "PHASE_COMPLETE"
```

독립 평가는 작업 완료 후 `@reviewer 최근 커밋의 변경사항을 평가해 주세요`로 호출.

---

## 11. 훅 작동 흐름 요약

```
bash 명령 실행 시도
    │
    ▼
[enforce-conda.sh]
    pip/python/pytest에 conda activate interaction 없으면 → deny
    │ (통과)
    ▼
[check-package-safety.sh]
    npm/pip install이면 패키지명 추출 → typosquat/악성 패턴 검사 → deny
    │ (통과)
    ▼
[prompt 훅 — install 명령만]
    Claude(Haiku)가 웹서치로 패키지 안전성 확인 → 위험 시 차단
    │ (통과)
    ▼
명령 실행
    │
    ▼
[PostToolUse — Edit/Write만]
    .ts/.tsx → Prettier / .py → Black (conda run -n interaction)

에이전트 응답 완료 시도
    │
    ▼
[stop-verify.sh]
    tsc + npm test + pytest + progress 업데이트 확인
    미통과 → block (계속 작업 강제)
    stop_hook_active=true이면 → 통과 (무한 루프 방지)
```

---

## 12. 하네스가 해결하는 문제

| 하네스 | 해결하는 문제 |
|--------|--------------|
| CLAUDE.md (세션 루틴) | 매 세션 방향 잡기, PLAN.md 참조 강제 |
| claude-progress.md | 세션 간 기억 소실 → 핸드오프 |
| tasks.json | 작업 추적 (JSON이 MD보다 손상 저항력 높음) |
| Stop Hook | 미완성 상태 종료 방지 |
| reviewer 에이전트 | 자기평가 편향 방지 (생성/평가 분리) |
| enforce-conda.sh | 잘못된 Python 환경 실행 방지 (결정적 차단) |
| check-package-safety.sh | typosquat/악성 패키지 정적 차단 |
| prompt 훅 (웹서치) | 미지 패키지 동적 안전성 검증 |
| protect-sensitive.sh | .env/credential 파일 수정 차단 |
| PostToolUse 포매팅 | 코드 스타일 자동 통일 |
| rules/ | 경로별 아키텍처 규칙 강제 |
| skills/ | 반복 워크플로 원커맨드화 |