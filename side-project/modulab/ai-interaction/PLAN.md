# 한국어 단어 인터랙티브 시각화 프로젝트 구현 계획

## Context
사용자가 한국어 문장/단어/표현을 말하거나 입력하면, LLM(EXAONE 3.5-7.8B)이 핵심 단어를 추출하고 의미를 분석하여, 해당 단어의 한국어 텍스트 자체가 시각적 요소가 되는 인터랙티브 효과를 생성하는 프로젝트. 웹캠을 통한 손/얼굴 트래킹으로 텍스트 파티클과 상호작용한다.

## 기술 스택
| 영역 | 선택 |
|------|------|
| 프론트엔드 | React + Vite + TypeScript |
| 2D 렌더링 | PixiJS v8 |
| 3D 렌더링 | Three.js (EyeEffect 전용, lazy load) |
| 물리 엔진 | Matter.js |
| 트래킹 | MediaPipe (Hands + Face Mesh) |
| 음성인식 | Web Speech API (ko-KR) |
| 상태관리 | Zustand |
| 백엔드 | FastAPI (Python) |
| LLM | EXAONE 3.5-7.8B via vLLM |
| 배포 | Docker Compose (로컬 데모/전시용) |

## 프로젝트 구조

```
ai-interaction/
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   ├── public/fonts/NotoSansKR-Bold.ttf
│   └── src/
│       ├── main.tsx / App.tsx
│       ├── api/analyzeApi.ts              # POST /api/analyze 클라이언트
│       ├── components/
│       │   ├── InputPanel/                # SpeechInput + TextInput
│       │   ├── Canvas/                    # PixiCanvas + ThreeOverlay
│       │   ├── Webcam/                    # WebcamProvider, HandTracker, FaceTracker
│       │   └── Debug/DebugOverlay.tsx
│       ├── engine/
│       │   ├── Engine.ts                  # 메인 오케스트레이터 (tick loop)
│       │   ├── PhysicsWorld.ts            # Matter.js 래퍼
│       │   ├── TextParticle.ts            # Matter.Body + PIXI.Sprite 쌍
│       │   ├── TextParticlePool.ts        # 오브젝트 풀 (PIXI.Sprite 재사용)
│       │   ├── CollisionManager.ts        # 공간 해시 기반 충돌 감지
│       │   └── InteractionBridge.ts       # MediaPipe → 물리 좌표 변환
│       ├── effects/
│       │   ├── EffectRegistry.ts          # 템플릿명 → Effect 클래스 매핑
│       │   ├── BaseEffect.ts             # 추상 기반 클래스
│       │   ├── RainEffect.ts             # 비 (떨어지기 + 물 튀김)
│       │   ├── WindEffect.ts             # 바람 (좌우 흔들림 + 튕김)
│       │   ├── WaterfallEffect.ts        # 폭포 (수직 스트림 + 분기)
│       │   ├── ImpactEffect.ts           # 쾅 (대형 텍스트 + 누적 눌림)
│       │   ├── EyeEffect.ts              # 눈(안구) - Three.js 사용
│       │   ├── SnowEffect.ts             # 눈(자연) - 느린 낙하 + 부서짐
│       │   └── GenericEffect.ts          # 폴백 효과
│       ├── tracking/
│       │   ├── HandTrackingService.ts
│       │   ├── FaceTrackingService.ts
│       │   ├── TrackingCoordMapper.ts     # 정규화 좌표 → 캔버스 좌표
│       │   └── types.ts
│       ├── store/
│       │   ├── useAppStore.ts             # 앱 상태 (입력, 분석 결과)
│       │   └── useTrackingStore.ts        # 트래킹 데이터 (고빈도, 명령적 읽기)
│       ├── hooks/
│       │   ├── useSpeechRecognition.ts
│       │   ├── useAnalyze.ts
│       │   └── useAnimationFrame.ts
│       ├── types/analysis.ts
│       └── utils/
│
├── backend/
│   ├── pyproject.toml / requirements.txt
│   ├── Dockerfile
│   └── app/
│       ├── main.py                        # FastAPI 앱, CORS
│       ├── config.py                      # vLLM URL, 모델명 설정
│       ├── routers/analyze.py             # POST /api/analyze
│       ├── services/
│       │   ├── llm_service.py             # vLLM OpenAI SDK 클라이언트
│       │   └── prompt_builder.py          # 시스템/유저 프롬프트 구성
│       ├── models/
│       │   ├── request.py                 # Pydantic: AnalyzeRequest
│       │   └── response.py               # Pydantic: AnalyzeResponse
│       └── templates/
│           └── template_definitions.json
│
└── shared/
    └── templates.schema.json              # FE/BE 공유 JSON Schema
```

## 핵심 데이터 흐름

```
[음성/텍스트 입력] → useAppStore → POST /api/analyze
                                        ↓
                              FastAPI → prompt_builder → vLLM (EXAONE)
                                        ↓
                              AnalyzeResponse (JSON)
                              {keyword, template, params, meanings}
                                        ↓
[프론트엔드]  Engine → EffectRegistry.lookup(template)
                     → activeEffect.init(params)
                                        ↓
              [매 프레임 60fps]
              activeEffect.update(delta)     ← 파티클 생성/관리
              PhysicsWorld.step()             ← Matter.js 물리 연산
              TextParticlePool.sync()         ← Body 위치 → Sprite 위치
              CollisionManager.check()        ← 손 좌표 vs 파티클 충돌
                     ↓                              ↑
              activeEffect.onHandCollision()   TrackingStore
              activeEffect.onFaceUpdate()      (MediaPipe → 좌표변환)
                                                    ↑
                                              [웹캠 스트림]
```

## LLM 응답 형식 (vLLM guided_json으로 강제)

```json
{
  "keyword": "비",
  "template": "rain",
  "params": {
    "speed": 3.0,
    "density": 50,
    "size": "medium",
    "color": "#4A90D9",
    "direction": "down",
    "intensity": 5.0
  },
  "meanings": null,
  "selected_meaning": null
}
```

다의어의 경우 `meanings` 배열로 가능한 의미들이 반환되고, 백엔드에서 `random.choice`로 하나를 선정하여 `selected_meaning`에 설정.

## 인터랙션 효과별 설계

| 효과 | 중력 | 생성 위치 | 생성 빈도 | 충돌 반응 | 특수 사항 |
|------|------|-----------|-----------|-----------|-----------|
| Rain | (0, speed×2) | 상단 랜덤 X | density/sec | 방사형 임펄스 + 페이드아웃 | |
| Wind | (0, 0.1) | 좌/우 가장자리 | density/2/sec | 손 이동 방향으로 튕김 | 사인파 수평력 |
| Waterfall | (0, speed×4) | 상단 중앙 범위 | density×2/sec | 수평 편향 (스트림 분기) | 손 위치 기준 좌우 분리 |
| Impact | (0, speed×10) | 상단 중앙 | 음성 이벤트 시 | 스택 누름 (Y scale 압축) | 화면 절반 크기, 반복 입력 누적 |
| Eye | 없음 (Three.js) | 화면 중앙 | 1회 | 흔들림 + 찡그림 | 눈 깜빡임/시선 미러링 |
| Snow | (0, speed×0.3) | 상단 랜덤 X | density/3/sec | 파편으로 분해 (shatter) | Rain보다 느림, 수평 드리프트 |
| Generic | (0, speed) | 중앙 | density/sec | 밀어내기 | 폴백 |

## 핵심 기술 고려사항

### 텍스처 캐싱 (성능 핵심)
동일한 한국어 키워드('비' 등)로 200개 파티클을 만들 때, `PIXI.Text`를 매번 생성하면 200개 텍스처가 생긴다. **키워드+폰트크기 조합별로 텍스처를 1회 렌더링 후 `PIXI.Sprite`로 공유**하여 O(n)→O(1) 텍스처 생성.

### 오브젝트 풀 설계
- PIXI.Sprite: 풀링 (GPU 텍스처 재활용, GC 방지)
- Matter.Body: 매번 새로 생성 (한국어 글자폭이 다르므로 크기 재설정 필요, CPU 비용 저렴)

### MediaPipe 성능 예산
- 트래킹 감지: 매 2~3프레임마다 실행 (~20fps 트래킹)
- 렌더링: 매 프레임 60fps 유지
- 손/얼굴 감지를 교대 프레임에서 실행 가능 (N: 손, N+1: 얼굴, N+2: 스킵)

### 웹캠 미러링
TrackingCoordMapper에서 x좌표 반전: `canvasX = canvasWidth - (normalizedX × canvasWidth)`

### ImpactEffect 반복 입력 처리
활성 효과와 동일 템플릿 재입력 시 `activeEffect.onRepeat(params)` 호출하여 효과 중첩 (파괴→재생성 대신).

### Three.js + PixiJS 공존
Three.js 캔버스를 PixiJS 위에 `position: absolute; pointer-events: none`으로 오버레이. EyeEffect 활성 시에만 동적 임포트로 로드.

## 구현 단계

### Phase 1: 스캐폴딩 + 정적 렌더링
- 프로젝트 구조 생성 (Vite React TS + FastAPI)
- PixiJS v8 설정, 한국어 텍스트 1개 렌더링
- Matter.js 연동, TextParticle 구현
- TextParticlePool + 화면 경계 벽 추가
- **완료 기준**: 한국어 글자가 중력에 의해 떨어져 바닥에 착지

### Phase 2: RainEffect 구현
- BaseEffect 추상 클래스 + EffectRegistry 생성
- RainEffect 구현 (타이머 기반 생성, 랜덤 X, 속도/밀도 설정)
- Engine.ts 구축 (tick 루프 오케스트레이션)
- 하드코딩된 파라미터로 테스트
- **완료 기준**: 버튼 클릭 시 '비' 글자가 화면에 비처럼 내림

### Phase 3: 웹캠 + 손 인터랙션
- WebcamProvider (getUserMedia)
- HandTrackingService (MediaPipe HandLandmarker)
- TrackingCoordMapper + CollisionManager (공간 해시)
- RainEffect.onHandCollision() 구현
- 성능 튜닝 (프레임 스킵, FPS 측정)
- **완료 기준**: '비' 글자에 손을 가져다 대면 물 튀김 효과와 함께 사라짐

### Phase 4: 백엔드 + LLM 연동
- FastAPI 서버 + CORS 설정
- llm_service.py (OpenAI SDK → vLLM)
- prompt_builder.py 프롬프트 엔지니어링
- `guided_json`을 통한 구조화된 출력 강제
- 프론트엔드 API 클라이언트 + 파이프라인 연결
- **완료 기준**: "지금 밖에 비가 오고 있어" 입력 → LLM 분석 → Rain 효과 자동 실행

### Phase 5: 음성 입력
- useSpeechRecognition 훅 (ko-KR, continuous)
- SpeechInput 컴포넌트 (토글 + 중간 결과 표시)
- 분석 파이프라인 연결
- **완료 기준**: 마이크로 "비가 와" 말하면 비 효과 시작

### Phase 6: 나머지 효과 구현
- WindEffect (사인파 수평력 + 손 튕김)
- WaterfallEffect (수직 스트림 + 손 위치 분기)
- ImpactEffect (대형 텍스트 + 누적 눌림 + 음성 반복)
- SnowEffect (느린 낙하 + shatter)
- EyeEffect (Three.js 타원 + 얼굴 트래킹 미러)
- GenericEffect (폴백)
- **완료 기준**: 모든 예시 인터랙션이 정상 동작

### Phase 7: 얼굴 트래킹 통합
- FaceTrackingService (FaceLandmarker)
- 눈 깜빡임 감지 (Eye Aspect Ratio)
- 홍채 랜드마크로 시선 방향 추출
- EyeEffect.onFaceUpdate() 연결
- ThreeOverlay 구현
- **완료 기준**: '눈(안구)' 효과가 사용자 눈 깜빡임/시선을 미러링

### Phase 8: 마무리 + Docker
- DebugOverlay (FPS, 파티클 수, 트래킹 상태)
- 효과 전환 애니메이션
- docker-compose.yml (vLLM GPU + backend + frontend)
- UI 폴리시, 로딩/에러 처리
- 성능 프로파일링
- 다양한 한국어 입력 테스트
- **완료 기준**: `docker compose up`으로 전체 데모 실행

## 검증 방법
1. **단위 테스트**: 백엔드 프롬프트 빌더, Pydantic 모델 검증
2. **통합 테스트**: 한국어 문장 → API → JSON 응답 형식 확인
3. **시각 테스트**: 각 효과별 예시 문장으로 브라우저에서 직접 확인
4. **성능 테스트**: Chrome DevTools Performance 탭으로 60fps 유지 확인
5. **인터랙션 테스트**: 웹캠 앞에서 손/얼굴 동작으로 모든 충돌 반응 검증
