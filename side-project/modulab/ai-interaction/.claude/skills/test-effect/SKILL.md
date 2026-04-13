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
