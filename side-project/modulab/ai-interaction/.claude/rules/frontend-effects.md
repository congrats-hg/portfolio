---
paths:
  - "frontend/src/effects/**/*.ts"
---
- 모든 Effect는 BaseEffect를 상속
- 필수 구현 메서드: init, update, onHandCollision, onFaceUpdate, destroy
- destroy()에서 반드시 모든 파티클을 풀에 반환
- EffectRegistry에 등록하지 않으면 사용 불가
- 파라미터 범위: speed(1-10), density(1-100), intensity(1-10)
- EyeEffect만 Three.js 사용 가능, 나머지는 PixiJS만 사용
