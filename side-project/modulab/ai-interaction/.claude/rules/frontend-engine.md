---
paths:
  - "frontend/src/engine/**/*.ts"
---
- Engine.ts의 update() 루프 순서: effect.update → physics.step → pool.sync → collision.check
- TextParticle은 반드시 Matter.Body + PIXI.Sprite 쌍으로 구성
- CollisionManager는 공간 해시 그리드 사용 (O(n*m) 전수 검사 금지)
- PhysicsWorld.step()에서 Matter.Engine.update() 호출 시 delta 전달 필수
- sprite.anchor.set(0.5, 0.5)로 Matter.js body center와 동기화
