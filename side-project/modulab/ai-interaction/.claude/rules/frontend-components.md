---
paths:
  - "frontend/src/components/**/*.tsx"
---
- React 컴포넌트에서 PixiJS ticker에 직접 접근하지 않음 (Engine을 통해서만)
- useTrackingStore는 컴포넌트에서 구독 금지 (useAppStore만 구독)
- WebcamProvider는 Context로 video element 제공
- PixiCanvas는 Application mount/unmount만 담당
