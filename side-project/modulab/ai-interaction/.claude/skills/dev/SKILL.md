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
