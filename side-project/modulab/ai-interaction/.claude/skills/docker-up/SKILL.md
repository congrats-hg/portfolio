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
