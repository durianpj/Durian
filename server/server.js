// ──────────────────────────────────────────────
// HR 챗봇 프론트엔드 서버 (Express + 프록시)
//
// 역할:
//   1) index.html 등 정적 파일을 브라우저에 서빙
//   2) 브라우저의 /api/* 요청을 백엔드(localhost:8000)로 중계 (프록시)
//
// 프록시를 두는 이유:
//   - 브라우저는 다른 출처(다른 포트)로 직접 요청 보내면 CORS 정책에 막힘
//   - 프론트 서버를 통해 백엔드로 전달하면 같은 출처에서 요청한 셈이 되어 차단 없음
//   - 백엔드 코드를 수정하지 않고도 CORS 문제 해결
// ──────────────────────────────────────────────

const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');

// 상위 폴더(dma/)의 .env 파일을 읽어 환경변수로 등록
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const app = express();

// 프론트 서버가 사용할 포트 (.env의 FRONT_PORT, 기본 3000)
const PORT = process.env.FRONT_PORT || 3000;

// 백엔드(FastAPI) 주소 (.env의 BACKEND_URL, 기본 localhost:8000)
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// ──────────────────────────────────────────────
// 1. /api/* 요청을 백엔드로 중계
//    예: 브라우저가 POST /api/rag-chat 보내면
//        → 서버가 POST http://localhost:8000/rag-chat 로 전달
// ──────────────────────────────────────────────
app.use(
  '/api',
  createProxyMiddleware({
    target: BACKEND_URL,
    changeOrigin: true,
    // 요청 경로에서 '/api' 부분 제거하고 백엔드로 전달
    pathRewrite: { '^/api': '' },
  })
);

// ──────────────────────────────────────────────
// 2. index.html 하나만 명시적으로 응답
//    이전에는 dma/ 전체를 express.static으로 공개해 인사 CSV·백엔드 소스까지 노출됨
//    공개할 파일이 index.html 하나뿐이므로 그 파일만 / 경로에서 직접 보냄
// ──────────────────────────────────────────────
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'index.html'));
});

// ──────────────────────────────────────────────
// 3. 서버 시작
// ──────────────────────────────────────────────
app.listen(PORT, () => {
  console.log('────────────────────────────────────────');
  console.log(`HR 챗봇 프론트 서버 실행 중`);
  console.log(`프론트 주소: http://localhost:${PORT}`);
  console.log(`백엔드 연결 대상: ${BACKEND_URL}`);
  console.log('────────────────────────────────────────');
});
