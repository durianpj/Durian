---
name: project-pipeline
description: RAG 챗봇 프로젝트 진행 현황 및 다음 단계
metadata: 
  node_type: memory
  type: project
  originSessionId: 52ee8692-ecea-4863-a326-12882b69efed
---

인사 데이터 기반 RAG 챗봇 프로젝트 (두리안정보기술)

**구성:** FastAPI + OpenSearch + Ollama (gemma3:4b)
**임베딩 모델:** paraphrase-multilingual-MiniLM-L12-v2 (384차원)
**데이터:** 더미 인사 데이터 3개 CSV, 각 2,000건

**Why:** 보안 레벨별로 인덱스를 7개로 나눠 접근 권한을 제어하는 RAG 시스템 구축이 목표다.

**완료된 단계:**
1. ✅ 데이터 전처리 (`데이터 전처리 정제/preprocessing.ipynb`) — 파일별 독립 처리, 정규식 없이 기초 문법으로 작성
2. ✅ JSONL 변환 (`JSONL 변환/jsonl_conversion.ipynb`) — embedding_text 생성, embedding_vector는 빈 리스트로 초기화

**다음 단계:**
3. 임베딩 생성 — sentence-transformers로 embedding_vector 채우기
4. OpenSearch 인덱싱 — 7개 보안 레벨 인덱스로 분리 후 적재
5. FastAPI RAG 챗봇

**How to apply:** 다음 단계 작업 시 위 완료 단계 결과물을 입력으로 사용한다. 임베딩 생성은 `JSONL 변환/output/*.jsonl`을 읽어서 처리한다.
