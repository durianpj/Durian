# 두리안정보기술 인사 데이터 RAG 챗봇 — 데이터 파이프라인

인사 데이터를 전처리하고 OpenSearch에 적재하는 파이프라인입니다.

---

## 파이프라인 구조

```
Preprocessing/  →  JSONL/  →  Chunking/  →  embedding_indexer/
  원본 CSV 정제     JSONL 변환    청킹 분할      임베딩 생성 + OpenSearch 적재
```

---

## 실행 순서

### 사전 준비

OpenSearch가 실행 중이어야 합니다.  
각 폴더의 `.env` 파일에서 경로 및 설정값을 확인하세요.

---

### 1단계 — 전처리 (`Preprocessing/`)

원본 CSV를 정제하여 `Preprocessing/output/`에 저장합니다.

```
Preprocessing/dataset/*.csv  →  Preprocessing/output/*_정제.csv
```

**실행**
```
Preprocessing/preprocessing.ipynb
```

**환경 설정** (`Preprocessing/.env`)
```
INPUT_DIR=dataset
OUTPUT_DIR=output
```

---

### 2단계 — JSONL 변환 (`JSONL/`)

정제된 CSV를 JSONL 형식으로 변환합니다.

```
Preprocessing/output/*_정제.csv  →  JSONL/output/*_정제.jsonl
```

**실행**
```
JSONL/jsonl_conversion.ipynb
```

**환경 설정** (`JSONL/.env`)
```
INPUT_DIR=../Preprocessing/output
OUTPUT_DIR=output
```

---

### 3단계 — 청킹 (`Chunking/`)

JSONL의 `embedding_text`를 200자 단위로 분할합니다.

```
JSONL/output/*_정제.jsonl  →  Chunking/output/*_정제.jsonl
```

**실행**
```
Chunking/splitting_chunking.ipynb
```

**환경 설정** (`Chunking/.env`)
```
INPUT_DIR=../JSONL/output
OUTPUT_DIR=output
CHUNK_SIZE=200
CHUNK_OVERLAP=50
```

---

### 4단계 — 임베딩 생성 및 OpenSearch 적재 (`embedding_indexer/`)

청킹된 JSONL을 읽어 인덱스별로 `embedding_text`를 재구성하고  
임베딩 벡터를 생성하여 OpenSearch 7개 인덱스에 적재합니다.

```
Chunking/output/*_정제.jsonl  →  OpenSearch (hr_basic_1~3, hr_performance_2~3, hr_salary_2~3)
```

**실행 (노트북)**
```
embedding_indexer/embedding_indexer.ipynb
```

**실행 (스크립트)**
```bash
python embedding_indexer/embedding_indexer.py
```

**환경 설정** (`embedding_indexer/.env`)
```
INPUT_DIR=../Chunking/output

OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USER=your_admin
OPENSEARCH_PASSWORD=your_password
OPENSEARCH_USE_SSL=true
OPENSEARCH_VERIFY_CERTS=false

EMBED_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
```

---

## OpenSearch 인덱스 구조

| 인덱스명 | 보안 레벨 | 포함 데이터 |
|---|---|---|
| `hr_basic_1` | 1 | 이름, 부서, 직급, 이메일 등 일반 인적사항 |
| `hr_basic_2` | 2 | 생년월일, 학력, 이전직장 등 |
| `hr_basic_3` | 3 | 주민등록번호, 주소, 퇴직정보 |
| `hr_performance_2` | 2 | 성과점수, 인사고과, 자격증, 포상이력 |
| `hr_performance_3` | 3 | 징계이력, 징계사유, 자격증수당여부 |
| `hr_salary_2` | 2 | 잔업시간, 미사용휴가일수 |
| `hr_salary_3` | 3 | 연봉, 계좌번호, 4대보험 등 |

**접근 권한**: `permission_level = MAX(부서레벨, 직급레벨)`  
본인 데이터는 레벨 무관 전체 접근 가능.

---

## 데이터 파일 구성

| 파일 | 컬럼 수 | 레코드 수 |
|---|---|---|
| `기본인사정보.csv` | 30개 | 2,000건 |
| `역량성과.csv` | 13개 | 2,000건 |
| `급여정보.csv` | 7개 | 2,000건 |

---

## 임베딩 모델

- 모델: `paraphrase-multilingual-MiniLM-L12-v2`
- 벡터 차원: 384
- 검색 방식: Hybrid Search (BM25 + KNN)
