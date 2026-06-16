# 두리안정보기술 인사 데이터 RAG 챗봇

인사 데이터를 OpenSearch에 적재하는 **데이터 파이프라인**, 권한 기반 검색·답변을 제공하는 **FastAPI 백엔드**,
그리고 사용자가 질문을 입력하는 **챗봇 UI(HTML + Alpine.js)** 까지 포함된 통합 시스템입니다.
원본 CSV를 읽어 정제 → 레코드 변환 → 청킹 → 임베딩 적재까지 한 번에 처리하고,
RAG(검색 증강 생성) 방식으로 LLM에게 답변을 만들게 합니다.

---

## 사전 준비

1. **OpenSearch 실행** — `localhost:9200`에서 동작 중이어야 합니다.
   nori 플러그인이 없으면 스크립트가 자동으로 안내합니다.

2. **Node.js 설치** — LTS 버전 권장 ([nodejs.org](https://nodejs.org)).
   설치 확인: `node -v`

3. **Ollama + LLM 모델 준비** — 백엔드가 답변 생성에 사용합니다.
   - Ollama 설치 ([ollama.com](https://ollama.com))
   - 모델 다운로드:
     ```bash
     ollama pull gemma3:4b
     ```
   - 실행 확인: `ollama list` 결과에 `gemma3:4b` 포함

4. **원본 CSV 준비** — `data/dataset/` 폴더에 아래 3개 파일을 둡니다.

   | 파일 | 컬럼 수 | 레코드 수 |
   |---|---|---|
   | `기본인사정보.csv` | 30개 | 2,000건 |
   | `역량성과.csv` | 13개 | 2,000건 |
   | `급여정보.csv` | 7개 | 2,000건 |

5. **`.env` 생성** — `.env.example`을 복사해서 `.env`로 만들고 값(특히 OpenSearch 비밀번호, HF 토큰)을 채웁니다.
   ```bash
   cp .env.example .env
   ```

---

## 실행

데이터 적재 → 백엔드 → 프론트 순서로 진행합니다.

### 1. Python 의존성 설치 (최초 1회)
```bash
pip install -r requirements.txt
```

### 2. Node.js 의존성 설치 (최초 1회)
`server/` 폴더로 이동해서 실행합니다.
```bash
cd server
npm install
cd ..
```

### 3. 파이프라인 실행 — 데이터 적재
```bash
python pipeline.py
```
- 1~3단계가 순서대로 실행됩니다.
- 처음 실행하면 7개 인덱스를 생성하고 전체를 적재합니다.
- 다시 실행하면 **값이 바뀐 직원만** 다시 적재하고, 나머지는 건너뜁니다.
- 사용자 사전(`config/user_dictionary.txt`)이 바뀌면 인덱스를 자동으로 재생성합니다.

### 4. 백엔드(FastAPI) 실행
```bash
uvicorn app.main:app --reload
```
- `localhost:8000`에서 떠야 합니다.

### 5. 프론트 서버(Express) 실행 — 새 터미널
```bash
node server/server.js
```
- `localhost:3000`에서 뜹니다.
- `index.html`을 서빙하고 `/api/*` 요청을 백엔드로 중계합니다.
- 프록시 덕분에 CORS 문제 없이 챗봇 API 호출 가능합니다.

### 6. 브라우저 접속
```
http://localhost:3000
```

---

### 동작 흐름

```
브라우저  →  localhost:3000           (index.html 다운로드)
브라우저  →  localhost:3000/api/...   (챗봇 요청)
               ↓ server.js 프록시
            localhost:8000/...        (FastAPI 백엔드)
               ↓
            OpenSearch (검색) + Ollama (답변 생성)
```

### 관련 파일

| 파일 | 역할 |
|---|---|
| `pipeline.py` | 데이터 전처리 + OpenSearch 적재 |
| `app/main.py` | FastAPI 백엔드 (챗봇 API) |
| `index.html` | 챗봇 UI (Alpine.js 기반, 단일 HTML 파일) |
| `server/server.js`  | Express 프록시 서버 (정적 파일 서빙 + API 중계) |
| `server/package.json` | Node.js 의존성 명세 |
| `requirements.txt` | Python 의존성 명세 |

---

## 프로젝트 구조

```
pipeline.py            전체 파이프라인 (1~3단계)
requirements.txt       Python 의존성

app/                   FastAPI 백엔드
  main.py              API 엔드포인트 (/chat, /rag-chat)
  services/            검색·LLM·질문 분석 로직

server/                Express 프론트 서버 (정적 파일 + API 프록시)
  server.js
  package.json

index.html             챗봇 UI (단일 파일, Alpine.js + Tailwind)

config/
  user_dictionary.txt  nori 사용자 사전

data/                  실행 시 원본 CSV를 두는 폴더 (CSV는 저장소에 포함되지 않음)
```

> `.env`, 원본 CSV(`data/dataset/`), 실행 중 생성되는 로그·상태 파일은
> 저장소에 포함되지 않습니다(위 *사전 준비* 참고).

---

## 파이프라인 구조

3개 단계가 하나의 스크립트(`pipeline.py`)로 통합되어 있습니다.
단계 사이의 데이터는 **중간 파일을 만들지 않고 메모리로 전달**합니다.

```
1단계: 전처리      원본 CSV 검증·교정
   ↓ (메모리)
2단계: 레코드 변환  직원별 레코드(dict) 생성
   ↓ (메모리)
3단계: 인덱싱      인덱스별 (필드 필터링 → 청킹 → 임베딩 → OpenSearch 적재)
                  변경된 직원만 증분 적재
```

> **청킹은 별도 단계가 아니라 3단계 인덱싱 안에서 인덱스별로 수행**됩니다.
> 전체 필드를 한꺼번에 청킹하면 그 인덱스에 안 들어갈 필드까지 토큰을 차지해
> 같은 인덱스의 필드가 여러 청크에 흩어집니다 (예: `hr_basic_3` 의
> 주민등록번호와 주소가 다른 청크에 들어감).
> 인덱스별로 필요한 필드만 먼저 골라낸 뒤 청킹하면, 한 인덱스의 필드가
> 같은 청크에 모입니다.

증분 적재 기준은 별도 상태 파일이 아니라 **OpenSearch에 적재된 값과 직접 비교**합니다.
값이 바뀐 직원만 다시 임베딩하고, 바뀐 필드는 `changed` 배열에 이력으로 남깁니다.

---

## 단계별 설명

### 1단계 — 전처리
원본 CSV를 컬럼별로 검증·교정하고, 문제 있는 행은 제거합니다.
결측치는 모두 `미입력`으로 통일합니다.

### 2단계 — 레코드 변환
정제된 데이터를 직원별 레코드로 만들고, 검색용 `embedding_text`를 구성합니다.
이름·부서·직급은 `hr_basic_1`에만 들어갑니다.

### 3단계 — 인덱싱 (인덱스별 청킹 + 변경감지)
인덱스별로 아래 흐름을 수행합니다.

1. **필드 필터링** — 이 인덱스에 들어갈 필드만 골라냅니다.
2. **청킹** — 골라낸 텍스트를 토큰 한계(`MAX_TOKENS`)에 맞춰 청크로 나눕니다.
   인덱스 단위로 청킹하므로 같은 인덱스의 필드끼리 같은 청크에 모입니다.
3. **임베딩** — 청크 텍스트를 임베딩 벡터로 변환합니다.
4. **변경감지 + 적재** — OpenSearch에 이미 있는 값과 비교해 **바뀐 직원만**
   재적재하며, 바뀐 필드는 `changed` 이력에 기록합니다.

> 퇴사 직원도 인덱스에서 삭제하지 않습니다(인사 이력 보존). 접근 권한 회수는 적재가 아닌 별도 영역에서 처리합니다.

---

## OpenSearch 인덱스 구조

| 인덱스명 | 보안 레벨 | 포함 데이터 |
|---|---|---|
| `hr_basic_1` | 1 | 이름, 성별, 나이, 부서, 팀, 직급, 직책, 입사일, 근속기간, 채용경로, 계약형태, 회사명, 사업장위치, 이메일 |
| `hr_basic_2` | 2 | 생년월일, 병역, 학력, 출신대학, 학점, 전화번호, 이전직장명, 이전최종직급, 이전담당업무 |
| `hr_basic_3` | 3 | 주민등록번호, 주소, 퇴직구분, 퇴직일자 |
| `hr_performance_2` | 2 | 성과점수, 인사고과(2020~2024), 자격증, TOEIC점수, 포상이력 |
| `hr_performance_3` | 3 | 징계이력, 징계사유, 자격증수당여부 |
| `hr_salary_2` | 2 | 잔업시간, 미사용휴가일수 |
| `hr_salary_3` | 3 | 연봉, 급여은행, 계좌번호, 4대보험가입여부 |

**접근 권한**: `permission_level = MAX(부서레벨, 직급레벨)`
본인 데이터는 레벨 무관 전체 접근 가능.
(권한 계산은 파이프라인이 아니라 검색·접근제어 영역에서 수행합니다.)

---

## 임베딩 모델

- 모델: `paraphrase-multilingual-MiniLM-L12-v2`
- 벡터 차원: 384
- 검색 방식: Hybrid Search (BM25 + KNN)
- KNN 엔진: lucene / hnsw / cosinesimil

---

## 에러 로그

1~3단계에서 발생한 문제는 `data/error.log` 한 파일에 단계별로 구분되어 기록됩니다.

- **1단계**: 전처리 검증 에러 (어떤 값이 왜 교정·제거됐는지)
- **3단계 청킹 경고**: 빈 텍스트 스킵 / 토큰 한계 초과 (청킹이 3단계 인덱싱 안에서 일어남)
- **3단계 적재 실패**: OpenSearch 적재 실패
