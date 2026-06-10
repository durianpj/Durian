import os
import re
import json
import shutil
import hashlib
import zipfile
import urllib.request
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR            = Path(__file__).resolve().parent.parent
INPUT_DIR           = Path(os.getenv('INPUT_DIR', str(BASE_DIR / 'Chunking' / 'output')))
OPENSEARCH_HOME     = Path(os.getenv('OPENSEARCH_HOME', str(BASE_DIR / 'opensearch-3.3.2'))).resolve()
LAST_INDEXED_FILE   = Path(__file__).resolve().parent / 'last_indexed.txt'
LAST_DICT_HASH_FILE = Path(__file__).resolve().parent / 'last_dict_hash.txt'
USER_DICT_FILE      = BASE_DIR / 'user_dictionary.txt'
SYNONYM_FILE        = BASE_DIR / 'synonym.txt'

# ── OpenSearch 연결 설정 ────────────────────────────────────────────────────────
OPENSEARCH_HOST         = os.getenv('OPENSEARCH_HOST', 'localhost')
OPENSEARCH_PORT         = int(os.getenv('OPENSEARCH_PORT', 9200))
OPENSEARCH_USER         = os.environ['OPENSEARCH_USER']
OPENSEARCH_PASSWORD     = os.environ['OPENSEARCH_PASSWORD']
OPENSEARCH_USE_SSL      = os.getenv('OPENSEARCH_USE_SSL', 'true').lower() == 'true'
OPENSEARCH_VERIFY_CERTS = os.getenv('OPENSEARCH_VERIFY_CERTS', 'false').lower() == 'true'

# ── 임베딩 모델 ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = os.getenv('EMBED_MODEL_NAME', 'paraphrase-multilingual-MiniLM-L12-v2')
EMBEDDING_DIM        = int(os.getenv('EMBED_DIMENSION', '384'))
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBED_BATCH_SIZE', '64'))

# ── KNN 설정 ───────────────────────────────────────────────────────────────────
KNN_ENGINE     = os.getenv('KNN_ENGINE',     'lucene')
KNN_METHOD     = os.getenv('KNN_METHOD',     'hnsw')
KNN_SPACE_TYPE = os.getenv('KNN_SPACE_TYPE', 'cosinesimil')

# ── 인덱스별 설정 ───────────────────────────────────────────────────────────────
INDEX_CONFIG = {
    'hr_basic_1': {
        'security_level': 1,
        'source': '기본인사정보',
        'fields': [
            '이름', '성별', '나이', '입사일', '근속기간',
            '채용경로', '계약형태', '회사명', '사업장위치',
            '부서', '팀', '직책', '부서레벨', '직급레벨', '직급', '이메일',
        ],
    },
    'hr_basic_2': {
        'security_level': 2,
        'source': '기본인사정보',
        'fields': [
            '이름', '부서', '직급',
            '생년월일', '병역', '학력', '출신대학', '학점',
            '전화번호', '이전직장명', '이전최종직급', '이전담당업무',
        ],
    },
    'hr_basic_3': {
        'security_level': 3,
        'source': '기본인사정보',
        'fields': [
            '이름', '부서', '직급',
            '주민등록번호', '주소', '퇴직구분', '퇴직일자',
        ],
    },
    'hr_performance_2': {
        'security_level': 2,
        'source': '역량성과',
        'fields': [
            '이름', '부서', '직급',
            '성과점수',
            '인사고과_2020', '인사고과_2021', '인사고과_2022',
            '인사고과_2023', '인사고과_2024',
            '자격증', 'TOEIC점수', '포상이력',
        ],
    },
    'hr_performance_3': {
        'security_level': 3,
        'source': '역량성과',
        'fields': [
            '이름', '부서', '직급',
            '징계이력', '징계사유', '자격증수당여부',
        ],
    },
    'hr_salary_2': {
        'security_level': 2,
        'source': '급여정보',
        'fields': [
            '이름', '부서', '직급',
            '잔업시간', '미사용휴가일수',
        ],
    },
    'hr_salary_3': {
        'security_level': 3,
        'source': '급여정보',
        'fields': [
            '이름', '부서', '직급',
            '연봉', '급여은행', '계좌번호', '4대보험가입여부',
        ],
    },
}

# source → 해당 인덱스 목록
SOURCE_TO_INDICES = {}
for idx_name, cfg in INDEX_CONFIG.items():
    SOURCE_TO_INDICES.setdefault(cfg['source'], []).append(idx_name)

# 전체 필드 목록 (embedding_text 파싱용)
ALL_FIELDS = sorted(
    {f for cfg in INDEX_CONFIG.values() for f in cfg['fields']},
    key=len, reverse=True
)


# ── 사용자 사전 변경 감지 ──────────────────────────────────────────────────────
def get_dict_hash() -> str:
    if not USER_DICT_FILE.exists():
        return ''
    return hashlib.md5(USER_DICT_FILE.read_bytes()).hexdigest()


def read_last_dict_hash() -> str:
    if not LAST_DICT_HASH_FILE.exists():
        return ''
    return LAST_DICT_HASH_FILE.read_text(encoding='utf-8').strip()


def write_last_dict_hash(hash_value: str):
    LAST_DICT_HASH_FILE.write_text(hash_value, encoding='utf-8')


# ── 마지막 인덱싱 시간 관리 ────────────────────────────────────────────────────
def read_last_indexed():
    if not LAST_INDEXED_FILE.exists():
        return None
    return LAST_INDEXED_FILE.read_text(encoding='utf-8').strip()


def write_last_indexed(timestamp: str):
    LAST_INDEXED_FILE.write_text(timestamp, encoding='utf-8')


# ── 인덱스 매핑 ────────────────────────────────────────────────────────────────
def build_index_body(security_level: int) -> dict:
    return {
        'settings': {
            'index': {'knn': True},
            'analysis': {
                'tokenizer': {
                    'nori_tokenizer': {
                        'type': 'nori_tokenizer',
                        'user_dictionary': 'user_dictionary.txt',
                    }
                },
                'filter': {
                    'synonym_filter': {
                        'type': 'synonym',
                        'synonyms_path': 'synonym.txt',
                    }
                },
                'analyzer': {
                    'korean_analyzer': {
                        'type': 'custom',
                        'tokenizer': 'nori_tokenizer',
                        'filter': ['synonym_filter'],
                    }
                }
            },
        },
        'mappings': {
            '_meta': {'security_level': security_level},
            'properties': {
                'employee_id':      {'type': 'keyword'},
                'employee_name':    {'type': 'keyword'},
                'department':       {'type': 'keyword'},
                'department_level': {'type': 'integer'},
                'job_grade':        {'type': 'keyword'},
                'job_grade_level':  {'type': 'integer'},
                'source':           {'type': 'keyword'},
                'timestamp':        {'type': 'keyword'},
                'changed':          {'type': 'object', 'enabled': False},
                'embedding_text':   {'type': 'text', 'analyzer': 'korean_analyzer'},
                'embedding_vector': {
                    'type': 'knn_vector',
                    'dimension': EMBEDDING_DIM,
                    'method': {
                        'engine': KNN_ENGINE,
                        'name': KNN_METHOD,
                        'space_type': KNN_SPACE_TYPE,
                    },
                },
            }
        },
    }


def ensure_user_dictionary():
    for src, filename in [(USER_DICT_FILE, 'user_dictionary.txt'), (SYNONYM_FILE, 'synonym.txt')]:
        dst = OPENSEARCH_HOME / 'config' / filename
        if not src.exists():
            print(f'{filename} 파일이 없습니다. 건너뜀')
            continue
        try:
            shutil.copy(src, dst)
            print(f'복사 완료: {dst}')
        except Exception as e:
            print(f'{filename} 복사 실패 → {e}')
            print(f'OPENSEARCH_HOME 경로를 확인해주세요. ({OPENSEARCH_HOME})')
            raise SystemExit(1)


def ensure_nori_plugin(client):
    nori_dir = OPENSEARCH_HOME / 'plugins' / 'analysis-nori'
    if nori_dir.exists():
        return

    info = client.info()
    version = info['version']['number']
    url = (
        f'https://artifacts.opensearch.org/releases/plugins/'
        f'analysis-nori/{version}/analysis-nori-{version}.zip'
    )
    zip_path = Path(os.getenv('TEMP', '/tmp')) / f'analysis-nori-{version}.zip'

    print(f'nori 플러그인 다운로드 중... ({url})')
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f'nori 플러그인 다운로드 실패 → {e}')
        print('네트워크 연결을 확인하거나 수동으로 플러그인을 설치해주세요.')
        raise SystemExit(1)

    nori_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(nori_dir)
    zip_path.unlink()

    print('nori 플러그인 설치 완료!')
    print('OpenSearch를 재시작한 후 다시 실행해 주세요.')
    raise SystemExit(0)


def parse_embedding_text(text):
    pattern = '(' + '|'.join(re.escape(f) for f in ALL_FIELDS) + '): '
    parts = re.split(pattern, text)
    result = {}
    i = 1
    while i < len(parts) - 1:
        key = parts[i]
        val = parts[i + 1].strip()
        result[key] = val
        i += 2
    return result


def build_filtered_text(parsed, fields):
    parts = [f'{f}: {parsed[f]}' for f in fields if parsed.get(f)]
    return ' '.join(parts)


def create_indices(client):
    print('=== 인덱스 생성 ===')

    current_hash = get_dict_hash()
    last_hash    = read_last_dict_hash()
    dict_changed = current_hash != last_hash

    if dict_changed:
        print('  사용자 사전 변경 감지 → 기존 인덱스 삭제 후 재생성')
        for name in INDEX_CONFIG:
            if client.indices.exists(index=name):
                client.indices.delete(index=name)
                print(f'  삭제: {name}')

    for name, config in INDEX_CONFIG.items():
        if client.indices.exists(index=name):
            print(f'  이미 존재: {name}  (건너뜀)')
            continue
        try:
            client.indices.create(index=name, body=build_index_body(config['security_level']))
            print(f'  생성 완료: {name}  (security_level={config["security_level"]})')
        except Exception as e:
            print(f'  인덱스 생성 실패: {name}  → {e}')
            raise SystemExit(1)

    if dict_changed:
        write_last_dict_hash(current_hash)


def get_existing_employee_ids(client, index_name):
    resp = client.search(
        index=index_name,
        body={
            'size': 0,
            'aggs': {'employees': {'terms': {'field': 'employee_id', 'size': 100000}}},
        },
    )
    return {b['key'] for b in resp['aggregations']['employees']['buckets']}


def needs_update(rec, last_indexed):
    if last_indexed is None:
        return True
    changed = rec.get('changed', [])
    if not changed:
        return False
    latest = max(entry['timestamp'] for entry in changed)
    return latest > last_indexed


def load_data(client, model, last_indexed):
    print('\n=== 데이터 적재 ===')
    print(f'  기준 시간: {last_indexed or "첫 실행 (전체 적재)"}')

    jsonl_files = sorted(INPUT_DIR.glob('*.jsonl'))
    if not jsonl_files:
        print(f'JSONL 파일 없음: {INPUT_DIR}')
        print('JSONL 변환 스크립트를 먼저 실행해주세요.')
        raise SystemExit(1)

    for jsonl_file in jsonl_files:
        records = []
        try:
            with open(jsonl_file, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            print(f'JSONL 파일 읽기 실패: {jsonl_file.name} → {e}')
            raise SystemExit(1)

        if not records:
            continue

        source_name = records[0].get('source', '')
        index_names = SOURCE_TO_INDICES.get(source_name)

        if not index_names:
            print(f'  건너뜀: {jsonl_file.name}  (source={source_name})')
            continue

        print(f'\n  {jsonl_file.name}  ({source_name})  총 {len(records):,}건')

        for index_name in index_names:
            fields = INDEX_CONFIG[index_name]['fields']

            existing_ids = get_existing_employee_ids(client, index_name)

            new_ids     = set()
            update_ids  = set()
            for rec in records:
                emp_id = rec.get('employee_id', '')
                if emp_id not in existing_ids:
                    new_ids.add(emp_id)
                elif needs_update(rec, last_indexed):
                    update_ids.add(emp_id)

            process_ids = new_ids | update_ids

            if not process_ids:
                print(f'    [{index_name}]  변경 없음 (건너뜀)')
                continue

            # 변경된 직원의 기존 청크 삭제 (청크 수 변동 대비)
            if update_ids:
                client.delete_by_query(
                    index=index_name,
                    body={'query': {'terms': {'employee_id': list(update_ids)}}},
                    params={'refresh': 'true'},
                )

            target_records = [r for r in records if r.get('employee_id') in process_ids]

            texts = []
            valid_records = []
            for rec in target_records:
                parsed   = parse_embedding_text(rec.get('embedding_text', ''))
                filtered = build_filtered_text(parsed, fields)
                if not filtered:
                    continue
                texts.append(filtered)
                valid_records.append(rec)

            if not texts:
                continue

            vectors = model.encode(
                texts, batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False, convert_to_numpy=True,
            )

            chunk_counters = {}
            actions = []
            for i, rec in enumerate(valid_records):
                emp_id = rec.get('employee_id', str(i))
                chunk_counters[emp_id] = chunk_counters.get(emp_id, 0) + 1
                doc_id = f"{emp_id}_{chunk_counters[emp_id]:04d}"
                actions.append({
                    '_index':  index_name,
                    '_id':     doc_id,
                    '_source': {
                        'employee_id':      rec.get('employee_id', ''),
                        'employee_name':    rec.get('employee_name', ''),
                        'department':       rec.get('department', ''),
                        'department_level': int(rec.get('department_level', 0) or 0),
                        'job_grade':        rec.get('job_grade', ''),
                        'job_grade_level':  int(rec.get('job_grade_level', 0) or 0),
                        'source':           rec.get('source', ''),
                        'timestamp':        rec.get('timestamp', ''),
                        'changed':          rec.get('changed', []),
                        'embedding_text':   texts[i],
                        'embedding_vector': vectors[i].tolist(),
                    },
                })

            success, failed = helpers.bulk(client, actions, raise_on_error=False)
            print(
                f'    [{index_name}]  '
                f'신규 {len(new_ids)}명 / 변경 {len(update_ids)}명  →  '
                f'{success:,}건 성공 / 실패 {len(failed)}건'
            )
            if failed:
                print(f'    경고: {len(failed)}건 적재 실패 → {failed[0]}')


def main():
    client = OpenSearch(
        hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=OPENSEARCH_USE_SSL,
        verify_certs=OPENSEARCH_VERIFY_CERTS,
        ssl_show_warn=False,
    )

    try:
        client.info()
    except Exception as e:
        print(f'\nOpenSearch 연결 실패 →\n {e}\n')
        print(f'OpenSearch가 실행 중인지 확인해주세요. ({OPENSEARCH_HOST}:{OPENSEARCH_PORT})')
        raise SystemExit(1)

    ensure_user_dictionary()
    ensure_nori_plugin(client)

    last_indexed = read_last_indexed()

    print(f'임베딩 모델 로딩: {EMBEDDING_MODEL}')
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        print(f'임베딩 모델 로딩 실패 → {e}')
        print(f'모델명({EMBEDDING_MODEL})과 네트워크 연결을 확인해주세요.')
        raise SystemExit(1)

    create_indices(client)
    load_data(client, model, last_indexed)

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    write_last_indexed(now)

    print(f'\n=== 전체 완료 ({now}) ===')


if __name__ == '__main__':
    main()
