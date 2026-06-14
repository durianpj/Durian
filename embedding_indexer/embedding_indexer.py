import os
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

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR            = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')
INPUT_DIR           = Path(os.getenv('INPUT_DIR', str(BASE_DIR / 'Chunking' / 'output')))
OPENSEARCH_HOME     = Path(os.getenv('OPENSEARCH_HOME', str(BASE_DIR / 'opensearch-3.3.2'))).resolve()
LAST_INDEXED_FILE   = Path(__file__).resolve().parent / 'last_indexed.txt'
LAST_USER_DICT_FILE = Path(__file__).resolve().parent / 'last_user_dictionary.txt'
USER_DICT_FILE      = Path(__file__).resolve().parent / 'user_dictionary.txt'
SYNONYM_FILE        = Path(__file__).resolve().parent / 'synonym.txt'

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
        'fields': [
            '이름', '성별', '나이', '입사일', '근속기간',
            '채용경로', '계약형태', '회사명', '사업장위치',
            '부서', '팀', '직책', '직급', '이메일',
        ],
    },
    'hr_basic_2': {
        'security_level': 2,
        'fields': [
            '생년월일', '병역', '학력', '출신대학', '학점',
            '전화번호', '이전직장명', '이전최종직급', '이전담당업무',
        ],
    },
    'hr_basic_3': {
        'security_level': 3,
        'fields': [
            '주민등록번호', '주소', '퇴직구분', '퇴직일자',
        ],
    },
    'hr_performance_2': {
        'security_level': 2,
        'fields': [
            '성과점수',
            '인사고과_2020', '인사고과_2021', '인사고과_2022',
            '인사고과_2023', '인사고과_2024',
            '자격증', 'TOEIC점수', '포상이력',
        ],
    },
    'hr_performance_3': {
        'security_level': 3,
        'fields': [
            '징계이력', '징계사유', '자격증수당여부',
        ],
    },
    'hr_salary_2': {
        'security_level': 2,
        'fields': [
            '잔업시간', '미사용휴가일수',
        ],
    },
    'hr_salary_3': {
        'security_level': 3,
        'fields': [
            '연봉', '급여은행', '계좌번호', '4대보험가입여부',
        ],
    },
}

# ── 사용자 사전 변경 감지 ──────────────────────────────────────────────────────
def get_dict_hash():
    # 사용자 사전 변경 감지용 해시값을 만든다.
    # user_dictionary.txt(토크나이저 사전)와 synonym.txt(동의어 사전)를 모두 반영하므로
    # 두 파일 중 하나라도 내용이 바뀌면 해시가 달라져 인덱스 재생성 대상이 된다.
    hasher = hashlib.md5()
    for path in [USER_DICT_FILE, SYNONYM_FILE]:
        if path.exists():
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def read_last_dict_hash():
    if not LAST_USER_DICT_FILE.exists():
        return ''
    return LAST_USER_DICT_FILE.read_text(encoding='utf-8').strip()


def write_last_dict_hash(hash_value):
    LAST_USER_DICT_FILE.write_text(hash_value, encoding='utf-8')


# ── 마지막 인덱싱 시간 관리 ────────────────────────────────────────────────────
def read_last_indexed():
    if not LAST_INDEXED_FILE.exists():
        return None
    return LAST_INDEXED_FILE.read_text(encoding='utf-8').strip()


def write_last_indexed(timestamp):
    LAST_INDEXED_FILE.write_text(timestamp, encoding='utf-8')


# ── 인덱스 매핑 ────────────────────────────────────────────────────────────────
def build_index_body(security_level):
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
        except Exception as error:
            print(f'{filename} 복사 실패 → {error}')
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
    except Exception as error:
        print(f'nori 플러그인 다운로드 실패 → {error}')
        print('네트워크 연결을 확인하거나 수동으로 플러그인을 설치해주세요.')
        raise SystemExit(1)

    nori_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as archive:
        archive.extractall(nori_dir)
    zip_path.unlink()

    print('nori 플러그인 설치 완료!')
    print('OpenSearch를 재시작한 후 다시 실행해 주세요.')
    raise SystemExit(0)


def parse_embedding_text(text):
    # embedding_text는 한 줄에 하나씩 '필드명: 값' 형식 (줄바꿈으로 구분)
    # 줄 단위로 나눈 뒤 첫 ': ' 기준으로 필드명과 값을 분리한다
    result = {}
    for line in text.split('\n'):
        line = line.strip()
        if ': ' not in line:
            continue
        key, value = line.split(': ', 1)
        result[key.strip()] = value.strip()
    return result


def build_filtered_text(parsed, fields):
    # 청크에서 이 인덱스의 필드만 골라 임베딩용 텍스트를 만든다.
    # 청킹 단계와 같은 줄바꿈(\n) 구분으로 형식을 통일한다.
    parts = [f'{field}: {parsed[field]}' for field in fields if parsed.get(field)]
    return '\n'.join(parts)


def create_indices(client):
    # 인덱스를 생성하고, 사용자 사전이 바뀐 경우 재생성 여부를 사용자에게 확인한다.
    #
    # [사전 변경 시 재생성 확인 기능]
    #   사전(user_dictionary.txt / synonym.txt)이 바뀌면 인덱스를 새 사전으로
    #   다시 만들어야 검색에 반영된다. 재생성은 기존 인덱스를 삭제(= 적재된 데이터
    #   전부 삭제)하는 작업이라 자동으로 하지 않고 y/n 으로 물어본다.
    #     - y : 기존 인덱스를 삭제하고 재생성한 뒤, 변경 해시를 갱신한다.
    #     - n : 기존 인덱스를 그대로 두고 해시도 갱신하지 않는다.
    #           (변경이 묻히지 않도록 다음 실행 때 다시 안내한다)
    #     - 첫 실행처럼 삭제할 인덱스가 없으면 묻지 않고 바로 생성한다.
    #
    # [주의] input() 으로 입력을 받으므로 이 스크립트는 대화형이다. 스케줄러 자동 실행,
    #        CI, 백그라운드 등 사람이 없는 환경에서는 입력 대기 상태로 멈출 수 있다.
    #        자동 실행이 필요하면 이 부분을 환경변수 기반(예: AUTO_REINDEX)으로
    #        분기하도록 수정해야 한다.
    print('=== 인덱스 생성 ===')

    current_hash = get_dict_hash()
    last_hash    = read_last_dict_hash()
    dict_changed = current_hash != last_hash

    # 사전이 바뀌었고, 실제로 삭제할 기존 인덱스가 있을 때만 사용자에게 확인한다.
    apply_change = dict_changed
    if dict_changed:
        existing = [name for name in INDEX_CONFIG if client.indices.exists(index=name)]
        if existing:
            print('  사용자 사전(user_dictionary.txt / synonym.txt) 변경 감지됨')
            print(f'  재생성하려면 기존 인덱스 {len(existing)}개를 삭제해야 합니다: {", ".join(existing)}')
            answer = input('  삭제하고 재생성할까요? (y/n): ').strip().lower()
            if answer == 'y':
                for name in existing:
                    client.indices.delete(index=name)
                    print(f'  삭제: {name}')
            else:
                print('  재생성을 건너뜁니다. (기존 인덱스 유지, 다음 실행 때 다시 안내)')
                apply_change = False

    for name, config in INDEX_CONFIG.items():
        if client.indices.exists(index=name):
            print(f'  이미 존재: {name}  (건너뜀)')
            continue
        try:
            client.indices.create(index=name, body=build_index_body(config['security_level']))
            print(f'  생성 완료: {name}  (security_level={config["security_level"]})')
        except Exception as error:
            print(f'  인덱스 생성 실패: {name}  → {error}')
            raise SystemExit(1)

    if apply_change:
        write_last_dict_hash(current_hash)


def get_existing_employee_ids(client, index_name):
    # 인덱스에 이미 적재된 직원 ID 목록을 가져온다 (신규/기존 직원을 구분하기 위함).
    #
    # terms 집계는 '직원 ID를 종류별로 모아 명단을 만들어줘'라는 요청이고,
    # size 는 '단, 최대 이 개수까지만 만들어줘'라는 상한이다.
    # 지금은 10만(100000)으로 잡혀 있어 직원이 10만 명을 넘으면 초과분이 명단에서 빠지고,
    # 빠진 직원은 '신규'로 잘못 판단되어 중복 적재될 수 있다.
    #   - 현재 직원 수(약 2천 명)에서는 여유가 커서 전혀 문제 없다.
    #   - 만약 직원이 10만 명을 넘어설 규모가 되면, 이 size 를 키우는 것만으로는 부족하고
    #     composite 집계(페이지 단위로 나눠 가져오기)로 바꿔야 정확하다.
    resp = client.search(
        index=index_name,
        body={
            'size': 0,
            'aggs': {'employees': {'terms': {'field': 'employee_id', 'size': 100000}}},
        },
    )
    return {bucket['key'] for bucket in resp['aggregations']['employees']['buckets']}


def needs_update(rec, last_indexed):
    if last_indexed is None:
        return True
    changed = rec.get('changed', [])
    if not changed:
        return False
    latest = max(entry['timestamp'] for entry in changed)
    return latest > last_indexed


def load_data(client, model, last_indexed):
    # JSONL 청크들을 읽어 OpenSearch 인덱스에 적재한다.
    #
    # [전체 흐름]
    #   1) INPUT_DIR의 JSONL 파일을 하나씩 읽는다 (한 줄 = 청크 1건).
    #   2) 각 인덱스(INDEX_CONFIG)를 돌면서, 그 인덱스의 fields에 해당하는 청크만 골라 적재한다.
    #      (= 필드 기반 라우팅. 한 파일이 보안등급별 여러 인덱스로 나뉘어 들어갈 수 있다)
    #   3) 이미 적재된 직원은 건너뛰고, 새 직원·변경된 직원만 처리한다 (증분 적재).
    #   4) 고른 청크를 임베딩 벡터로 바꿔 OpenSearch 문서로 만들고 한 번에 대량 적재한다.
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
            with open(jsonl_file, encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as error:
            print(f'JSONL 파일 읽기 실패: {jsonl_file.name} → {error}')
            raise SystemExit(1)

        if not records:
            continue

        print(f'\n  {jsonl_file.name}  총 {len(records):,}건')

        # 필드 기반 라우팅: 모든 인덱스 순회하면서 매칭되는 필드 있으면 적재
        for index_name, cfg in INDEX_CONFIG.items():
            fields = cfg['fields']

            existing_ids = get_existing_employee_ids(client, index_name)

            # 이 인덱스에 처리할 직원을 추린다.
            #   - new_ids    : 아직 인덱스에 없는 신규 직원
            #   - update_ids : 이미 있지만 데이터가 바뀐 직원 (needs_update로 판단)
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

            target_records = [record for record in records if record.get('employee_id') in process_ids]

            # 처리 대상 청크에서 이 인덱스의 필드만 남긴 텍스트를 만든다(보안등급 분리).
            # 내용이 있는 청크만 골라 두 리스트에 같은 순서로 쌓아둔다.
            #   - texts         : 임베딩할 텍스트 목록
            #   - valid_records : texts와 같은 순서로 짝지어지는 원본 청크
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

            # 텍스트 목록을 한꺼번에 임베딩 벡터로 변환한다 (texts와 같은 순서로 vectors 생성).
            vectors = model.encode(
                texts, batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False, convert_to_numpy=True,
            )

            # valid_records(원본 청크) / texts(임베딩 텍스트) / vectors(임베딩 결과)는
            # 모두 같은 순서·길이로 만들어졌다. zip으로 셋을 나란히 묶어 한 건씩 꺼내면
            # 같은 위치의 항목끼리 한 청크를 이룬다. 이를 OpenSearch 문서로 변환한다.
            chunk_counters = {}   # 직원별 청크 번호를 매기는 카운터
            actions = []
            for rec, text, vector in zip(valid_records, texts, vectors):
                emp_id = rec.get('employee_id', '')
                # 한 직원에게 청크가 여러 개일 수 있으므로 1부터 번호를 매겨 문서 id를 만든다.
                # 예: EMP0001_0001, EMP0001_0002 ...
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
                        'embedding_text':   text,
                        'embedding_vector': vector.tolist(),
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
    except Exception as error:
        print(f'\nOpenSearch 연결 실패 →\n {error}\n')
        print(f'OpenSearch가 실행 중인지 확인해주세요. ({OPENSEARCH_HOST}:{OPENSEARCH_PORT})')
        raise SystemExit(1)

    ensure_user_dictionary()
    ensure_nori_plugin(client)

    last_indexed = read_last_indexed()

    print(f'임베딩 모델 로딩: {EMBEDDING_MODEL}')
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as error:
        print(f'임베딩 모델 로딩 실패 → {error}')
        print(f'모델명({EMBEDDING_MODEL})과 네트워크 연결을 확인해주세요.')
        raise SystemExit(1)

    create_indices(client)
    load_data(client, model, last_indexed)

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    write_last_indexed(now)

    print(f'\n=== 전체 완료 ({now}) ===')


if __name__ == '__main__':
    main()
