import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
INPUT_DIR = Path(os.getenv('INPUT_DIR', str(BASE_DIR / 'Preprocessing' / 'output')))

# ── OpenSearch 연결 설정 ────────────────────────────────────────────────────────
OPENSEARCH_HOST         = os.getenv('OPENSEARCH_HOST', 'localhost')
OPENSEARCH_PORT         = int(os.getenv('OPENSEARCH_PORT', 9200))
OPENSEARCH_USER         = os.getenv('OPENSEARCH_USER', 'admin')
OPENSEARCH_PASSWORD     = os.getenv('OPENSEARCH_PASSWORD', 'Admin1234!')
OPENSEARCH_USE_SSL      = os.getenv('OPENSEARCH_USE_SSL', 'true').lower() == 'true'
OPENSEARCH_VERIFY_CERTS = os.getenv('OPENSEARCH_VERIFY_CERTS', 'false').lower() == 'true'

# ── 임베딩 모델 (데이터구조정의서 기준 384차원) ─────────────────────────────────
EMBEDDING_MODEL = os.getenv('EMBED_MODEL_NAME', 'paraphrase-multilingual-MiniLM-L12-v2')

# 결측값 처리 기준
MISSING = {'', '미입력', 'nan', 'NaN', 'None', 'none'}

# ── 인덱스별 설정 ───────────────────────────────────────────────────────────────
# security_level : 접근 권한 레벨 (인덱스 자체 설정으로 관리, 문서에는 포함 안 함)
# source         : 읽어올 전처리 CSV 파일명 (확장자 제외)
# fields         : 해당 인덱스에 포함할 필드 목록
# doc_prefix     : 문서 ID 앞에 붙는 유형 코드 (BAS / PERF / SAL)
INDEX_CONFIG = {
    'hr_basic_1': {
        'security_level': 1,
        'source': '기본인사정보_정제',
        'fields': [
            '이름', '성별', '나이', '입사일', '근속기간',
            '채용경로', '계약형태', '회사명', '사업장위치',
            '팀', '직책', '부서레벨', '직급레벨', '이메일',
        ],
        'doc_prefix': 'BAS',
    },
    'hr_basic_2': {
        'security_level': 2,
        'source': '기본인사정보_정제',
        'fields': [
            '생년월일', '병역', '학력', '출신대학', '학점',
            '전화번호', '이전직장명', '이전최종직급', '이전담당업무',
        ],
        'doc_prefix': 'BAS',
    },
    'hr_basic_3': {
        'security_level': 3,
        'source': '기본인사정보_정제',
        'fields': ['주민등록번호', '주소', '퇴직구분', '퇴직일자'],
        'doc_prefix': 'BAS',
    },
    'hr_performance_2': {
        'security_level': 2,
        'source': '역량성과_정제',
        'fields': [
            '성과점수',
            '인사고과_2020', '인사고과_2021', '인사고과_2022',
            '인사고과_2023', '인사고과_2024',
            '자격증', 'TOEIC점수', '포상이력',
        ],
        'doc_prefix': 'PERF',
    },
    'hr_performance_3': {
        'security_level': 3,
        'source': '역량성과_정제',
        'fields': ['징계이력', '징계사유', '자격증수당여부'],
        'doc_prefix': 'PERF',
    },
    'hr_salary_2': {
        'security_level': 2,
        'source': '급여정보_정제',
        'fields': ['잔업시간', '미사용휴가일수'],
        'doc_prefix': 'SAL',
    },
    'hr_salary_3': {
        'security_level': 3,
        'source': '급여정보_정제',
        'fields': ['연봉', '급여은행', '계좌번호', '4대보험가입여부'],
        'doc_prefix': 'SAL',
    },
}

# ── 인덱스 매핑 (7개 인덱스 공통 구조) ─────────────────────────────────────────
INDEX_BODY = {
    'settings': {
        'index': {'knn': True},
        'analysis': {
            'analyzer': {
                'korean_analyzer': {
                    'type': 'custom',
                    'tokenizer': 'nori_tokenizer',
                }
            }
        },
    },
    'mappings': {
        'properties': {
            'employee_id':      {'type': 'keyword'},
            'department':       {'type': 'keyword'},
            'position':         {'type': 'keyword'},
            'embedding_text':   {'type': 'text', 'analyzer': 'korean_analyzer'},
            'embedding_vector': {
                'type': 'knn_vector',
                'dimension': 384,
                'method': {
                    'engine': 'lucene',
                    'name': 'hnsw',
                    'space_type': 'cosinesimil',
                },
            },
        }
    },
}


def clean(val):
    text = str(val).strip()
    return text if text not in MISSING else ''


def make_doc_id(prefix, emp_id):
    # EMP0001 → BAS_0001
    num = emp_id.replace('EMP', '')
    return f'{prefix}_{num}'


def build_embedding_text(row, index_fields, common):
    # 이름/부서/직급은 모든 인덱스에 공통으로 포함 (문맥 파악용)
    common_keys = set()
    parts = []
    for key in ['이름', '부서', '직급']:
        val = common.get(key, '')
        if val:
            parts.append(f'{key}: {val}')
            common_keys.add(key)
    # 해당 인덱스 전용 필드 추가 (공통 키 중복 제외)
    for field in index_fields:
        if field in common_keys:
            continue
        val = clean(str(row.get(field, '')))
        if val:
            parts.append(f'{field}: {val}')
    return ' '.join(parts)


def create_indices(client):
    print('=== 인덱스 생성 ===')
    for name, config in INDEX_CONFIG.items():
        if client.indices.exists(index=name):
            client.indices.delete(index=name)
            print(f'  기존 삭제: {name}')
        client.indices.create(index=name, body=INDEX_BODY)
        print(f'  생성 완료: {name}  (security_level={config["security_level"]})')


def load_data(client, model):
    print('\n=== 데이터 적재 ===')

    # 같은 CSV를 여러 인덱스에서 읽을 때 중복 로딩 방지
    csv_cache = {}

    # 기본인사정보에서 이름/부서/직급 미리 조회 (역량성과·급여정보에서도 사용)
    basic_path = INPUT_DIR / '기본인사정보_정제.csv'
    basic_lookup = {}
    if basic_path.exists():
        basic_df = pd.read_csv(basic_path, encoding='utf-8-sig', dtype=object)
        csv_cache['기본인사정보_정제'] = basic_df
        for row in basic_df.to_dict('records'):
            emp_id = str(row.get('사원번호', '')).strip()
            basic_lookup[emp_id] = {
                '이름': clean(row.get('이름', '')),
                '부서': clean(row.get('부서', '')),
                '직급': clean(row.get('직급', '')),
            }
    else:
        print(f'  경고: 기본인사정보 파일 없음 ({basic_path})')

    for index_name, config in INDEX_CONFIG.items():
        source  = config['source']
        fields  = config['fields']
        prefix  = config['doc_prefix']

        # CSV 로딩 (캐시 활용)
        if source not in csv_cache:
            path = INPUT_DIR / f'{source}.csv'
            if not path.exists():
                print(f'  파일 없음: {path}  → {index_name} 건너뜀')
                continue
            csv_cache[source] = pd.read_csv(path, encoding='utf-8-sig', dtype=object)
        df = csv_cache[source]

        print(f'\n  [{index_name}]  총 {len(df):,}건 처리 중...')

        # 임베딩 텍스트 일괄 생성
        texts  = []
        metas  = []
        for row in df.to_dict('records'):
            emp_id = str(row.get('사원번호', '')).strip()
            common = basic_lookup.get(emp_id, {})
            text   = build_embedding_text(row, fields, common)
            if not text:
                continue
            texts.append(text)
            metas.append({
                'emp_id':     emp_id,
                'department': common.get('부서', ''),
                'position':   common.get('직급', ''),
            })

        # 실시간 임베딩 생성 (배치 64개씩)
        vectors = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

        # bulk API로 일괄 적재
        actions = []
        for i, meta in enumerate(metas):
            actions.append({
                '_index':  index_name,
                '_id':     make_doc_id(prefix, meta['emp_id']),
                '_source': {
                    'employee_id':      meta['emp_id'],
                    'department':       meta['department'],
                    'position':         meta['position'],
                    'embedding_text':   texts[i],
                    'embedding_vector': vectors[i].tolist(),
                },
            })

        success, failed = helpers.bulk(client, actions, raise_on_error=False)
        print(f'  완료: {success:,}건 성공  /  실패: {len(failed)}건')


def main():
    client = OpenSearch(
        hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=OPENSEARCH_USE_SSL,
        verify_certs=OPENSEARCH_VERIFY_CERTS,
        ssl_show_warn=False,
    )

    print(f'임베딩 모델 로딩: {EMBEDDING_MODEL}')
    model = SentenceTransformer(EMBEDDING_MODEL)

    create_indices(client)
    load_data(client, model)

    print('\n=== 전체 완료 ===')


if __name__ == '__main__':
    main()
