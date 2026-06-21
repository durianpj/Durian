# ══════════════════════════════════════════════════════════════════════════════
# 파이프라인 설정 · 상수
# ══════════════════════════════════════════════════════════════════════════════
# 경로, OpenSearch 연결 정보, 임베딩/KNN 설정, 인덱스 구성(INDEX_CONFIG),
# 그리고 각 단계가 공통으로 쓰는 상수를 한곳에 모았다.
import os
from pathlib import Path
from datetime import date
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent  # pipeline_modules/ 의 상위 = 프로젝트 루트
load_dotenv(BASE_DIR / '.env')

# 입력 · 로그 경로
DATASET_DIR    = BASE_DIR / 'data' / 'dataset'    # 원본 CSV 폴더
ERROR_LOG_PATH = BASE_DIR / 'data' / 'error.log'  # 1~4단계 에러 로그 (점검용)

# 사전 · 상태 파일 (config 폴더)
CONFIG_DIR          = BASE_DIR / 'config'
USER_DICT_FILE      = CONFIG_DIR / 'user_dictionary.txt'       # nori 토크나이저 사용자 사전
LAST_USER_DICT_FILE = CONFIG_DIR / 'last_user_dictionary.txt'  # 사전 변경 감지용 해시 저장

# OpenSearch 설치 위치
OPENSEARCH_HOME = Path(os.getenv('OPENSEARCH_HOME', str(BASE_DIR / 'opensearch-3.3.2'))).resolve()

# OpenSearch 연결 설정
OPENSEARCH_HOST         = os.getenv('OPENSEARCH_HOST', 'localhost')
OPENSEARCH_PORT         = int(os.getenv('OPENSEARCH_PORT', 9200))
OPENSEARCH_USER         = os.environ['OPENSEARCH_USER']
OPENSEARCH_PASSWORD     = os.environ['OPENSEARCH_PASSWORD']
OPENSEARCH_USE_SSL      = os.getenv('OPENSEARCH_USE_SSL', 'true').lower() == 'true'
OPENSEARCH_VERIFY_CERTS = os.getenv('OPENSEARCH_VERIFY_CERTS', 'false').lower() == 'true'

# 임베딩 모델
EMBEDDING_MODEL      = os.getenv('EMBED_MODEL_NAME', 'paraphrase-multilingual-MiniLM-L12-v2')
EMBEDDING_DIM        = int(os.getenv('EMBED_DIMENSION', '384'))
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBED_BATCH_SIZE', '64'))

# 청크당 최대 토큰 수
MAX_TOKENS = int(os.environ['MAX_TOKENS'])

# KNN 설정
KNN_ENGINE     = os.getenv('KNN_ENGINE',     'lucene')
KNN_METHOD     = os.getenv('KNN_METHOD',     'hnsw')
KNN_SPACE_TYPE = os.getenv('KNN_SPACE_TYPE', 'cosinesimil')

# 인덱스별 설정 (보안등급별로 어떤 필드가 들어가는지 정의)
INDEX_CONFIG = {
    'hr_basic_1': {
        'required_level': 1,
        'fields': [
            '이름', '성별', '나이', '입사일', '근속기간',
            '채용경로', '계약형태', '회사명', '사업장위치',
            '부서', '팀', '직책', '직급', '이메일',
        ],
    },
    'hr_basic_2': {
        'required_level': 2,
        'fields': [
            '생년월일', '병역', '학력', '출신대학', '학점',
            '전화번호', '이전직장명', '이전최종직급', '이전담당업무',
        ],
    },
    'hr_basic_3': {
        'required_level': 3,
        'fields': [
            '주민등록번호', '주소', '퇴직구분', '퇴직일자',
        ],
    },
    'hr_performance_2': {
        'required_level': 2,
        'fields': [
            '성과점수',
            '인사고과_2020', '인사고과_2021', '인사고과_2022',
            '인사고과_2023', '인사고과_2024',
            '자격증', 'TOEIC점수', '포상이력',
        ],
    },
    'hr_performance_3': {
        'required_level': 3,
        'fields': [
            '징계이력', '징계사유', '자격증수당여부',
        ],
    },
    'hr_salary_2': {
        'required_level': 2,
        'fields': [
            '잔업시간', '미사용휴가일수',
        ],
    },
    'hr_salary_3': {
        'required_level': 3,
        'fields': [
            '연봉', '급여은행', '계좌번호', '4대보험가입여부',
        ],
    },
}


# ── 1단계 전처리용 상수 ─────────────────────────────────────────────────────
TODAY = date.today()

MIN_AGE,          MAX_AGE          = 18, 80
MIN_SALARY,       MAX_SALARY       = 20_000_000, 500_000_000
MIN_OVERTIME,     MAX_OVERTIME     = 0, 52
MIN_UNUSED_LEAVE, MAX_UNUSED_LEAVE = 0, 30
MIN_GPA,          MAX_GPA          = 0.0, 4.5
MIN_SCORE,        MAX_SCORE        = 0, 100
MIN_TOEIC,        MAX_TOEIC        = 0, 990

DEPARTMENTS = ['개발부', '인사부', '영업부', '마케팅부', '기획부']

DEPT_TEAM_MAP = {
    '개발부':   ['백엔드팀', '프론트팀', 'AI팀', '인프라팀'],
    '인사부':   ['채용팀', '교육팀'],
    '영업부':   ['국내영업팀', '해외영업팀'],
    '마케팅부': ['디지털마케팅팀', '브랜드팀'],
    '기획부':   ['전략기획팀', '사업기획팀']
}

DEPT_LEVEL_MAP = {
    '개발부':   1,
    '인사부':   3,
    '영업부':   1,
    '마케팅부': 1,
    '기획부':   1
}

GRADE_LEVEL_MAP = {
    '사원': 1,
    '대리': 1,
    '과장': 1,
    '차장': 2,
    '부장': 2,
    '이사': 3,
    '사장': 3
}

POSITIONS        = ['팀원', '팀장', '본부장', '대표이사']
PERF_GRADES      = ['S', 'A', 'B', 'C', 'D', 'F']
INSURANCE_VALUES = ['가입', '미가입']
SUBSIDY_VALUES   = ['해당', '비해당']
PERF_YEARS       = [2020, 2021, 2022, 2023, 2024]


# ── 2단계 변환용 상수 ───────────────────────────────────────────────────────
# 아래 두 값은 '비어있다'고 판단할 문자열 목록이다.
# pandas가 빈 칸을 'nan'이나 'NaN'으로 읽어오고, 전처리 후에는 '미입력'으로 통일했으므로
# 이 목록에 해당하면 "값이 없는 것"으로 취급한다.
MISSING_VALUES = {'', '미입력', 'nan', 'NaN', 'None', 'none'}

# embedding_text에 포함할 필드 순서 목록.
# 이 순서대로 '필드명: 값' 한 줄씩 이어 붙여 문장을 만든다.
# 이름·부서·직급은 기본인사정보에서만 추가하므로 여기엔 포함하지 않는다.
EMBEDDING_FIELDS = [
    '성별', '나이', '생년월일', '주민등록번호', '병역',
    '입사일', '근속기간',
    '학력', '출신대학', '학점', '채용경로', '계약형태',
    '이전직장명', '이전최종직급', '이전담당업무',
    '회사명', '사업장위치',
    '팀', '직책',
    '퇴직구분', '퇴직일자',
    '이메일', '전화번호', '주소',
    '연봉', '잔업시간', '미사용휴가일수', '급여은행', '계좌번호', '4대보험가입여부',
    '성과점수',
    '인사고과_2020', '인사고과_2021', '인사고과_2022', '인사고과_2023', '인사고과_2024',
    '자격증', 'TOEIC점수', '자격증수당여부', '포상이력', '징계이력', '징계사유',
]


# ── 3단계 변경감지용 상수 ───────────────────────────────────────────────────
# 변경 감지 때 비교할 꼬리표(메타데이터) 필드 목록. (키, 한글이름, 정수레벨여부)
META_COMPARE_FIELDS = [
    ('employee_name',    '이름',     False),
    ('department',       '부서',     False),
    ('department_level', '부서레벨', True),
    ('job_grade',        '직급',     False),
    ('job_grade_level',  '직급레벨', True),
]
