import os
import json
import shutil
import hashlib
import zipfile
import urllib.request
from pathlib import Path
from datetime import date, datetime
import pandas as pd
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer

print(f'pandas 버전: {pd.__version__}')
print('라이브러리 로딩 완료!')

# ══════════════════════════════════════════════════════════════════════════════
# 경로 · 설정
# ══════════════════════════════════════════════════════════════════════════════
# 이 파일은 3단계(전처리 → JSONL 변환 → 인덱싱)를 하나로 합친 파이프라인이다.
# 단계 사이의 데이터는 파일이 아니라 메모리(변수)로 직접 넘긴다.
# 변경 감지(증분 적재)는 OpenSearch에 이미 적재된 값과 비교해서 판단한다.
#
# 청킹은 별도 단계가 아니라 인덱싱 단계 안에서 인덱스별로 수행한다.
# (전체 필드를 한꺼번에 청킹하면 인덱스에 안 들어갈 필드까지 토큰을 차지해
#  같은 인덱스의 필드가 여러 청크에 흩어진다. 인덱스별로 필요한 필드만 골라
#  청킹해야 한 인덱스 안의 필드가 같은 청크에 모인다.)

BASE_DIR = Path(__file__).resolve().parent
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



# ══════════════════════════════════════════════════════════════════════════════
# 1단계: 전처리
# ══════════════════════════════════════════════════════════════════════════════
# 원본 CSV를 읽어 컬럼별로 검증·교정하고, 문제 있는 행은 제거한다.
# 결과는 정제된 DataFrame들(메모리)로 반환하고, 어떤 값이 왜 바뀌었는지는 error.log에 남긴다.

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

# 전처리 중 파일 하나를 처리할 때마다 새로 비워서 쓰는 작업용 저장소
_errors    = []   # 에러 로그 모음
drop_rows  = set()  # 제거할 행 번호 모음
valid_rrn  = {}   # 주민번호에서 뽑은 정상 생년월일 (행 번호 -> date)
valid_hire = {}   # 정상 입사일 (행 번호 -> date)


def log(row, emp_id, column, original_value, reason):
    # 문제가 발생한 행·사원번호·컬럼·원본값·사유를 _errors에 모아둔다.
    # 즉시 파일에 쓰지 않는 이유: 단계가 끝난 뒤 all_errors로 합쳐서
    # write_error_log 한 곳에서 한 번에 기록하기 위해서다.
    _errors.append({
        '행': row,
        '사원번호': emp_id,
        '컬럼': column,
        '원본값': original_value,
        '사유': reason
    })


def calc_age(birth):
    # 한국 나이 계산 방식: 올해 연도 - 출생 연도 + 1
    return TODAY.year - birth.year + 1


def tenure_years(hire_date):
    # 근속기간(년): 올해 연도 - 입사 연도 (만 나이 방식이 아닌 연도 차이)
    return TODAY.year - hire_date.year


def parse_date(text):
    # 날짜 문자열을 date 객체로 변환한다. 파싱 불가하면 None을 반환.
    # pandas to_datetime이 '2023-01-01', '2023/01/01', '20230101' 등 다양한 형식을 처리해준다.
    try:
        text = str(text).strip()
        if not text or text in ('nan', 'NaN', 'None', '미입력'):
            return None
        return pd.to_datetime(text).date()
    except Exception:
        return None


def birth_from_rrn(rrn):
    # 주민등록번호 앞 6자리(YYMMDD)와 7번째 자리(세기 구분자)로 생년월일을 복원한다.
    # 7번째 자리: 1·2 → 1900년대 출생, 3·4 → 2000년대 출생
    # 그 외(5~9, 외국인 등록번호 등)는 지원하지 않으므로 None 반환.
    try:
        digits = rrn.replace('-', '')
        yy = int(digits[0:2])
        mm = int(digits[2:4])
        dd = int(digits[4:6])
        seven = int(digits[6])
        if seven in (1, 2):
            year = 1900 + yy
        elif seven in (3, 4):
            year = 2000 + yy
        else:
            return None
        return date(year, mm, dd)
    except Exception:
        return None


def parse_array(cell_value):
    # 쉼표로 구분된 문자열(예: '정보처리기사,SQLD')을 리스트로 분리한다.
    # 자격증·포상이력·징계이력처럼 여러 값이 한 칸에 들어오는 컬럼에 사용한다.
    if not cell_value or str(cell_value).strip() in ('nan', 'NaN', 'None'):
        return []
    items = []
    for item in str(cell_value).split(','):
        item = item.strip()
        if item:
            items.append(item)
    return items


def is_valid_emp_id(emp_id):
    # 사원번호 형식: 'EMP' + 숫자 4자리 = 총 7자리 (예: EMP0001)
    # 길이·접두사·숫자 여부를 순서대로 검사해 하나라도 어긋나면 False
    if len(emp_id) != 7:
        return False
    if emp_id[:3] != 'EMP':
        return False
    if not emp_id[3:].isdigit():
        return False
    return True


def is_valid_name(name):
    # 이름은 한글·영문 알파벳만 허용한다.
    # 숫자나 특수문자가 섞여 있으면 오입력(예: 시스템 오류로 코드가 들어온 경우)으로 판단해 행을 제거한다.
    for char in name:
        if not char.isalpha():
            return False
    return True


def is_valid_rrn_format(rrn):
    # 주민등록번호 형식: 숫자6자리 + '-' + 숫자7자리 = 총 14자리
    # 하이픈 위치(인덱스 6)와 앞뒤 숫자 여부를 직접 확인한다.
    if len(rrn) != 14:
        return False
    if rrn[6] != '-':
        return False
    if not rrn[:6].isdigit():
        return False
    if not rrn[7:].isdigit():
        return False
    return True


def is_valid_email(email):
    # '@' 가 정확히 1개인지, 로컬 파트가 비어있지 않은지,
    # 도메인에 점이 있는지, 도메인이 점으로 끝나지 않는지를 검사한다.
    # 라이브러리 없이 기초 검사만 하는 이유: 정규식 금지 정책 때문.
    if email.count('@') != 1:
        return False
    parts = email.split('@')
    local = parts[0]
    domain = parts[1]
    if not local:
        return False
    if '.' not in domain:
        return False
    if domain.endswith('.'):
        return False
    return True


def is_valid_phone(phone):
    # 전화번호 형식: '0'으로 시작 + 하이픈 2개로 3파트 분리 + 각 파트 숫자만
    # 예: 010-1234-5678, 02-123-4567 모두 통과
    if not phone.startswith('0'):
        return False
    if phone.count('-') != 2:
        return False
    parts = phone.split('-')
    for part in parts:
        if not part.isdigit():
            return False
    return True


def is_valid_bank(bank):
    # 은행명은 한글·영문·공백만 허용한다.
    # 숫자나 특수문자가 있으면 계좌번호가 잘못 들어온 경우일 가능성이 높다.
    for char in bank:
        if not char.isalpha() and char != ' ':
            return False
    return True


def validate_empid(df):
    # 사원번호가 없거나 형식이 틀린 행은 행 전체를 제거한다.
    # 사원번호가 없으면 이후 모든 필드의 '사원번호' 참조가 무의미해지기 때문이다.
    # 중복 발견 시 먼저 나온 행(이전 행)을 제거하고 나중 행을 유지한다.
    # → 최신 데이터가 CSV 뒤쪽에 추가되는 구조를 가정한다.
    if '사원번호' not in df.columns:
        return
    seen_emp = {}
    for row, raw in df['사원번호'].items():
        if raw:
            emp_id = str(raw).strip()
        else:
            emp_id = ''
        if not emp_id or emp_id in ('nan', 'NaN'):
            log(row, emp_id, '사원번호', emp_id, '결측')
            drop_rows.add(row)
        elif not is_valid_emp_id(emp_id):
            log(row, emp_id, '사원번호', emp_id, '형식 오류 (EMP+4자리)')
            drop_rows.add(row)
        elif emp_id in seen_emp:
            log(row, emp_id, '사원번호', emp_id, '중복 → 이전 행 제거')
            drop_rows.add(seen_emp[emp_id])
            seen_emp[emp_id] = row
        else:
            seen_emp[emp_id] = row
        df.at[row, '사원번호'] = emp_id


def validate_name(df):
    # 이름이 없거나 숫자·특수문자가 섞인 행은 행 전체를 제거한다.
    # 이름은 검색 결과의 직원 식별에 사용되므로, 오입력이 있으면 결과 신뢰도가 떨어진다.
    if '이름' not in df.columns:
        return
    for row, raw in df['이름'].items():
        name = str(raw).strip() if raw else ''
        emp_id = df.at[row, '사원번호']
        if not name or name in ('nan', 'NaN'):
            log(row, emp_id, '이름', name, '결측')
            drop_rows.add(row)
        elif not is_valid_name(name):
            log(row, emp_id, '이름', name, '숫자/특수문자 포함')
            drop_rows.add(row)


def validate_rrn(df):
    # 주민번호 검증과 함께, 파싱에 성공한 생년월일을 valid_rrn에 저장해둔다.
    # valid_rrn은 이후 validate_gender·validate_birth·validate_age에서
    # 생년월일/성별을 교정하는 데 사용한다 (주민번호가 가장 신뢰할 수 있는 출처이기 때문).
    # 결측이거나 파싱 실패인 경우에는 행을 제거하지 않고 '미입력'으로 처리한다.
    # → 주민번호 없이도 다른 인사 정보는 충분히 의미가 있어서 행을 살린다.
    if '주민등록번호' not in df.columns:
        return
    for row, raw in df['주민등록번호'].items():
        rrn = str(raw).strip() if raw else ''
        emp_id = df.at[row, '사원번호']
        if not rrn or rrn in ('nan', 'NaN'):
            log(row, emp_id, '주민등록번호', rrn, '결측')
            df.at[row, '주민등록번호'] = '미입력'
            continue
        if not is_valid_rrn_format(rrn):
            log(row, emp_id, '주민등록번호', rrn, '형식 오류')
            df.at[row, '주민등록번호'] = '미입력'
            continue
        birth = birth_from_rrn(rrn)
        if birth is None:
            log(row, emp_id, '주민등록번호', rrn, '날짜 파싱 불가')
            df.at[row, '주민등록번호'] = '미입력'
            continue
        if not (MIN_AGE <= calc_age(birth) <= MAX_AGE):
            log(row, emp_id, '주민등록번호', rrn, f'생년월일 범위 초과 (나이={calc_age(birth)})')
            df.at[row, '주민등록번호'] = '미입력'
            continue
        # 여기까지 통과하면 주민번호에서 생년월일을 확실하게 뽑을 수 있다는 의미
        valid_rrn[row] = birth


def validate_gender(df):
    # 주민번호 7번째 자리로 성별을 검증·교정한다.
    # 7번째 자리가 홀수(1·3)면 남성, 짝수(2·4)면 여성.
    # 입력된 성별과 다를 경우 주민번호가 더 신뢰할 수 있는 출처이므로 주민번호로 교정한다.
    # 주민번호가 없으면 강제 교정이 불가능하므로 '미입력'으로 남긴다.
    if '성별' not in df.columns:
        return
    for row in df.index:
        rrn_str = ''
        if '주민등록번호' in df.columns:
            rrn_str = str(df.at[row, '주민등록번호']).replace('-', '')
        current_gender = str(df.at[row, '성별']).strip() if df.at[row, '성별'] else ''
        if rrn_str and rrn_str not in ('미입력', 'nan', 'NaN') and len(rrn_str) >= 7 and rrn_str.isdigit():
            correct_gender = '남' if int(rrn_str[6]) in (1, 3) else '여'
            if current_gender != correct_gender:
                log(row, df.at[row, '사원번호'], '성별', current_gender, f'주민번호와 불일치 → {correct_gender} 교정')
                df.at[row, '성별'] = correct_gender
        elif current_gender not in ('남', '여'):
            log(row, df.at[row, '사원번호'], '성별', current_gender, '결측/이상치 + 주민번호 없음')
            df.at[row, '성별'] = '미입력'


def validate_birth(df):
    # 생년월일 교정 우선순위:
    #   1순위) 주민번호에서 파싱한 날짜(valid_rrn) → 가장 신뢰할 수 있으므로 무조건 사용
    #   2순위) 직접 입력된 생년월일 → 범위(만 18~80세) 안에 있으면 사용
    # 주민번호가 있으면 직접 입력값이 범위를 벗어나도 교정 로그만 남기고 주민번호 값을 쓴다.
    # 주민번호가 없고 직접 입력값도 없거나 범위 초과이면 '미입력'.
    if '생년월일' not in df.columns:
        return
    for row in df.index:
        rrn_birth = valid_rrn.get(row)
        birth_str = str(df.at[row, '생년월일']).strip() if df.at[row, '생년월일'] else ''
        if birth_str in ('nan', 'NaN'):
            birth_str = ''
        birth = parse_date(birth_str)
        if rrn_birth:
            # 주민번호에서 뽑은 날짜가 있으면 그것을 기준으로 삼는다
            if birth is None or not (MIN_AGE <= calc_age(birth) <= MAX_AGE):
                if birth is not None:
                    log(row, df.at[row, '사원번호'], '생년월일', birth_str, '범위 초과 → 주민번호로 교정')
                df.at[row, '생년월일'] = rrn_birth.strftime('%Y-%m-%d')
            else:
                df.at[row, '생년월일'] = birth.strftime('%Y-%m-%d')
        else:
            # 주민번호가 없으니 직접 입력값만 믿을 수 있다
            if birth is None:
                log(row, df.at[row, '사원번호'], '생년월일', birth_str, '결측/파싱불가 + 주민번호 없음')
                df.at[row, '생년월일'] = '미입력'
            elif not (MIN_AGE <= calc_age(birth) <= MAX_AGE):
                log(row, df.at[row, '사원번호'], '생년월일', birth_str, '범위 초과')
                df.at[row, '생년월일'] = '미입력'
            else:
                df.at[row, '생년월일'] = birth.strftime('%Y-%m-%d')


def validate_age(df):
    # 나이는 입력값을 그대로 쓰지 않고 생년월일에서 재계산한다.
    # 입력 나이는 작성 시점 기준이라 파이프라인 실행 시점과 달라질 수 있기 때문이다.
    # 재계산 우선순위: 1) 교정된 생년월일 컬럼 → 2) valid_rrn (주민번호에서 뽑은 날짜)
    # 둘 다 없으면 나이를 신뢰할 수 없으므로 '미입력'.
    if '나이' not in df.columns:
        return
    for row in df.index:
        birth_str = str(df.at[row, '생년월일']) if '생년월일' in df.columns else '미입력'
        if birth_str in ('nan', 'NaN'):
            birth_str = '미입력'
        birth = parse_date(birth_str) if birth_str != '미입력' else None
        if birth:
            df.at[row, '나이'] = calc_age(birth)
        elif valid_rrn.get(row):
            # 생년월일 컬럼이 '미입력'이지만 주민번호에서는 날짜를 뽑은 경우
            df.at[row, '나이'] = calc_age(valid_rrn[row])
        else:
            log(row, df.at[row, '사원번호'], '나이', df.at[row, '나이'], '생년월일·주민번호 없음')
            df.at[row, '나이'] = '미입력'


def validate_military(df):
    # 여성은 병역의무가 없으므로 성별이 '여'이면 무조건 '해당없음'으로 덮어쓴다.
    # 남성이거나 성별을 모르는데 병역 값이 비어있으면 '미입력'.
    # 단, 남성인데 병역 값이 있으면(예: '복무완료', '면제') 건드리지 않는다.
    if '병역' not in df.columns:
        return
    for row in df.index:
        gender = str(df.at[row, '성별']).strip() if '성별' in df.columns else ''
        military = str(df.at[row, '병역']).strip() if df.at[row, '병역'] else ''
        if military in ('nan', 'NaN'):
            military = ''
        if gender == '여':
            df.at[row, '병역'] = '해당없음'
        elif not military:
            log(row, df.at[row, '사원번호'], '병역', military, '결측')
            df.at[row, '병역'] = '미입력'


def validate_hire(df):
    # 입사일 유효성 검사 기준:
    #   1) 파싱이 가능해야 한다 (날짜 형식 오류 제외)
    #   2) 미래 날짜이면 안 된다 (오입력)
    #   3) 생년월일 + MIN_AGE(18년) 이전 날짜이면 안 된다 (만 18세 미만 취업 불가)
    # 유효한 입사일은 valid_hire에 저장해 validate_tenure·validate_retire에서 활용한다.
    if '입사일' not in df.columns:
        return
    for row, raw in df['입사일'].items():
        hire_str = str(raw).strip() if raw else ''
        if hire_str in ('nan', 'NaN'):
            hire_str = ''
        hire_date = parse_date(hire_str)
        emp_id = df.at[row, '사원번호']
        birth_str = str(df.at[row, '생년월일']) if '생년월일' in df.columns else '미입력'
        if birth_str in ('nan', 'NaN'):
            birth_str = '미입력'
        birth = parse_date(birth_str) if birth_str != '미입력' else None
        # 생년월일 기준 최조 입사 가능일: 생일과 같은 월·일로 MIN_AGE년 후
        earliest_hire = date(birth.year + MIN_AGE, birth.month, birth.day) if birth else None
        if hire_date is None:
            log(row, emp_id, '입사일', hire_str, '결측/파싱불가')
            df.at[row, '입사일'] = '미입력'
        elif hire_date > TODAY:
            log(row, emp_id, '입사일', hire_str, '현재 날짜 초과')
            df.at[row, '입사일'] = '미입력'
        elif earliest_hire and hire_date < earliest_hire:
            log(row, emp_id, '입사일', hire_str, '만 18세 이전')
            df.at[row, '입사일'] = '미입력'
        else:
            valid_hire[row] = hire_date
            df.at[row, '입사일'] = hire_date.strftime('%Y-%m-%d')


def validate_tenure(df):
    # 근속기간도 나이와 마찬가지로 입력값을 믿지 않고 입사일에서 재계산한다.
    # 입사일이 없으면(valid_hire에 없으면) 계산 불가 → '미입력'.
    if '근속기간' not in df.columns:
        return
    for row in df.index:
        hire_date = valid_hire.get(row)
        if hire_date:
            df.at[row, '근속기간'] = tenure_years(hire_date)
        else:
            log(row, df.at[row, '사원번호'], '근속기간', df.at[row, '근속기간'], '입사일 없어 계산 불가')
            df.at[row, '근속기간'] = '미입력'


def validate_edu(df):
    # 학력과 출신대학은 서로 일관성이 있어야 한다:
    #   - 학력이 없는데 출신대학이 있으면 → 대학교명에 '전문'이 있으면 전문대졸, 없으면 대졸로 역추론
    #   - 학력이 '고졸'인데 출신대학이 있으면 → 출신대학을 지운다 (모순)
    #   - 학력이 대졸/전문대졸/대학원졸인데 출신대학이 없으면 → '미입력'
    #   - 학력='전문대졸'인데 대학교명에 '전문'이 없으면 → '미입력' (불일치)
    #   - 학력='대졸'인데 대학교명에 '전문'이 있으면 → '미입력' (불일치)
    if '학력' not in df.columns:
        return
    for row in df.index:
        edu = str(df.at[row, '학력']).strip() if df.at[row, '학력'] else ''
        if edu in ('nan', 'NaN'):
            edu = ''
        univ = str(df.at[row, '출신대학']).strip() if '출신대학' in df.columns and df.at[row, '출신대학'] else ''
        if univ in ('nan', 'NaN'):
            univ = ''
        emp_id = df.at[row, '사원번호']
        if not edu or edu == '미입력':
            # 학력은 없지만 출신대학이 있으면 학교명으로 학력을 역추론한다
            if univ and univ != '미입력':
                edu = '전문대졸' if '전문' in univ else '대졸'
                df.at[row, '학력'] = edu
            else:
                log(row, emp_id, '학력', edu, '결측 + 출신대학 없음')
                df.at[row, '학력'] = '미입력'
                edu = '미입력'
        if edu == '고졸':
            # 고졸인데 대학교명이 있으면 입력 오류 → 출신대학 삭제
            if univ:
                log(row, emp_id, '출신대학', univ, '고졸인데 출신대학 존재 → 삭제')
                df.at[row, '출신대학'] = ''
        elif edu in ('대졸', '전문대졸', '대학원졸') and '출신대학' in df.columns:
            if not univ:
                log(row, emp_id, '출신대학', univ, '결측')
                df.at[row, '출신대학'] = '미입력'
            elif edu == '전문대졸' and '전문' not in univ:
                # 학력은 전문대졸인데 일반대학 이름이 들어온 경우
                log(row, emp_id, '출신대학', univ, '학력=전문대졸인데 일반대학교명')
                df.at[row, '출신대학'] = '미입력'
            elif edu == '대졸' and '전문' in univ:
                # 학력은 대졸인데 전문대 이름이 들어온 경우
                log(row, emp_id, '출신대학', univ, '학력=대졸인데 전문대학교명')
                df.at[row, '출신대학'] = '미입력'


def validate_career(df):
    # 학점: 고졸이면 대학 학점 자체가 없으므로 빈 값으로 처리한다.
    # 학점: 대졸 이상인데 숫자로 변환 불가하거나 범위(0.0~4.5) 초과이면 '미입력'.
    # 채용경로·계약형태: 결측이면 '미입력' (검색 필터 기준이 되므로 값이 있어야 한다).
    # 이전직장 관련 3필드(이전직장명·이전최종직급·이전담당업무): 결측이면 '미입력'.
    # → 경력직이든 신입이든 구분 없이 비어있으면 모두 '미입력'으로 통일한다.
    if '학점' in df.columns:
        for row in df.index:
            edu = str(df.at[row, '학력']).strip() if '학력' in df.columns else ''
            gpa_raw = df.at[row, '학점']
            if edu == '고졸':
                # 고졸은 대학 학점이 없으므로 빈 값으로 설정
                df.at[row, '학점'] = ''
                continue
            try:
                gpa = float(str(gpa_raw).strip())
            except Exception:
                gpa = None
                log(row, df.at[row, '사원번호'], '학점', gpa_raw, '숫자 변환 불가')
            if gpa is None:
                df.at[row, '학점'] = '미입력'
            elif not (MIN_GPA <= gpa <= MAX_GPA):
                log(row, df.at[row, '사원번호'], '학점', gpa_raw, f'범위 초과 ({MIN_GPA}~{MAX_GPA})')
                df.at[row, '학점'] = '미입력'
    for col in ['채용경로', '계약형태']:
        if col not in df.columns:
            continue
        for row, raw in df[col].items():
            cell_value = str(raw).strip() if raw else ''
            if cell_value in ('nan', 'NaN'):
                cell_value = ''
            if not cell_value:
                log(row, df.at[row, '사원번호'], col, cell_value, '결측')
                df.at[row, col] = '미입력'
    if '채용경로' in df.columns:
        for row in df.index:
            hire_route = str(df.at[row, '채용경로']).strip()
            # 이전직장 3필드는 채용경로에 상관없이 비어있으면 모두 '미입력'
            for col in ['이전직장명', '이전최종직급', '이전담당업무']:
                if col not in df.columns:
                    continue
                cell_value = str(df.at[row, col]).strip() if df.at[row, col] else ''
                if cell_value in ('nan', 'NaN'):
                    cell_value = ''
                if not cell_value:
                    df.at[row, col] = '미입력'


def validate_dept(df):
    # 부서는 인덱스 라우팅(어느 인덱스에 어느 직원이 들어가는지)과
    # 검색 권한 필터(부서레벨)의 핵심 키이므로, 허용 목록에 없으면 행 전체를 제거한다.
    # 팀은 부서에 종속되므로 DEPT_TEAM_MAP으로 부서-팀 매핑을 검증한다.
    # 단, 사장·이사(임원급)는 특정 팀에 속하지 않을 수 있으므로 팀 검증을 건너뛴다.
    for col in ['회사명', '사업장위치']:
        if col not in df.columns:
            continue
        for row, raw in df[col].items():
            cell_value = str(raw).strip() if raw else ''
            if cell_value in ('nan', 'NaN'):
                cell_value = ''
            if not cell_value:
                log(row, df.at[row, '사원번호'], col, cell_value, '결측')
                df.at[row, col] = '미입력'
    if '부서' not in df.columns:
        return
    for row, raw in df['부서'].items():
        dept = str(raw).strip() if raw else ''
        if dept in ('nan', 'NaN'):
            dept = ''
        if not dept or dept not in DEPARTMENTS:
            log(row, df.at[row, '사원번호'], '부서', dept, '결측/이상치 → 행 제거')
            drop_rows.add(row)
    if '팀' in df.columns:
        for row in df.index:
            grade = str(df.at[row, '직급']).strip() if '직급' in df.columns and df.at[row, '직급'] else ''
            # 임원급(사장·이사)은 팀 배정이 없어도 되므로 건너뛴다
            if grade in ('사장', '이사'):
                continue
            dept = str(df.at[row, '부서']).strip()
            team = str(df.at[row, '팀']).strip() if df.at[row, '팀'] else ''
            if team in ('nan', 'NaN'):
                team = ''
            if not team:
                log(row, df.at[row, '사원번호'], '팀', team, '결측')
                df.at[row, '팀'] = '미입력'
            elif dept in DEPT_TEAM_MAP and team not in DEPT_TEAM_MAP[dept]:
                # 부서에 속할 수 없는 팀 이름이 들어온 경우
                log(row, df.at[row, '사원번호'], '팀', team, f'부서({dept})-팀 매핑 불일치')
                df.at[row, '팀'] = '미입력'
    if '부서레벨' in df.columns:
        # 부서레벨은 검색 권한 필터에 사용한다. DEPT_LEVEL_MAP에서 자동으로 채운다.
        for row in df.index:
            dept = str(df.at[row, '부서']).strip()
            if dept in DEPT_LEVEL_MAP:
                df.at[row, '부서레벨'] = DEPT_LEVEL_MAP[dept]
            else:
                log(row, df.at[row, '사원번호'], '부서레벨', df.at[row, '부서레벨'], '부서 없음 → 행 제거')
                drop_rows.add(row)


def validate_grade(df):
    # 직급도 부서와 마찬가지로 권한 필터의 핵심 키이므로, 허용 목록에 없으면 행 전체를 제거한다.
    # 직책 규칙:
    #   - 직급이 '사장'이면 직책은 반드시 '대표이사'여야 한다 (불일치 시 교정).
    #   - '대표이사'는 전체 CSV에서 한 명만 허용 (ceo_row로 첫 번째 행을 기록하고, 이후 발견 시 '미입력').
    # 직급레벨은 검색 권한 필터에 사용한다. GRADE_LEVEL_MAP에서 자동으로 채운다.
    if '직급' not in df.columns:
        return
    for row, raw in df['직급'].items():
        grade = str(raw).strip() if raw else ''
        if grade in ('nan', 'NaN'):
            grade = ''
        if not grade or grade not in GRADE_LEVEL_MAP:
            log(row, df.at[row, '사원번호'], '직급', grade, '결측/이상치 → 행 제거')
            drop_rows.add(row)
    if '직책' in df.columns:
        ceo_row = None   # 대표이사를 처음 발견한 행 번호 (중복 방지용)
        for row in df.index:
            grade = str(df.at[row, '직급']).strip()
            position = str(df.at[row, '직책']).strip() if df.at[row, '직책'] else ''
            if position in ('nan', 'NaN'):
                position = ''
            if not position or position not in POSITIONS:
                log(row, df.at[row, '사원번호'], '직책', position, '결측/이상치')
                df.at[row, '직책'] = '미입력'
                continue
            if grade == '사장' and position != '대표이사':
                # 직급이 사장이면 직책은 반드시 대표이사여야 한다
                log(row, df.at[row, '사원번호'], '직책', position, '직급=사장인데 직책≠대표이사 → 교정')
                df.at[row, '직책'] = '대표이사'
                position = '대표이사'
            if position == '대표이사':
                if ceo_row is None:
                    ceo_row = row   # 처음 만난 대표이사 행을 기록
                else:
                    # 두 번째 이후 대표이사는 중복이므로 '미입력'으로 처리
                    log(row, df.at[row, '사원번호'], '직책', position, '대표이사 중복 (1명만 허용)')
                    df.at[row, '직책'] = '미입력'
    if '직급레벨' in df.columns:
        for row in df.index:
            grade = str(df.at[row, '직급']).strip()
            if grade in GRADE_LEVEL_MAP:
                df.at[row, '직급레벨'] = GRADE_LEVEL_MAP[grade]
            else:
                log(row, df.at[row, '사원번호'], '직급레벨', df.at[row, '직급레벨'], '직급 없음 → 행 제거')
                drop_rows.add(row)


def validate_retire(df):
    # 퇴직구분과 퇴직일자는 짝으로 존재해야 한다.
    #   - 퇴직일자는 있는데 퇴직구분이 없으면 → 퇴직구분을 '미입력'으로 기록하고 로그
    #   - 퇴직구분은 있는데 퇴직일자가 없으면 → 퇴직일자를 '미입력'으로 기록하고 로그
    #   - 둘 다 없으면 → 현직 직원으로 간주하고 둘 다 '미입력' (조용히 처리)
    # 날짜 범위 규칙:
    #   - 퇴직일자는 입사일보다 이전일 수 없다
    #   - 퇴직일자는 현재 날짜를 초과할 수 없다 (미래 퇴직은 사전 입력 불가)
    if '퇴직구분' not in df.columns:
        return
    for row in df.index:
        retire_type = str(df.at[row, '퇴직구분']).strip() if df.at[row, '퇴직구분'] else ''
        if retire_type in ('nan', 'NaN'):
            retire_type = ''
        retire_date_str = ''
        if '퇴직일자' in df.columns and df.at[row, '퇴직일자']:
            retire_date_str = str(df.at[row, '퇴직일자']).strip()
            if retire_date_str in ('nan', 'NaN'):
                retire_date_str = ''
        emp_id = df.at[row, '사원번호']
        hire_date  = valid_hire.get(row)
        retire_date = parse_date(retire_date_str)
        if retire_date_str and retire_date is None:
            log(row, emp_id, '퇴직일자', retire_date_str, '날짜 파싱 불가')
            df.at[row, '퇴직일자'] = '미입력'
            retire_date_str = ''
        if retire_date:
            if hire_date and retire_date < hire_date:
                # 입사 전에 퇴직하는 것은 논리적으로 불가
                log(row, emp_id, '퇴직일자', retire_date_str, '입사일 이전')
                df.at[row, '퇴직일자'] = '미입력'
                retire_date = None
                retire_date_str = ''
            elif retire_date > TODAY:
                log(row, emp_id, '퇴직일자', retire_date_str, '현재 날짜 초과')
                df.at[row, '퇴직일자'] = '미입력'
                retire_date = None
                retire_date_str = ''
        # 퇴직구분·퇴직일자 짝 검사
        if retire_date and not retire_type:
            log(row, emp_id, '퇴직구분', retire_type, '퇴직일자 있는데 퇴직구분 없음')
            df.at[row, '퇴직구분'] = '미입력'
        elif not retire_type:
            df.at[row, '퇴직구분'] = '미입력'
        if retire_type and not retire_date_str and '퇴직일자' in df.columns:
            log(row, emp_id, '퇴직일자', retire_date_str, '퇴직구분 있는데 퇴직일자 없음')
            df.at[row, '퇴직일자'] = '미입력'
        elif not retire_date_str and '퇴직일자' in df.columns:
            df.at[row, '퇴직일자'] = '미입력'


def validate_contact(df):
    # 이메일 중복 체크: 같은 이메일을 서로 다른 사원번호가 쓰면 안 된다.
    # seen_email은 {이메일: 사원번호}로 관리하며, 같은 사원번호가 두 번 나오면 중복이 아니다.
    # 전화번호: '0'으로 시작하고 하이픈 2개로 3파트 구조인지 확인한다 (is_valid_phone 참고).
    # 주소: 결측이면 '미입력' (형식 검증은 하지 않는다 — 주소 형식이 너무 다양하기 때문).
    seen_email = {}
    if '이메일' in df.columns:
        for row, raw in df['이메일'].items():
            email = str(raw).strip() if raw else ''
            if email in ('nan', 'NaN'):
                email = ''
            emp_id = df.at[row, '사원번호']
            if not email:
                log(row, emp_id, '이메일', email, '결측')
                df.at[row, '이메일'] = '미입력'
            elif not is_valid_email(email):
                log(row, emp_id, '이메일', email, '형식 오류')
                df.at[row, '이메일'] = '미입력'
            elif email in seen_email and seen_email[email] != emp_id:
                # 서로 다른 사원번호가 같은 이메일을 공유하는 경우 → 중복
                log(row, emp_id, '이메일', email, '중복')
                df.at[row, '이메일'] = '미입력'
            else:
                seen_email[email] = emp_id
    if '전화번호' in df.columns:
        for row, raw in df['전화번호'].items():
            phone = str(raw).strip() if raw else ''
            if phone in ('nan', 'NaN'):
                phone = ''
            if not phone:
                log(row, df.at[row, '사원번호'], '전화번호', phone, '결측')
                df.at[row, '전화번호'] = '미입력'
            elif not is_valid_phone(phone):
                log(row, df.at[row, '사원번호'], '전화번호', phone, '형식 오류')
                df.at[row, '전화번호'] = '미입력'
    if '주소' in df.columns:
        for row, raw in df['주소'].items():
            address = str(raw).strip() if raw else ''
            if address in ('nan', 'NaN'):
                address = ''
            if not address:
                log(row, df.at[row, '사원번호'], '주소', address, '결측')
                df.at[row, '주소'] = '미입력'


def validate_salary(df):
    # 연봉·잔업시간·미사용휴가일수는 범위 기반으로 이상치를 판단한다.
    #   - 연봉: 2천만~5억 원 (MIN_SALARY~MAX_SALARY) — 2024년 기준 최저~최고 급여 범위
    #   - 잔업시간: 0~52시간/월 (MIN_OVERTIME~MAX_OVERTIME) — 법정 최대 초과근무 기준
    #   - 미사용휴가: 0~30일 (MIN_UNUSED_LEAVE~MAX_UNUSED_LEAVE)
    # float()을 거친 뒤 int()로 변환하는 이유: CSV에서 '2500000.0'처럼 소수점이 붙어오는 경우 처리.
    # 급여은행: 한글·영문·공백만 허용 (숫자·특수문자가 있으면 계좌번호가 잘못 들어온 것으로 판단).
    # 계좌번호: 형식 검증 없이 결측 여부만 확인 (은행마다 형식이 달라 통일 불가).
    # 4대보험가입여부: '가입'/'미가입' 두 값만 허용.
    for col, min_val, max_val in [
        ('연봉',          MIN_SALARY,       MAX_SALARY),
        ('잔업시간',       MIN_OVERTIME,     MAX_OVERTIME),
        ('미사용휴가일수', MIN_UNUSED_LEAVE, MAX_UNUSED_LEAVE),
    ]:
        if col not in df.columns:
            continue
        for row, raw in df[col].items():
            try:
                # CSV에서 '3000000.0'처럼 소수점이 붙어올 수 있으므로 float → int 순서로 변환
                val = int(float(raw))
            except Exception:
                log(row, df.at[row, '사원번호'], col, raw, '숫자 변환 불가')
                df.at[row, col] = '미입력'
                continue
            if not (min_val <= val <= max_val):
                log(row, df.at[row, '사원번호'], col, raw, f'범위 초과 ({min_val:,}~{max_val:,})')
                df.at[row, col] = '미입력'
    if '급여은행' in df.columns:
        for row, raw in df['급여은행'].items():
            bank = str(raw).strip() if raw else ''
            if bank in ('nan', 'NaN'):
                bank = ''
            if not bank:
                log(row, df.at[row, '사원번호'], '급여은행', bank, '결측')
                df.at[row, '급여은행'] = '미입력'
            elif not is_valid_bank(bank):
                log(row, df.at[row, '사원번호'], '급여은행', bank, '숫자/특수문자 포함')
                df.at[row, '급여은행'] = '미입력'
    if '계좌번호' in df.columns:
        for row, raw in df['계좌번호'].items():
            account = str(raw).strip() if raw else ''
            if account in ('nan', 'NaN'):
                account = ''
            if not account:
                log(row, df.at[row, '사원번호'], '계좌번호', account, '결측')
                df.at[row, '계좌번호'] = '미입력'
    if '4대보험가입여부' in df.columns:
        for row, raw in df['4대보험가입여부'].items():
            insurance = str(raw).strip() if raw else ''
            if insurance in ('nan', 'NaN'):
                insurance = ''
            if not insurance or insurance not in INSURANCE_VALUES:
                log(row, df.at[row, '사원번호'], '4대보험가입여부', insurance, '결측/이상치')
                df.at[row, '4대보험가입여부'] = '미입력'


def validate_perf(df):
    # 성과점수: 0~100점 정수. 문자열이나 소수점이 들어오면 변환 시도 후 실패하면 '미입력'.
    # 인사고과(연도별): S/A/B/C/D/F 6단계 고정값 외의 값은 이상치로 간주해 '미입력'.
    # 인사고과는 빈 값(아직 평가 안 함)도 '미입력'으로 통일해 LLM 할루시네이션을 방지한다.
    if '성과점수' in df.columns:
        for row, raw in df['성과점수'].items():
            try:
                score = int(raw)
            except Exception:
                log(row, df.at[row, '사원번호'], '성과점수', raw, '숫자 변환 불가')
                df.at[row, '성과점수'] = '미입력'
                continue
            if not (MIN_SCORE <= score <= MAX_SCORE):
                log(row, df.at[row, '사원번호'], '성과점수', raw, f'범위 초과 ({MIN_SCORE}~{MAX_SCORE})')
                df.at[row, '성과점수'] = '미입력'
    for year in PERF_YEARS:
        col = f'인사고과_{year}'
        if col not in df.columns:
            continue
        for row in df.index:
            grade_val = str(df.at[row, col]).strip() if df.at[row, col] else ''
            if grade_val in ('nan', 'NaN'):
                grade_val = ''
            emp_id = df.at[row, '사원번호']
            if not grade_val:
                # 빈 값은 해당 연도 평가가 없다는 의미로 '미입력' 처리
                df.at[row, col] = '미입력'
                continue
            if grade_val not in PERF_GRADES:
                log(row, emp_id, col, grade_val, '고정값 외')
                df.at[row, col] = '미입력'


def validate_qual(df):
    # TOEIC점수: 0~990 범위 정수. float 거친 뒤 int 변환 (소수점 표기 대응).
    # 자격증수당여부: '해당'/'비해당' 두 값만 허용.
    # 자격증·포상이력·징계이력: 쉼표 구분 리스트 → parse_array로 정제해 다시 쉼표 결합.
    #   비어있거나 파싱 결과가 없으면 '미입력'.
    # 징계이력·징계사유는 퇴직구분·퇴직일자처럼 짝으로 존재해야 한다:
    #   - 징계이력은 있는데 징계사유가 없으면 → 징계사유 '미입력' + 로그
    #   - 징계이력은 없는데 징계사유가 있으면 → 데이터 오류이므로 징계사유 '미입력' + 로그
    #   - 둘 다 없으면 → 조용히 '미입력' 처리
    if 'TOEIC점수' in df.columns:
        for row, raw in df['TOEIC점수'].items():
            toeic_str = str(raw).strip() if raw else ''
            if not toeic_str or toeic_str in ('nan', 'NaN'):
                df.at[row, 'TOEIC점수'] = '미입력'
                continue
            try:
                toeic = int(float(raw))
            except Exception:
                log(row, df.at[row, '사원번호'], 'TOEIC점수', raw, '숫자 변환 불가')
                df.at[row, 'TOEIC점수'] = '미입력'
                continue
            df.at[row, 'TOEIC점수'] = toeic
            if not (MIN_TOEIC <= toeic <= MAX_TOEIC):
                log(row, df.at[row, '사원번호'], 'TOEIC점수', raw, f'범위 초과 ({MIN_TOEIC}~{MAX_TOEIC})')
                df.at[row, 'TOEIC점수'] = '미입력'
    if '자격증수당여부' in df.columns:
        for row, raw in df['자격증수당여부'].items():
            subsidy = str(raw).strip() if raw else ''
            if subsidy in ('nan', 'NaN'):
                subsidy = ''
            if not subsidy or subsidy not in SUBSIDY_VALUES:
                log(row, df.at[row, '사원번호'], '자격증수당여부', subsidy, '결측/이상치')
                df.at[row, '자격증수당여부'] = '미입력'
    for col in ['자격증', '포상이력', '징계이력']:
        if col not in df.columns:
            continue
        for row, raw in df[col].items():
            # 쉼표로 구분된 여러 값을 정제해 다시 쉼표로 합친다 (공백·빈 항목 제거)
            items = parse_array(raw)
            df.at[row, col] = ','.join(items) if items else '미입력'
    if '징계이력' in df.columns and '징계사유' in df.columns:
        for row in df.index:
            discipline_history = str(df.at[row, '징계이력']).strip()
            if discipline_history in ('nan', 'NaN', '미입력'):
                discipline_history = ''
            discipline_reason = str(df.at[row, '징계사유']).strip() if df.at[row, '징계사유'] else ''
            if discipline_reason in ('nan', 'NaN', '미입력'):
                discipline_reason = ''
            emp_id = df.at[row, '사원번호']
            if discipline_history and not discipline_reason:
                # 징계 기록은 있는데 사유가 없으면 불완전한 데이터
                log(row, emp_id, '징계사유', discipline_reason, '징계이력 있는데 징계사유 없음')
                df.at[row, '징계사유'] = '미입력'
            elif not discipline_history and discipline_reason:
                # 징계 기록도 없는데 사유만 있으면 입력 오류
                log(row, emp_id, '징계사유', discipline_reason, '징계이력 없는데 징계사유 존재')
                df.at[row, '징계사유'] = '미입력'
            elif not discipline_history and not discipline_reason:
                # 둘 다 없으면 징계 없는 직원 → 조용히 '미입력'
                df.at[row, '징계사유'] = '미입력'


def run_preprocessing():
    # 원본 CSV들을 읽어 컬럼별 검증·교정 후 정제된 DataFrame들을 메모리로 반환한다.
    # _errors / drop_rows / valid_rrn / valid_hire 는 파일 하나를 처리할 때마다 새로 비운다.
    global _errors, drop_rows, valid_rrn, valid_hire

    print('\n========== 1단계: 전처리 ==========')
    print(f'입력 폴더: {DATASET_DIR}')

    csv_files = sorted(DATASET_DIR.glob('*.csv'))
    if not csv_files:
        raise SystemExit(f'CSV 파일 없음: {DATASET_DIR}')

    dfs = {}
    source_filenames = {}   # source_name (path.stem) -> 원본 CSV 파일명 (source 필드에 사용)
    for path in csv_files:
        df = pd.read_csv(path, encoding='utf-8-sig', dtype=object)
        dfs[path.stem] = df
        source_filenames[path.stem] = path.name
        print(f'  로딩: {path.name}  ({len(df):,}행 / {len(df.columns)}열)')

    cleaned = {}
    all_errors = []

    for source_name, df in dfs.items():
        print(f'\n처리 중: {source_name}  ({len(df):,}행)')
        _errors    = []
        drop_rows  = set()
        valid_rrn  = {}
        valid_hire = {}

        validate_empid(df)
        validate_name(df)
        validate_rrn(df)
        validate_gender(df)
        validate_birth(df)
        validate_age(df)
        validate_military(df)
        validate_hire(df)
        validate_tenure(df)
        validate_edu(df)
        validate_career(df)
        validate_dept(df)
        validate_grade(df)
        validate_retire(df)
        validate_contact(df)
        validate_salary(df)
        validate_perf(df)
        validate_qual(df)

        for err in _errors:
            err['파일명'] = source_name
        all_errors.extend(_errors)
        print(f'  에러: {len(_errors):,}건')

        df_clean = df.drop(index=list(drop_rows)).reset_index(drop=True)
        cleaned[source_name] = df_clean
        print(f'  정제 결과: {len(df_clean):,}행 (제거 {len(drop_rows)}행)')

    # 에러는 파일로 바로 쓰지 않고 모아서 반환한다.
    # (1~3단계 에러를 마지막에 write_error_log 한 곳에서 단계별로 구분해 한 파일에 남긴다)
    print(f'\n전처리 에러 누적: {len(all_errors):,}건')

    return cleaned, all_errors, source_filenames


# ══════════════════════════════════════════════════════════════════════════════
# 2단계: JSONL 변환
# ══════════════════════════════════════════════════════════════════════════════
# 정제된 DataFrame을 직원별 레코드(dict)로 바꾼다.
# embedding_text는 '필드명: 값'을 한 줄에 하나씩, 줄바꿈(\n)으로 이어 붙여 만든다.
# changed(변경이력)는 여기서 만들지 않고, 4단계에서 OpenSearch와 비교해 채운다 (그래서 일단 빈 목록).

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


def clean(val):
    # 값을 문자열로 변환하고 앞뒤 공백을 제거한다.
    # 변환 결과가 MISSING_VALUES에 해당하면 빈 문자열('')을 반환한다.
    # → 호출하는 쪽에서 빈 문자열인지만 확인하면 "값 없음"을 판단할 수 있다.
    cleaned = str(val).strip()
    return cleaned if cleaned not in MISSING_VALUES else ''


def build_embedding_text(row, info, source_name):
    # 직원 한 명의 한 행(row)을 '필드명: 값\n필드명: 값\n...' 형태의 문자열로 만든다.
    # 이 문자열이 나중에 임베딩 모델의 입력이 된다.
    # → 임베딩 모델은 문장 형태의 텍스트를 받으므로, 표 형식이 아닌 문장처럼 만들어야 한다.
    parts = []      # 완성된 '필드명: 값' 줄들을 모으는 리스트
    seen_keys = []  # 중복 추가 방지용 (이름/부서/직급을 두 번 넣지 않기 위해 사용)

    # 기본인사정보 CSV일 때만 이름·부서·직급을 맨 앞에 넣는다.
    # 다른 CSV(역량성과, 급여정보 등)에는 이 세 필드를 넣지 않는다.
    # → 데이터구조정의서 기준: 이름·부서·직급은 hr_basic_1 인덱스에만 들어가야 하기 때문이다.
    if '기본인사정보' in source_name:
        for key in ['이름', '부서', '직급']:
            # basic_lookup에서 먼저 가져오고, 없으면 현재 행(row)에서 가져온다.
            val = info.get(key) or clean(str(row.get(key, '')))
            if not val:
                val = '미입력'
            parts.append(f'{key}: {val}')
            seen_keys.append(key)

    # EMBEDDING_FIELDS 순서대로 이 CSV에 존재하는 필드만 추가한다.
    # CSV마다 포함된 컬럼이 다르므로, 없는 필드는 조용히 건너뛴다.
    for field in EMBEDDING_FIELDS:
        if field not in row:
            continue  # 이 CSV에 해당 컬럼 자체가 없으면 스킵
        val = clean(str(row[field]))
        if not val:
            val = '미입력'
        if field not in seen_keys:
            parts.append(f'{field}: {val}')
            seen_keys.append(field)

    # 각 줄을 줄바꿈(\n)으로 연결해 하나의 문자열로 반환한다.
    # 예시 결과: "이름: 홍길동\n부서: 개발부\n직급: 대리\n나이: 30\n..."
    return '\n'.join(parts)


def to_record(row, source_name, basic_lookup, source_csv):
    # DataFrame의 한 행(row)을 OpenSearch에 저장할 레코드(딕셔너리) 형태로 변환한다.
    # 레코드는 두 종류의 필드를 가진다:
    #   1) 꼬리표 필드: employee_id, employee_name, department 등
    #      → 검색 결과 필터링·권한 체크에 사용하므로 keyword 타입으로 별도 저장
    #   2) embedding_text / embedding_vector
    #      → 실제 검색 대상 텍스트와 그것을 벡터로 변환한 값
    #
    # 이름·부서·직급은 기본인사정보 CSV에만 있으므로, 다른 CSV에서 이 레코드를 만들 때는
    # basic_lookup(기본인사정보 조회 딕셔너리)에서 사원번호로 찾아 채운다.
    # basic_lookup에도 없으면(기본인사정보 CSV 자체가 없는 경우) 빈 문자열로 남긴다.
    #
    # source_csv: 이 데이터가 어느 원본 파일에서 왔는지 기록 (예: '기본인사정보.csv')
    emp_id = str(row.get('사원번호', '')).strip()
    info   = basic_lookup.get(emp_id, {})  # 사원번호로 기본인사정보 조회

    return {
        'employee_id':      emp_id,
        # info(기본인사정보)에서 먼저 가져오고, 없으면 현재 row에서 가져온다
        'employee_name':    info.get('이름', clean(row.get('이름', ''))),
        'department':       info.get('부서', clean(row.get('부서', ''))),
        'department_level': clean(row.get('부서레벨', '')) or info.get('부서레벨', ''),
        'job_grade':        info.get('직급', clean(row.get('직급', ''))),
        'job_grade_level':  clean(row.get('직급레벨', '')) or info.get('직급레벨', ''),
        'embedding_text':   build_embedding_text(row, info, source_name),
        'source':           source_csv,
        'changed':          [],  # 변경이력은 3단계(인덱싱)에서 OpenSearch와 비교해 채운다
    }


def run_jsonl_conversion(dfs_clean, source_filenames):
    # 1단계에서 정제된 DataFrame들을 직원별 레코드(딕셔너리) 리스트로 바꾼다.
    # 결과인 records_by_source는 {소스명: [레코드, 레코드, ...]} 구조다.
    # 이 레코드들이 3단계에서 OpenSearch에 실제로 저장된다.
    print('\n========== 2단계: JSONL 변환 ==========')

    # ── basic_lookup 만들기 ──────────────────────────────────────────────────
    # 역량성과·급여정보 CSV에는 이름·부서·직급 컬럼이 없다.
    # 그래서 기본인사정보 CSV를 먼저 {사원번호: {이름, 부서, 직급, ...}} 딕셔너리로 만들어두고,
    # 다른 CSV를 레코드로 변환할 때 사원번호로 찾아 꼬리표 필드에 채워 넣는다.
    basic_key = next((k for k in dfs_clean if '기본인사정보' in k), None)
    basic_df = dfs_clean.get(basic_key, pd.DataFrame()) if basic_key else pd.DataFrame()
    basic_lookup = {}
    if not basic_df.empty:
        for row in basic_df.to_dict('records'):
            emp_id = str(row.get('사원번호', '')).strip()
            basic_lookup[emp_id] = {
                '이름':     str(row.get('이름', '')),
                '부서':     str(row.get('부서', '')),
                '부서레벨': str(row.get('부서레벨', '')),
                '직급':     str(row.get('직급', '')),
                '직급레벨': str(row.get('직급레벨', '')),
            }

    if not basic_lookup:
        print('경고: 기본인사정보가 없어 이름/부서/직급이 채워지지 않습니다.')
    print(f'기본인사정보 조회 딕셔너리: {len(basic_lookup):,}건')

    # ── 소스별 레코드 변환 ───────────────────────────────────────────────────
    # 각 CSV(소스)의 DataFrame을 행 단위로 순회하며 to_record()로 레코드를 만든다.
    # DataFrame.to_dict('records') 는 각 행을 {컬럼명: 값} 딕셔너리로 바꿔준다.
    records_by_source = {}
    for source_name, df in dfs_clean.items():
        source_csv = source_filenames.get(source_name, source_name)  # 원본 CSV 파일명
        records = []
        for row in df.to_dict('records'):
            record = to_record(row, source_name, basic_lookup, source_csv)
            records.append(record)
        records_by_source[source_name] = records
        print(f'  변환: {source_name}  ({len(records):,}건)')

    return records_by_source


# ══════════════════════════════════════════════════════════════════════════════
# 청킹 헬퍼 (인덱싱 단계 안에서 호출된다)
# ══════════════════════════════════════════════════════════════════════════════
# 청킹은 별도 단계가 아니라 인덱싱(3단계) 안에서 인덱스별로 호출한다.
# - 이유: 인덱스마다 들어가는 필드 목록이 다른데, 전체 필드를 미리 청킹하면
#         그 인덱스에 안 들어갈 필드까지 토큰을 차지해 같은 인덱스의 필드가
#         여러 청크에 흩어진다 (예: hr_basic_3 의 주민등록번호와 주소가 다른 청크).
# - 해결: 인덱스별로 필요한 필드만 먼저 골라낸 뒤, 그 필터링된 텍스트를
#         MAX_TOKENS 기준으로 청킹한다. 각 인덱스의 필드가 같은 청크에 모인다.

def normalize_text(text):
    # 필드 줄 단위로 공백을 정리한다 (줄바꿈 구조는 유지).
    lines = []
    for line in text.split('\n'):
        words = line.strip().split()
        if words:
            lines.append(' '.join(words))
    return '\n'.join(lines)


def chunk_by_tokens(embedding_text, max_tokens, tokenizer):
    # 필드를 하나씩 누적하며 토큰 한계로 청크를 나눈다.
    # 각 필드의 토큰 수를 미리 한 번씩만 계산해두고 누적 합으로 한계를 판단한다 (빠름).
    fields = []
    for line in embedding_text.split('\n'):
        if line.strip():
            fields.append(line)

    # 각 필드를 special token 없이 토큰화한 길이를 미리 계산
    field_lengths = []
    for field in fields:
        field_lengths.append(len(tokenizer.encode(field, add_special_tokens=False)))
    # 필드 사이를 잇는 줄바꿈(\n)의 토큰 수
    newline_length = len(tokenizer.encode('\n', add_special_tokens=False))
    # 청크 전체에 한 번 붙는 special token([CLS], [SEP]) 수
    special_length = tokenizer.num_special_tokens_to_add()

    chunks = []
    current_fields = []
    current_length = 0  # 현재 청크에 쌓인 토큰 수 (줄바꿈 포함, special 제외)

    for field, field_length in zip(fields, field_lengths):
        # 이 필드를 더하면 늘어나는 토큰 수 (앞에 필드가 있으면 줄바꿈도 더함)
        added = field_length
        if current_fields:
            added += newline_length

        # special token까지 더한 총 토큰이 한계를 넘으면 직전까지로 청크를 마감
        if special_length + current_length + added > max_tokens and current_fields:
            chunks.append('\n'.join(current_fields))
            current_fields = [field]
            current_length = field_length
        else:
            current_fields.append(field)
            current_length += added

    if current_fields:
        chunks.append('\n'.join(current_fields))

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 3단계: 인덱싱 (인덱스별 청킹 + 변경감지)
# ══════════════════════════════════════════════════════════════════════════════
# 청크를 인덱스별(보안등급별)로 라우팅하면서, OpenSearch에 이미 있는 값과 비교해
# 바뀐 직원만 다시 임베딩·적재한다. 어떤 필드가 바뀌었는지는 changed에 기록한다.

def get_dict_hash():
    # 사용자 사전 변경 감지용 해시값을 만든다.
    hasher = hashlib.md5()
    for path in [USER_DICT_FILE]:
        if path.exists():
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def read_last_dict_hash():
    # 지난번 실행 때 저장해둔 사전 해시값을 읽는다.
    # 파일이 없으면(처음 실행) 빈 문자열을 반환해 '항상 변경된 것'으로 처리한다.
    if not LAST_USER_DICT_FILE.exists():
        return ''
    return LAST_USER_DICT_FILE.read_text(encoding='utf-8').strip()


def write_last_dict_hash(hash_value):
    # 이번 실행에서 계산한 해시값을 파일에 저장해둔다.
    # 다음 실행 때 read_last_dict_hash()로 읽어 현재 해시와 비교한다.
    LAST_USER_DICT_FILE.write_text(hash_value, encoding='utf-8')


def build_index_body(security_level):
    # OpenSearch 인덱스 생성 요청 본문을 만든다.
    # settings:
    #   - knn: True → KNN(벡터 유사도 검색) 기능 활성화
    #   - nori_tokenizer: 한국어 형태소 분석기. decompound_mode='mixed' 는
    #     복합어를 원형과 분해 형태 모두 인덱싱해 '백엔드팀' → '백엔드' + '팀' 검색 가능
    # mappings:
    #   - _meta.security_level: 인덱스 레벨 보안등급. 검색 서비스에서 사용자 등급과 비교에 사용
    #   - keyword 타입: 정확히 일치 검색·집계에 사용 (분석 없이 그대로 저장)
    #   - changed: 변경이력 객체. enabled=False → OpenSearch가 색인을 만들지 않아
    #     저장 공간을 절약하면서도 _source에서 읽어올 수는 있다
    #   - embedding_vector: knn_vector 타입. KNN 벡터 검색에 사용
    return {
        'settings': {
            'index': {'knn': True},
            'analysis': {
                'tokenizer': {
                    'nori_tokenizer': {
                        'type': 'nori_tokenizer',
                        'decompound_mode': 'mixed',
                        'user_dictionary': 'user_dictionary.txt',
                    }
                },
                'analyzer': {
                    'korean_analyzer': {
                        'type': 'custom',
                        'tokenizer': 'nori_tokenizer',
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
    # nori 토크나이저는 사용자 사전 파일을 OpenSearch 설치 폴더의 config/ 안에서 읽는다.
    # 우리가 관리하는 사전 파일은 config/ 폴더(프로젝트)에 있으므로,
    # 파이프라인을 실행할 때마다 최신 파일을 OpenSearch config/ 폴더에 복사해 반영한다.
    # → 사전을 수정한 뒤 파이프라인만 다시 돌리면 자동으로 반영된다.
    for src, filename in [(USER_DICT_FILE, 'user_dictionary.txt')]:
        dst = OPENSEARCH_HOME / 'config' / filename
        if not src.exists():
            print(f'{filename} 파일이 없습니다. 건너뜀')
            continue
        try:
            shutil.copy(src, dst)
            print(f'복사 완료: {dst}')
        except Exception as error:
            # 복사 실패는 대부분 OPENSEARCH_HOME 경로가 잘못됐거나 권한 문제다.
            print(f'{filename} 복사 실패 → {error}')
            print(f'OPENSEARCH_HOME 경로를 확인해주세요. ({OPENSEARCH_HOME})')
            raise SystemExit(1)


def ensure_nori_plugin(client):
    # nori 플러그인(한국어 형태소 분석기)이 설치돼 있는지 확인하고, 없으면 자동으로 설치한다.
    # 설치 여부는 plugins/analysis-nori 폴더 존재 여부로 판단한다.
    nori_dir = OPENSEARCH_HOME / 'plugins' / 'analysis-nori'
    if nori_dir.exists():
        return  # 이미 설치돼 있으면 바로 종료

    # 설치돼 있지 않으면 현재 OpenSearch 버전에 맞는 플러그인 zip을 다운로드한다.
    # client.info()로 실행 중인 OpenSearch의 버전 번호를 확인한다.
    info = client.info()
    version = info['version']['number']
    url = (
        f'https://artifacts.opensearch.org/releases/plugins/'
        f'analysis-nori/{version}/analysis-nori-{version}.zip'
    )
    # 임시 폴더(TEMP)에 zip 파일을 받아 저장한다.
    zip_path = Path(os.getenv('TEMP', '/tmp')) / f'analysis-nori-{version}.zip'

    print(f'nori 플러그인 다운로드 중... ({url})')
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as error:
        print(f'nori 플러그인 다운로드 실패 → {error}')
        print('네트워크 연결을 확인하거나 수동으로 플러그인을 설치해주세요.')
        raise SystemExit(1)

    # zip 파일을 plugins/analysis-nori 폴더에 압축 해제한다.
    nori_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as archive:
        archive.extractall(nori_dir)
    zip_path.unlink()  # 압축 해제 후 zip 파일 삭제 (더 이상 필요 없으므로)

    # 플러그인을 설치한 뒤에는 OpenSearch를 재시작해야 적용된다.
    # 재시작 없이 계속 실행하면 nori 인덱스 생성이 실패하므로 여기서 종료한다.
    print('nori 플러그인 설치 완료!')
    print('OpenSearch를 재시작한 후 다시 실행해 주세요.')
    raise SystemExit(0)


def parse_embedding_text(text):
    # embedding_text("이름: 홍길동\n부서: 개발부\n...")를
    # {'이름': '홍길동', '부서': '개발부', ...} 딕셔너리로 역변환한다.
    # split(': ', 1) 에서 1은 '최대 1번만 분리'를 의미한다.
    # → 값 안에 ': '가 포함된 경우(예: '주소: 서울시 강남구: 역삼동')에도 키·값이 올바르게 나뉜다.
    result = {}
    for line in text.split('\n'):
        line = line.strip()
        if ': ' not in line:
            continue  # 형식이 맞지 않는 줄(빈 줄 등)은 건너뛴다
        key, value = line.split(': ', 1)
        result[key.strip()] = value.strip()
    return result


def build_filtered_text(parsed, fields):
    # parse_embedding_text()로 변환된 딕셔너리(parsed)에서
    # 이 인덱스에 필요한 필드(fields)만 골라 다시 '필드명: 값\n...' 문자열로 만든다.
    # → 인덱스마다 다른 필드만 임베딩하기 위해 필터링하는 단계다.
    # 예) hr_basic_3 인덱스라면 fields = ['주민등록번호', '주소', '퇴직구분', '퇴직일자']만 포함
    parts = []
    for field in fields:
        if parsed.get(field):  # 값이 있는 필드만 포함 (빈 값·미입력은 제외)
            parts.append(f'{field}: {parsed[field]}')
    return '\n'.join(parts)


def create_indices(client):
    # INDEX_CONFIG에 정의된 인덱스를 OpenSearch에 생성한다.
    # 인덱스가 이미 존재하면 건너뛰고, 없는 것만 새로 만든다.
    #
    # ── 사전 변경 감지 ──────────────────────────────────────────────────────
    # nori 토크나이저는 인덱스를 만들 때 사전을 읽어 분석기를 구성한다.
    # 즉, 사전을 수정해도 기존 인덱스에는 자동으로 반영되지 않는다.
    # → 사전이 바뀌면 인덱스를 삭제하고 다시 만들어야 새 사전이 적용된다.
    # 이를 위해 사전 파일의 MD5 해시값을 파일에 저장해두고,
    # 이번 실행의 해시와 비교해 변경 여부를 판단한다.
    print('\n========== 인덱스 생성 ==========')

    current_hash = get_dict_hash()       # 현재 사전 파일의 해시값
    last_hash    = read_last_dict_hash() # 지난번 실행 때 저장한 해시값
    dict_changed = current_hash != last_hash  # 두 값이 다르면 사전이 바뀐 것

    apply_change = dict_changed
    if dict_changed:
        # 사전이 바뀌었으니 기존 인덱스를 모두 삭제해 재생성 준비를 한다.
        existing = [name for name in INDEX_CONFIG if client.indices.exists(index=name)]
        if existing:
            print('  사용자 사전(user_dictionary.txt) 변경 감지됨')
            print(f'  기존 인덱스 {len(existing)}개를 삭제하고 재생성합니다: {", ".join(existing)}')
            for name in existing:
                client.indices.delete(index=name)
                print(f'  삭제: {name}')

    # INDEX_CONFIG의 각 인덱스를 순서대로 생성한다.
    # 삭제된 인덱스나 처음 실행 시에는 새로 만들고, 이미 있으면 건너뛴다.
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

    # 사전이 변경됐고 인덱스 재생성도 완료됐다면, 이번 해시값을 파일에 저장한다.
    # 다음 실행 때 사전이 또 바뀌었는지 비교하기 위해서다.
    if apply_change:
        write_last_dict_hash(current_hash)


def get_existing_docs(client, index_name):
    # 이 인덱스에 현재 저장된 문서를 모두 읽어 직원별로 묶어 반환한다.
    # 반환 구조: {사원번호: {'text': 전체텍스트, 'changed': 변경이력, 'meta': 꼬리표}}
    #
    # 이 함수가 필요한 이유:
    #   파이프라인을 여러 번 실행했을 때, 이미 저장된 직원 데이터와 새 데이터를 비교해
    #   바뀐 직원만 다시 임베딩하고 적재해야 한다 (불필요한 재계산 방지).
    #
    # helpers.scan을 사용하는 이유:
    #   일반 search 요청은 기본 10건, 최대 10,000건만 가져올 수 있다.
    #   직원 수가 많으면 그 이상 나올 수 있으므로, scan(스크롤 API)으로 전체를 안전하게 읽는다.
    if not client.indices.exists(index=index_name):
        return {}  # 인덱스가 아직 없으면 비교할 데이터도 없다

    chunks_by_emp  = {}   # 사원번호 -> [(doc_id, embedding_text), ...]
    changed_by_emp = {}   # 사원번호 -> 옛 변경이력 리스트
    meta_by_emp    = {}   # 사원번호 -> 옛 꼬리표(이름·부서·직급·레벨)

    # embedding_text와 꼬리표를 함께 읽는다.
    # 꼬리표(이름·부서·직급)는 어떤 인덱스든 공통으로 저장되므로,
    # 텍스트는 그대로여도 꼬리표만 바뀐 경우(예: 부서 이동)를 감지하려면 꼬리표도 비교해야 한다.
    source_fields = ['employee_id', 'embedding_text', 'changed',
                     'employee_name', 'department', 'department_level',
                     'job_grade', 'job_grade_level']
    search_body = {'_source': source_fields, 'query': {'match_all': {}}}
    for doc in helpers.scan(client, index=index_name, query=search_body):
        source = doc['_source']
        emp_id = source.get('employee_id', '')
        doc_id = doc['_id']
        # 직원 한 명의 청크 문서가 여러 개일 수 있으므로 리스트에 쌓는다
        if emp_id not in chunks_by_emp:
            chunks_by_emp[emp_id] = []
        chunks_by_emp[emp_id].append((doc_id, source.get('embedding_text', '')))
        changed_by_emp[emp_id] = source.get('changed', [])
        meta_by_emp[emp_id] = {
            'employee_name':    source.get('employee_name', ''),
            'department':       source.get('department', ''),
            'department_level': source.get('department_level', 0),
            'job_grade':        source.get('job_grade', ''),
            'job_grade_level':  source.get('job_grade_level', 0),
        }

    # 청크 문서들을 doc_id 순서로 정렬한 뒤 텍스트를 이어 붙인다.
    # 예) EMP0001_0001, EMP0001_0002 → 두 청크 텍스트를 줄바꿈으로 연결
    # → 청크가 여러 개여도 "직원 한 명의 전체 텍스트"로 만들어 새 데이터와 한 번에 비교한다.
    result = {}
    for emp_id in chunks_by_emp:
        pairs = chunks_by_emp[emp_id]
        pairs.sort()   # doc_id 문자열 오름차순 = _0001, _0002 순서
        texts = []
        for doc_id, text in pairs:
            texts.append(text)
        result[emp_id] = {
            'text':    '\n'.join(texts),
            'changed': changed_by_emp.get(emp_id, []),
            'meta':    meta_by_emp.get(emp_id, {}),
        }
    return result


# 변경 감지 때 비교할 꼬리표(메타데이터) 필드 목록. (키, 한글이름, 정수레벨여부)
META_COMPARE_FIELDS = [
    ('employee_name',    '이름',     False),
    ('department',       '부서',     False),
    ('department_level', '부서레벨', True),
    ('job_grade',        '직급',     False),
    ('job_grade_level',  '직급레벨', True),
]


def normalize_meta_value(value, is_level):
    # 레벨 필드는 저장될 때 정수라서, 비교 전에 정수 문자열로 맞춰 거짓 변경을 막는다.
    # (예: 새 값 '' 과 저장값 0 이 다르게 보이는 문제 방지)
    if is_level:
        return str(int(value or 0))
    return str(value)


def doc_signature(meta, text):
    # 꼬리표(meta) + 텍스트(text)를 줄바꿈으로 이은 하나의 문자열을 반환한다.
    # 두 문서의 signature가 같으면 아무것도 바뀌지 않은 것이고,
    # 다르면 무언가 바뀐 것이므로 다시 임베딩·적재해야 한다.
    # 이 방식을 쓰는 이유: 텍스트만 비교하면 부서·직급처럼 꼬리표에만 있는 변화를 놓친다.
    parts = []
    for key, label, is_level in META_COMPARE_FIELDS:
        parts.append(normalize_meta_value(meta.get(key, ''), is_level))
    parts.append(text)
    return '\n'.join(parts)


def build_change_entry(old_meta, new_meta, old_text, new_text, now):
    # 옛 문서와 새 문서를 필드 단위로 비교해 어떤 값이 바뀌었는지 기록한다.
    # 결과 형태: {'timestamp': '2024-01-15 10:30:00', 'fields': [{'field': '부서', 'old': '영업부', 'new': '개발부'}, ...]}
    # 이 기록은 OpenSearch의 changed 필드에 리스트로 쌓이며, 직원의 이력 조회에 활용된다.
    changed_fields = []

    # ① 꼬리표(이름·부서·직급·레벨) 비교
    # META_COMPARE_FIELDS의 각 필드를 정규화(normalize_meta_value)한 뒤 문자열로 비교한다.
    for key, label, is_level in META_COMPARE_FIELDS:
        old_value = normalize_meta_value(old_meta.get(key, ''), is_level)
        new_value = normalize_meta_value(new_meta.get(key, ''), is_level)
        if old_value != new_value:
            changed_fields.append({'field': label, 'old': old_value, 'new': new_value})

    # ② embedding_text 안의 필드 비교
    # 꼬리표에서 이미 비교한 필드는 건너뛴다 (이름·부서·직급이 두 번 기록되지 않도록).
    # parse_embedding_text로 '필드명: 값' 형식을 딕셔너리로 바꿔 필드별로 비교한다.
    meta_labels = {label for _, label, _ in META_COMPARE_FIELDS}
    old_fields = parse_embedding_text(old_text)
    new_fields = parse_embedding_text(new_text)
    for field_name in new_fields:
        if field_name in meta_labels:
            continue  # 꼬리표에서 이미 비교한 필드 중복 방지
        old_value = old_fields.get(field_name, '')  # 옛 문서에 없던 필드면 빈 문자열
        new_value = new_fields[field_name]
        if str(old_value) != str(new_value):
            changed_fields.append({'field': field_name, 'old': str(old_value), 'new': str(new_value)})

    return {'timestamp': now, 'fields': changed_fields}


def run_indexing(records_by_source, model, client):
    # 3단계 전체 흐름:
    #   ① 모든 소스의 필드를 직원별로 합친다 (employees 딕셔너리)
    #   ② 인덱스마다 → 필요한 필드만 필터링 → 청킹 → 옛 문서와 비교
    #   ③ 신규·변경 직원만 임베딩해 OpenSearch에 적재한다
    #
    # 인덱스마다 반복하는 이유:
    #   hr_basic_1은 이름·성별·나이 등, hr_salary_3은 연봉·계좌번호 등 필드가 다르다.
    #   인덱스별로 필요한 필드만 골라 청킹해야 같은 인덱스 필드끼리 같은 청크에 모인다.
    print('\n========== 3단계: 인덱싱 (인덱스별 청킹 + 변경감지) ==========')

    # 임베딩 모델의 토크나이저를 꺼낸다 (청킹 시 토큰 수 계산에 사용)
    tokenizer = model.tokenizer

    # 이번 실행에서 실제로 저장하는 문서에만 같은 시각을 찍는다.
    # 무변경 문서는 건드리지 않으므로 timestamp가 갱신되지 않는다.
    indexed_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    indexing_errors = []   # 적재 실패 모음 (마지막에 error.log로 남긴다)
    chunking_errors = []   # 청킹 경고 모음 (마지막에 error.log로 남긴다)

    # ── ① 직원별 필드 통합 ──────────────────────────────────────────────────
    # 기본인사정보·역량성과·급여정보 세 CSV의 레코드가 records_by_source에 소스별로 따로 있다.
    # 이를 사원번호 기준으로 합쳐 employees[사원번호]['parsed'] 에 모든 필드를 모은다.
    # → 나중에 인덱스별로 필드를 골라낼 때 한 곳에서 꺼낼 수 있게 된다.
    employees = {}        # 사원번호 -> {'meta': 레코드(꼬리표), 'parsed': {필드명: 값, ...}}
    field_to_source = {}  # 필드명 -> 원본 CSV 파일명 (예: '연봉' -> '급여정보.csv')
    for source_name in records_by_source:
        for rec in records_by_source[source_name]:
            emp_id = rec.get('employee_id', '')
            # embedding_text를 다시 딕셔너리로 파싱해 필드별로 접근할 수 있게 한다
            parsed = parse_embedding_text(rec.get('embedding_text', ''))
            if emp_id not in employees:
                # 처음 등장한 직원이면 빈 딕셔너리로 초기화
                employees[emp_id] = {'meta': rec, 'parsed': {}}
            # 같은 직원이 여러 소스에 있으면 parsed를 덮어쓰지 않고 합친다 (update)
            employees[emp_id]['parsed'].update(parsed)
            # 필드가 어느 CSV에서 왔는지 기록해둔다 (인덱스별 source 필드 값 결정에 사용)
            for field in parsed:
                if field not in field_to_source:
                    field_to_source[field] = rec.get('source', '')

    # ── ② 인덱스별 처리 ────────────────────────────────────────────────────────
    # INDEX_CONFIG에 정의된 인덱스(hr_basic_1, hr_basic_2, ...) 를 하나씩 처리한다.
    for index_name, config in INDEX_CONFIG.items():
        fields = config['fields']  # 이 인덱스에 저장할 필드 목록

        # 이 인덱스에 들어가는 필드들이 어느 원본 CSV에서 왔는지 확인한다.
        # 같은 인덱스의 필드는 모두 같은 CSV에서 오므로 첫 번째 매칭 필드만 확인한다.
        # 결과는 OpenSearch 문서의 source 필드에 저장된다 (예: '급여정보.csv').
        index_source = ''
        for field in fields:
            if field in field_to_source:
                index_source = field_to_source[field]
                break

        # (가) OpenSearch에서 이 인덱스의 기존 문서를 모두 읽어온다.
        # 나중에 새 데이터와 비교해 신규·변경·무변경을 판단한다.
        old_data = get_existing_docs(client, index_name)

        # (나) 각 직원에 대해 이 인덱스에 필요한 필드만 골라 청킹한다.
        # employees에는 모든 소스의 필드가 합쳐져 있으므로,
        # build_filtered_text로 이 인덱스 필드만 추출한 뒤 청킹한다.
        new_by_emp = {}   # 사원번호 -> {'meta': 레코드, 'chunk_texts': [청크텍스트, ...]}
        for emp_id in employees:
            info = employees[emp_id]
            # 이 인덱스에 필요한 필드만 골라 텍스트를 만든다
            filtered = build_filtered_text(info['parsed'], fields)
            if not filtered:
                # 이 직원에게 이 인덱스의 필드 데이터가 없으면 건너뛴다
                continue
            normalized = normalize_text(filtered)
            if not normalized.strip():
                chunking_errors.append({
                    '사원번호': emp_id,
                    '사유':    '빈 embedding_text로 스킵',
                    '상세':    index_name,
                })
                continue
            # MAX_TOKENS 기준으로 필드를 여러 청크로 나눈다
            chunk_texts = chunk_by_tokens(normalized, MAX_TOKENS, tokenizer)
            new_by_emp[emp_id] = {'meta': info['meta'], 'chunk_texts': chunk_texts}

            # 청킹 결과를 검증한다. 필드 하나가 MAX_TOKENS보다 길면
            # 임베딩 모델이 뒷부분을 잘라내므로 경고를 기록한다.
            for chunk_text in chunk_texts:
                token_count = len(tokenizer.encode(chunk_text))
                if token_count > MAX_TOKENS:
                    chunking_errors.append({
                        '사원번호': emp_id,
                        '사유':    '토큰 한계 초과',
                        '상세':    f'{index_name}: {token_count}토큰 > 한계 {MAX_TOKENS} (임베딩 시 잘릴 수 있음)',
                    })

        # (다) 신규 / 변경 / 무변경 분류
        # texts_to_embed와 plan은 반드시 같은 순서로 쌓아야 한다.
        # model.encode()가 texts_to_embed를 순서대로 벡터로 변환하고,
        # 그 결과를 plan과 zip으로 짝지어 문서를 만들기 때문이다.
        texts_to_embed = []   # 임베딩할 텍스트 목록 (청크 단위)
        plan           = []   # 적재할 문서 정보 목록: (사원번호, 꼬리표, 변경이력, 청크텍스트)
        update_emp_ids = []   # 기존 청크를 삭제해야 하는 직원 목록 (변경 직원만)
        new_count      = 0    # 신규 직원 수 (통계용)
        changed_count  = 0    # 변경 직원 수 (통계용)
        skip_count     = 0    # 무변경 직원 수 (통계용)

        for emp_id in new_by_emp:
            info = new_by_emp[emp_id]
            # 청크들을 다시 이어 붙여 "이 직원의 전체 텍스트"로 만든다 (비교용)
            new_text = '\n'.join(info['chunk_texts'])
            new_meta = info['meta']  # 꼬리표(이름·부서·직급 등)

            if emp_id not in old_data:
                # OpenSearch에 없는 직원 → 신규 적재, 변경이력 없음
                changed = []
                new_count += 1
            else:
                old_text = old_data[emp_id]['text']
                old_meta = old_data[emp_id]['meta']
                # doc_signature로 꼬리표+텍스트 전체를 비교한다.
                # 텍스트만 같아도 부서·직급이 바뀌면 signature가 달라진다.
                if doc_signature(old_meta, old_text) == doc_signature(new_meta, new_text):
                    # 아무것도 바뀌지 않았으면 임베딩·적재를 생략한다 (비용 절약)
                    skip_count += 1
                    continue
                # 뭔가 바뀐 직원 → 변경이력을 기존 이력 뒤에 덧붙인다
                old_changed = old_data[emp_id]['changed']
                changed = old_changed + [build_change_entry(old_meta, new_meta, old_text, new_text, indexed_at)]
                update_emp_ids.append(emp_id)  # 기존 청크 삭제 대상에 추가
                changed_count += 1

            # 이 직원의 청크들을 임베딩 목록과 plan에 추가한다
            for chunk_text in info['chunk_texts']:
                texts_to_embed.append(chunk_text)
                plan.append((emp_id, info['meta'], changed, chunk_text))

        if not plan:
            # 이 인덱스에서 신규·변경 직원이 한 명도 없으면 적재할 게 없다
            print(f'  [{index_name}] 변경 없음 (건너뜀)')
            continue

        # 변경된 직원의 기존 청크를 먼저 삭제한다.
        # 이유: 청킹 결과 청크 수가 달라질 수 있어서 기존 청크(_0001, _0002 등)를
        # 그냥 덮어쓰면 이전 청크가 남아 중복 문서가 생긴다.
        # delete_by_query + refresh='true' 로 삭제 후 즉시 검색에서 사라지게 한다.
        if update_emp_ids:
            client.delete_by_query(
                index=index_name,
                body={'query': {'terms': {'employee_id': update_emp_ids}}},
                params={'refresh': 'true'},
            )

        # 적재할 청크 텍스트들을 한꺼번에 임베딩 벡터로 변환한다.
        # batch_size만큼 묶어 처리하므로 직원 수가 많아도 메모리를 절약할 수 있다.
        # convert_to_numpy=True 는 결과를 numpy 배열로 반환해 .tolist() 호출이 가능하게 한다.
        vectors = model.encode(
            texts_to_embed, batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False, convert_to_numpy=True,
        )

        # plan과 vectors를 zip으로 묶어 각 청크의 벡터와 메타정보를 짝지어 문서를 만든다.
        # 문서 ID는 'EMP0001_0001' 형식 (사원번호 + 청크 순번 4자리).
        # chunk_counter로 직원별 청크 순번을 1부터 매긴다.
        chunk_counter = {}  # 사원번호 -> 현재까지 만든 청크 수
        actions = []        # helpers.bulk에 넘길 OpenSearch 적재 요청 목록
        for plan_item, vector in zip(plan, vectors):
            emp_id, meta, changed, chunk_text = plan_item
            chunk_counter[emp_id] = chunk_counter.get(emp_id, 0) + 1
            doc_id = f'{emp_id}_{chunk_counter[emp_id]:04d}'  # 예: EMP0001_0001
            actions.append({
                '_index': index_name,
                '_id':    doc_id,
                '_source': {
                    'employee_id':      meta.get('employee_id', ''),
                    'employee_name':    meta.get('employee_name', ''),
                    'department':       meta.get('department', ''),
                    'department_level': int(meta.get('department_level', 0) or 0),
                    'job_grade':        meta.get('job_grade', ''),
                    'job_grade_level':  int(meta.get('job_grade_level', 0) or 0),
                    'source':           index_source,
                    'timestamp':        indexed_at,
                    'changed':          changed,
                    'embedding_text':   chunk_text,
                    # vector는 numpy 배열이므로 .tolist()로 파이썬 리스트로 변환해야 JSON 직렬화된다
                    'embedding_vector': vector.tolist(),
                },
            })

        # helpers.bulk로 actions 목록을 OpenSearch에 한꺼번에 전송한다.
        # raise_on_error=False 로 설정해 일부 문서 적재 실패 시 예외를 던지지 않고
        # failed 리스트에 모아 계속 진행한다 (한 문서 실패로 전체가 멈추지 않게).
        success, failed = helpers.bulk(client, actions, raise_on_error=False)
        print(
            f'  [{index_name}]  '
            f'신규 {new_count}명 / 변경 {changed_count}명 / 무변경 {skip_count}명  →  '
            f'{success:,}건 적재 (실패 {len(failed)}건)'
        )
        if failed:
            print(f'    경고: {len(failed)}건 적재 실패 → {failed[0]}')
            # helpers.bulk의 실패 항목 구조: {동작이름: {'_id': ..., 'error': ...}}
            # list(item.values())[0] 으로 안쪽 딕셔너리를 꺼낸다.
            for item in failed:
                info = list(item.values())[0]
                indexing_errors.append({
                    '인덱스명': index_name,
                    '문서ID':  info.get('_id', ''),
                    '오류':    str(info.get('error', '')),
                })

    return indexing_errors, chunking_errors


# ══════════════════════════════════════════════════════════════════════════════
# 전체 실행
# ══════════════════════════════════════════════════════════════════════════════

def write_error_log(preprocessing_errors, chunking_errors, indexing_errors):
    # 1~3단계에서 수집한 에러·경고를 하나의 파일(error.log)에 단계별로 구분해 저장한다.
    # 실행이 끝난 뒤 이 파일을 열어 어떤 직원 데이터에 문제가 있었는지 확인할 수 있다.
    #
    # 섹션 구조:
    #   [1단계 전처리 에러] 컬럼이 왜 어떤 값으로 교정됐는지, 어떤 행이 제거됐는지
    #   [3단계 청킹 경고]   토큰 한계 초과·빈 텍스트 등 임베딩 관련 경고
    #   [3단계 적재 실패]   OpenSearch에 문서를 넣지 못한 경우
    #
    # encoding='utf-8-sig': 엑셀에서 열었을 때 한글이 깨지지 않도록 BOM을 붙인다.
    os.makedirs(ERROR_LOG_PATH.parent, exist_ok=True)
    with open(ERROR_LOG_PATH, 'w', encoding='utf-8-sig') as log_file:
        log_file.write('========== 1단계: 전처리 에러 ==========\n')
        columns = ['파일명', '행', '사원번호', '컬럼', '원본값', '사유']
        # pd.DataFrame으로 리스트를 표 형태로 만든 뒤 CSV 문자열로 변환해 파일에 쓴다
        log_file.write(pd.DataFrame(preprocessing_errors, columns=columns).to_csv(index=False))

        log_file.write('\n========== 3단계: 청킹 경고 (인덱싱 내) ==========\n')
        columns = ['사원번호', '사유', '상세']
        log_file.write(pd.DataFrame(chunking_errors, columns=columns).to_csv(index=False))

        log_file.write('\n========== 3단계: 적재 실패 ==========\n')
        columns = ['인덱스명', '문서ID', '오류']
        log_file.write(pd.DataFrame(indexing_errors, columns=columns).to_csv(index=False))

    total = len(preprocessing_errors) + len(chunking_errors) + len(indexing_errors)
    print(f'\n에러 로그 저장: {ERROR_LOG_PATH}  (총 {total:,}건)')


def main():
    # 전체 파이프라인의 진입점.
    # 실행 순서:
    #   1) OpenSearch 연결 확인
    #   2) 사용자 사전(user_dictionary.txt)을 OpenSearch config 폴더로 복사
    #   3) nori 플러그인 설치 여부 확인 (없으면 자동 다운로드 후 재시작 안내)
    #   4) 임베딩 모델 로딩
    #   5) 인덱스 생성 (사전 변경 감지 → 기존 인덱스 삭제 후 재생성 포함)
    #   6) 1단계(전처리) → 2단계(JSONL 변환) → 3단계(인덱싱) 순서로 실행
    #   7) 에러 로그 파일 저장
    print('\n========== OpenSearch 연결 ==========')
    client = OpenSearch(
        hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=OPENSEARCH_USE_SSL,
        verify_certs=OPENSEARCH_VERIFY_CERTS,
        ssl_show_warn=False,
    )

    try:
        # client.info()로 실제 연결 여부를 확인한다 (client 객체 생성만으로는 연결되지 않음)
        client.info()
    except Exception as error:
        print(f'\nOpenSearch 연결 실패 →\n {error}\n')
        print(f'OpenSearch가 실행 중인지 확인해주세요. ({OPENSEARCH_HOST}:{OPENSEARCH_PORT})')
        raise SystemExit(1)

    ensure_user_dictionary()
    ensure_nori_plugin(client)

    print(f'\n임베딩 모델 로딩: {EMBEDDING_MODEL}')
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as error:
        print(f'임베딩 모델 로딩 실패 → {error}')
        print(f'모델명({EMBEDDING_MODEL})과 네트워크 연결을 확인해주세요.')
        raise SystemExit(1)

    create_indices(client)

    # 파이프라인 실행 (단계 사이 데이터는 파일이 아니라 메모리로 전달)
    # 청킹은 별도 단계가 아니라 run_indexing 안에서 인덱스별로 수행된다.
    dfs_clean, preprocessing_errors, source_filenames = run_preprocessing()
    records_by_source                                  = run_jsonl_conversion(dfs_clean, source_filenames)
    indexing_errors, chunking_errors                   = run_indexing(records_by_source, model, client)

    # 1~3단계 에러를 한 파일에 단계별로 구분해 남긴다
    write_error_log(preprocessing_errors, chunking_errors, indexing_errors)

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n========== 전체 완료 ({now}) ==========')


if __name__ == '__main__':
    main()
