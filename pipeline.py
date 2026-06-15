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
SYNONYM_FILE        = CONFIG_DIR / 'synonym.txt'               # 동의어 사전
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
    _errors.append({
        '행': row,
        '사원번호': emp_id,
        '컬럼': column,
        '원본값': original_value,
        '사유': reason
    })


def calc_age(birth):
    return TODAY.year - birth.year + 1


def tenure_years(hire_date):
    return TODAY.year - hire_date.year


def parse_date(text):
    try:
        text = str(text).strip()
        if not text or text in ('nan', 'NaN', 'None', '미입력'):
            return None
        return pd.to_datetime(text).date()
    except Exception:
        return None


def birth_from_rrn(rrn):
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
    if not cell_value or str(cell_value).strip() in ('nan', 'NaN', 'None'):
        return []
    items = []
    for item in str(cell_value).split(','):
        item = item.strip()
        if item:
            items.append(item)
    return items


def is_valid_emp_id(emp_id):
    if len(emp_id) != 7:
        return False
    if emp_id[:3] != 'EMP':
        return False
    if not emp_id[3:].isdigit():
        return False
    return True


def is_valid_name(name):
    for char in name:
        if not char.isalpha():
            return False
    return True


def is_valid_rrn_format(rrn):
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
    for char in bank:
        if not char.isalpha() and char != ' ':
            return False
    return True


def validate_empid(df):
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
        valid_rrn[row] = birth


def validate_gender(df):
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
    if '생년월일' not in df.columns:
        return
    for row in df.index:
        rrn_birth = valid_rrn.get(row)
        birth_str = str(df.at[row, '생년월일']).strip() if df.at[row, '생년월일'] else ''
        if birth_str in ('nan', 'NaN'):
            birth_str = ''
        birth = parse_date(birth_str)
        if rrn_birth:
            if birth is None or not (MIN_AGE <= calc_age(birth) <= MAX_AGE):
                if birth is not None:
                    log(row, df.at[row, '사원번호'], '생년월일', birth_str, '범위 초과 → 주민번호로 교정')
                df.at[row, '생년월일'] = rrn_birth.strftime('%Y-%m-%d')
            else:
                df.at[row, '생년월일'] = birth.strftime('%Y-%m-%d')
        else:
            if birth is None:
                log(row, df.at[row, '사원번호'], '생년월일', birth_str, '결측/파싱불가 + 주민번호 없음')
                df.at[row, '생년월일'] = '미입력'
            elif not (MIN_AGE <= calc_age(birth) <= MAX_AGE):
                log(row, df.at[row, '사원번호'], '생년월일', birth_str, '범위 초과')
                df.at[row, '생년월일'] = '미입력'
            else:
                df.at[row, '생년월일'] = birth.strftime('%Y-%m-%d')


def validate_age(df):
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
            df.at[row, '나이'] = calc_age(valid_rrn[row])
        else:
            log(row, df.at[row, '사원번호'], '나이', df.at[row, '나이'], '생년월일·주민번호 없음')
            df.at[row, '나이'] = '미입력'


def validate_military(df):
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
            if univ and univ != '미입력':
                edu = '전문대졸' if '전문' in univ else '대졸'
                df.at[row, '학력'] = edu
            else:
                log(row, emp_id, '학력', edu, '결측 + 출신대학 없음')
                df.at[row, '학력'] = '미입력'
                edu = '미입력'
        if edu == '고졸':
            if univ:
                log(row, emp_id, '출신대학', univ, '고졸인데 출신대학 존재 → 삭제')
                df.at[row, '출신대학'] = ''
        elif edu in ('대졸', '전문대졸', '대학원졸') and '출신대학' in df.columns:
            if not univ:
                log(row, emp_id, '출신대학', univ, '결측')
                df.at[row, '출신대학'] = '미입력'
            elif edu == '전문대졸' and '전문' not in univ:
                log(row, emp_id, '출신대학', univ, '학력=전문대졸인데 일반대학교명')
                df.at[row, '출신대학'] = '미입력'
            elif edu == '대졸' and '전문' in univ:
                log(row, emp_id, '출신대학', univ, '학력=대졸인데 전문대학교명')
                df.at[row, '출신대학'] = '미입력'


def validate_career(df):
    if '학점' in df.columns:
        for row in df.index:
            edu = str(df.at[row, '학력']).strip() if '학력' in df.columns else ''
            gpa_raw = df.at[row, '학점']
            if edu == '고졸':
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
            for col in ['이전직장명', '이전최종직급', '이전담당업무']:
                if col not in df.columns:
                    continue
                cell_value = str(df.at[row, col]).strip() if df.at[row, col] else ''
                if cell_value in ('nan', 'NaN'):
                    cell_value = ''
                if not cell_value:
                    df.at[row, col] = '미입력'


def validate_dept(df):
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
                log(row, df.at[row, '사원번호'], '팀', team, f'부서({dept})-팀 매핑 불일치')
                df.at[row, '팀'] = '미입력'
    if '부서레벨' in df.columns:
        for row in df.index:
            dept = str(df.at[row, '부서']).strip()
            if dept in DEPT_LEVEL_MAP:
                df.at[row, '부서레벨'] = DEPT_LEVEL_MAP[dept]
            else:
                log(row, df.at[row, '사원번호'], '부서레벨', df.at[row, '부서레벨'], '부서 없음 → 행 제거')
                drop_rows.add(row)


def validate_grade(df):
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
        ceo_row = None
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
                log(row, df.at[row, '사원번호'], '직책', position, '직급=사장인데 직책≠대표이사 → 교정')
                df.at[row, '직책'] = '대표이사'
                position = '대표이사'
            if position == '대표이사':
                if ceo_row is None:
                    ceo_row = row
                else:
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
                log(row, emp_id, '퇴직일자', retire_date_str, '입사일 이전')
                df.at[row, '퇴직일자'] = '미입력'
                retire_date = None
                retire_date_str = ''
            elif retire_date > TODAY:
                log(row, emp_id, '퇴직일자', retire_date_str, '현재 날짜 초과')
                df.at[row, '퇴직일자'] = '미입력'
                retire_date = None
                retire_date_str = ''
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
    for col, min_val, max_val in [
        ('연봉',          MIN_SALARY,       MAX_SALARY),
        ('잔업시간',       MIN_OVERTIME,     MAX_OVERTIME),
        ('미사용휴가일수', MIN_UNUSED_LEAVE, MAX_UNUSED_LEAVE),
    ]:
        if col not in df.columns:
            continue
        for row, raw in df[col].items():
            try:
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
                df.at[row, col] = '미입력'
                continue
            if grade_val not in PERF_GRADES:
                log(row, emp_id, col, grade_val, '고정값 외')
                df.at[row, col] = '미입력'


def validate_qual(df):
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
                log(row, emp_id, '징계사유', discipline_reason, '징계이력 있는데 징계사유 없음')
                df.at[row, '징계사유'] = '미입력'
            elif not discipline_history and discipline_reason:
                log(row, emp_id, '징계사유', discipline_reason, '징계이력 없는데 징계사유 존재')
                df.at[row, '징계사유'] = '미입력'
            elif not discipline_history and not discipline_reason:
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

MISSING_VALUES = {'', '미입력', 'nan', 'NaN', 'None', 'none'}

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
    cleaned = str(val).strip()
    return cleaned if cleaned not in MISSING_VALUES else ''


def build_embedding_text(row, info, source_name):
    parts = []
    seen_keys = []

    # 기본인사정보일 때만 이름/부서/직급을 embedding_text에 포함한다
    # (데이터구조정의서 기준: 이름/부서/직급은 hr_basic_1에만 적재)
    if '기본인사정보' in source_name:
        for key in ['이름', '부서', '직급']:
            val = info.get(key) or clean(str(row.get(key, '')))
            if not val:
                val = '미입력'
            parts.append(f'{key}: {val}')
            seen_keys.append(key)

    # 해당 CSV에 존재하는 필드만 처리 (없는 필드는 스킵)
    for field in EMBEDDING_FIELDS:
        if field not in row:
            continue
        val = clean(str(row[field]))
        if not val:
            val = '미입력'
        if field not in seen_keys:
            parts.append(f'{field}: {val}')
            seen_keys.append(field)

    return '\n'.join(parts)


def to_record(row, source_name, basic_lookup, source_csv):
    # source_csv: 원본 CSV 파일명 (예: '기본인사정보.csv'). OpenSearch source 필드로 저장.
    emp_id = str(row.get('사원번호', '')).strip()
    info   = basic_lookup.get(emp_id, {})

    return {
        'employee_id':      emp_id,
        'employee_name':    info.get('이름', clean(row.get('이름', ''))),
        'department':       info.get('부서', clean(row.get('부서', ''))),
        'department_level': clean(row.get('부서레벨', '')) or info.get('부서레벨', ''),
        'job_grade':        info.get('직급', clean(row.get('직급', ''))),
        'job_grade_level':  clean(row.get('직급레벨', '')) or info.get('직급레벨', ''),
        'embedding_text':   build_embedding_text(row, info, source_name),
        'source':           source_csv,
        'changed':          [],
    }


def run_jsonl_conversion(dfs_clean, source_filenames):
    # 정제된 DataFrame들을 직원 레코드 리스트로 바꿔 소스별로 모아 반환한다.
    print('\n========== 2단계: JSONL 변환 ==========')

    # 기본인사정보에서 이름/부서/직급을 사원번호로 찾아 쓰기 위한 조회 딕셔너리
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

    records_by_source = {}
    for source_name, df in dfs_clean.items():
        source_csv = source_filenames.get(source_name, source_name)   # 원본 CSV 파일명
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
    # user_dictionary.txt와 synonym.txt를 모두 반영하므로 둘 중 하나만 바뀌어도 해시가 달라진다.
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
    # embedding_text는 한 줄에 하나씩 '필드명: 값' 형식 (줄바꿈으로 구분).
    # 줄 단위로 나눈 뒤 첫 ': ' 기준으로 필드명과 값을 분리한다.
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
    parts = []
    for field in fields:
        if parsed.get(field):
            parts.append(f'{field}: {parsed[field]}')
    return '\n'.join(parts)


def create_indices(client):
    # 인덱스를 생성한다. 사전(user_dictionary.txt / synonym.txt)이 바뀌면 새 사전으로
    # 다시 만들어야 검색에 반영되므로, 변경 감지 시 기존 인덱스를 자동으로 삭제하고 재생성한다.
    print('\n========== 인덱스 생성 ==========')

    current_hash = get_dict_hash()
    last_hash    = read_last_dict_hash()
    dict_changed = current_hash != last_hash

    # 사전이 바뀌었으면 기존 인덱스를 삭제하고 재생성한다.
    apply_change = dict_changed
    if dict_changed:
        existing = [name for name in INDEX_CONFIG if client.indices.exists(index=name)]
        if existing:
            print('  사용자 사전(user_dictionary.txt / synonym.txt) 변경 감지됨')
            print(f'  기존 인덱스 {len(existing)}개를 삭제하고 재생성합니다: {", ".join(existing)}')
            for name in existing:
                client.indices.delete(index=name)
                print(f'  삭제: {name}')

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


def get_existing_docs(client, index_name):
    # 이 인덱스에 이미 적재된 문서를 모두 읽어, 직원별로 옛 텍스트와 옛 변경이력을 모은다.
    # 직원 한 명에게 청크 문서가 여러 개일 수 있어, doc_id 순서대로 embedding_text를 이어 붙인다.
    # helpers.scan 은 인덱스의 모든 문서를 페이지 단위로 안전하게 훑어준다 (개수 제한 없음).
    if not client.indices.exists(index=index_name):
        return {}

    chunks_by_emp  = {}   # 사원번호 -> [(doc_id, embedding_text), ...]
    changed_by_emp = {}   # 사원번호 -> 옛 변경이력
    meta_by_emp    = {}   # 사원번호 -> 옛 꼬리표(메타데이터)

    # 텍스트뿐 아니라 꼬리표(이름·부서·직급·레벨)도 함께 읽어온다.
    # 꼬리표는 모든 인덱스 문서에 복사되므로, 이 인덱스 텍스트가 안 바뀌어도 꼬리표가
    # 바뀌면 감지해서 갱신해야 한다. (안 그러면 인덱스마다 부서·직급이 달라짐)
    source_fields = ['employee_id', 'embedding_text', 'changed',
                     'employee_name', 'department', 'department_level',
                     'job_grade', 'job_grade_level']
    search_body = {'_source': source_fields, 'query': {'match_all': {}}}
    for doc in helpers.scan(client, index=index_name, query=search_body):
        source = doc['_source']
        emp_id = source.get('employee_id', '')
        doc_id = doc['_id']
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

    result = {}
    for emp_id in chunks_by_emp:
        pairs = chunks_by_emp[emp_id]
        pairs.sort()   # doc_id 순서대로 정렬 (EMP0001_0001, EMP0001_0002 ...)
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
    # 문서에 저장되는 내용 전체(꼬리표 + 텍스트)를 비교용 한 문자열로 만든다.
    # 텍스트만 보지 않고 꼬리표까지 포함해야 부서·직급 같은 꼬리표만 바뀌어도 감지된다.
    parts = []
    for key, label, is_level in META_COMPARE_FIELDS:
        parts.append(normalize_meta_value(meta.get(key, ''), is_level))
    parts.append(text)
    return '\n'.join(parts)


def build_change_entry(old_meta, new_meta, old_text, new_text, now):
    # 꼬리표(메타)와 텍스트를 필드 단위로 비교해, 바뀐 필드 목록을 변경이력 한 건으로 만든다.
    # 해시를 쓰지 않고 'if 옛값 != 새값' 으로 직접 비교한다.
    changed_fields = []

    # 꼬리표(메타데이터) 비교
    for key, label, is_level in META_COMPARE_FIELDS:
        old_value = normalize_meta_value(old_meta.get(key, ''), is_level)
        new_value = normalize_meta_value(new_meta.get(key, ''), is_level)
        if old_value != new_value:
            changed_fields.append({'field': label, 'old': old_value, 'new': new_value})

    # 텍스트 필드 비교
    old_fields = parse_embedding_text(old_text)
    new_fields = parse_embedding_text(new_text)
    for field_name in new_fields:
        old_value = old_fields.get(field_name, '')
        new_value = new_fields[field_name]
        if str(old_value) != str(new_value):
            changed_fields.append({'field': field_name, 'old': str(old_value), 'new': str(new_value)})

    return {'timestamp': now, 'fields': changed_fields}


def run_indexing(records_by_source, model, client):
    # 인덱스별로 (필드 필터링 → 청킹 → 임베딩 → 변경감지 → 적재) 를 한 번에 수행한다.
    # 청킹이 이 함수 안에 들어와 있는 이유는 파일 상단 주석 참고.
    print('\n========== 3단계: 인덱싱 (인덱스별 청킹 + 변경감지) ==========')

    tokenizer = model.tokenizer

    # 이번 실행의 적재 시각. 실제로 저장하는 문서(신규·변경)에만 이 시각을 찍는다.
    indexed_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    indexing_errors = []   # 적재 실패 모음 (마지막에 error.log로 남긴다)
    chunking_errors = []   # 청킹 경고 모음 (마지막에 error.log로 남긴다)

    # 직원별로 모든 소스(기본인사정보 / 역량성과 / 급여정보)의 필드를 하나로 합친다.
    # 인덱스가 어느 소스 필드를 쓰든 한 곳에서 꺼낼 수 있게 만든다.
    # 동시에 필드별로 어느 원본 CSV에서 왔는지 추적해, 인덱스별 source 필드를 동적으로 결정한다.
    employees = {}        # 사원번호 -> {'meta': 레코드(이름/부서/직급 등), 'parsed': {필드: 값, ...}}
    field_to_source = {}  # 필드명 -> 원본 CSV 파일명 (예: '주민등록번호' -> '기본인사정보.csv')
    for source_name in records_by_source:
        for rec in records_by_source[source_name]:
            emp_id = rec.get('employee_id', '')
            parsed = parse_embedding_text(rec.get('embedding_text', ''))
            if emp_id not in employees:
                employees[emp_id] = {'meta': rec, 'parsed': {}}
            employees[emp_id]['parsed'].update(parsed)
            for field in parsed:
                if field not in field_to_source:
                    field_to_source[field] = rec.get('source', '')

    for index_name, config in INDEX_CONFIG.items():
        fields = config['fields']

        # 이 인덱스의 source 필드 값 = 인덱스 필드들이 온 원본 CSV 파일명.
        # 첫 번째로 매칭되는 필드의 출처를 사용한다 (한 인덱스의 필드는 모두 같은 CSV에서 옴).
        index_source = ''
        for field in fields:
            if field in field_to_source:
                index_source = field_to_source[field]
                break

        # (가) 이 인덱스에 이미 있는 직원별 '옛 텍스트' + '옛 변경이력'
        old_data = get_existing_docs(client, index_name)

        # (나) 인덱스별 필드 필터링 → 청킹
        # 이 인덱스에 들어갈 필드만 골라낸 뒤 그 텍스트를 MAX_TOKENS 기준으로 청킹한다.
        # → 이 인덱스의 필드끼리만 같은 청크 안에 모이게 된다.
        new_by_emp = {}   # 사원번호 -> {'meta': 레코드, 'chunk_texts': [청크텍스트, ...]}
        for emp_id in employees:
            info = employees[emp_id]
            filtered = build_filtered_text(info['parsed'], fields)
            if not filtered:
                continue
            normalized = normalize_text(filtered)
            if not normalized.strip():
                chunking_errors.append({
                    '사원번호': emp_id,
                    '사유':    '빈 embedding_text로 스킵',
                    '상세':    index_name,
                })
                continue
            chunk_texts = chunk_by_tokens(normalized, MAX_TOKENS, tokenizer)
            new_by_emp[emp_id] = {'meta': info['meta'], 'chunk_texts': chunk_texts}

            # 토큰 한계 초과 경고 (필드 하나가 MAX_TOKENS보다 길면 임베딩 시 뒷부분이 잘림)
            for chunk_text in chunk_texts:
                token_count = len(tokenizer.encode(chunk_text))
                if token_count > MAX_TOKENS:
                    chunking_errors.append({
                        '사원번호': emp_id,
                        '사유':    '토큰 한계 초과',
                        '상세':    f'{index_name}: {token_count}토큰 > 한계 {MAX_TOKENS} (임베딩 시 잘릴 수 있음)',
                    })

        # (다) 비교: 신규 / 변경 / 무변경으로 분류하고, 적재할 것만 추린다
        #      texts_to_embed 와 plan 은 같은 순서로 쌓아 임베딩 결과와 짝을 맞춘다.
        texts_to_embed = []
        plan           = []   # (사원번호, 메타, changed, 청크텍스트)
        update_emp_ids = []
        new_count      = 0
        changed_count  = 0
        skip_count     = 0

        for emp_id in new_by_emp:
            info = new_by_emp[emp_id]
            new_text = '\n'.join(info['chunk_texts'])   # 이 인덱스에서의 전체 텍스트
            new_meta = info['meta']                     # 청크 레코드(꼬리표 포함)

            if emp_id not in old_data:
                # 신규 직원 → 그냥 적재 (변경이력 없음)
                changed = []
                new_count += 1
            else:
                old_text = old_data[emp_id]['text']
                old_meta = old_data[emp_id]['meta']
                # 텍스트만 보지 않고 꼬리표(부서·직급·레벨 등)까지 포함해 비교한다.
                # 그래야 부서 변경처럼 이 인덱스 텍스트엔 없는 변화도 감지해 꼬리표를 갱신한다.
                if doc_signature(old_meta, old_text) == doc_signature(new_meta, new_text):
                    # 꼬리표·텍스트 모두 같으면 다시 임베딩하지 않고 건너뛴다
                    skip_count += 1
                    continue
                # 바뀐 직원 → 바뀐 필드를 기록해 옛 이력에 덧붙이고, 기존 청크는 뒤에서 삭제한다
                old_changed = old_data[emp_id]['changed']
                changed = old_changed + [build_change_entry(old_meta, new_meta, old_text, new_text, indexed_at)]
                update_emp_ids.append(emp_id)
                changed_count += 1

            for chunk_text in info['chunk_texts']:
                texts_to_embed.append(chunk_text)
                plan.append((emp_id, info['meta'], changed, chunk_text))

        if not plan:
            print(f'  [{index_name}] 변경 없음 (건너뜀)')
            continue

        # 바뀐 직원의 기존 청크는 먼저 삭제 (청크 수가 달라질 수 있으므로)
        if update_emp_ids:
            client.delete_by_query(
                index=index_name,
                body={'query': {'terms': {'employee_id': update_emp_ids}}},
                params={'refresh': 'true'},
            )

        # 텍스트 목록을 한꺼번에 임베딩 벡터로 변환 (texts_to_embed 와 같은 순서로 vectors 생성)
        vectors = model.encode(
            texts_to_embed, batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False, convert_to_numpy=True,
        )

        # OpenSearch 문서로 변환 (직원별로 1부터 청크 번호를 매겨 문서 id를 만든다)
        chunk_counter = {}
        actions = []
        for plan_item, vector in zip(plan, vectors):
            emp_id, meta, changed, chunk_text = plan_item
            chunk_counter[emp_id] = chunk_counter.get(emp_id, 0) + 1
            doc_id = f'{emp_id}_{chunk_counter[emp_id]:04d}'
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
                    'embedding_vector': vector.tolist(),
                },
            })

        success, failed = helpers.bulk(client, actions, raise_on_error=False)
        print(
            f'  [{index_name}]  '
            f'신규 {new_count}명 / 변경 {changed_count}명 / 무변경 {skip_count}명  →  '
            f'{success:,}건 적재 (실패 {len(failed)}건)'
        )
        if failed:
            print(f'    경고: {len(failed)}건 적재 실패 → {failed[0]}')
            # 실패 항목을 error.log에 남기려고 모은다.
            # helpers.bulk 의 실패 항목은 {동작이름: {_id, error, ...}} 형태라 안쪽 dict를 꺼낸다.
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
    # 1~3단계에서 나온 문제를 한 파일(error.log)에 단계별로 구분해 남긴다 (점검용 로그).
    # 단계마다 항목 성격이 달라 컬럼도 다르므로, 섹션 헤더로 나눈다.
    # 청킹은 3단계 인덱싱 안에서 일어나므로 청킹 경고와 적재 실패 모두 3단계 섹션에 들어간다.
    os.makedirs(ERROR_LOG_PATH.parent, exist_ok=True)
    with open(ERROR_LOG_PATH, 'w', encoding='utf-8-sig') as log_file:
        log_file.write('========== 1단계: 전처리 에러 ==========\n')
        columns = ['파일명', '행', '사원번호', '컬럼', '원본값', '사유']
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
    print('\n========== OpenSearch 연결 ==========')
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
