import pandas as pd
from datetime import date
from pathlib import Path
import os
from dotenv import load_dotenv

print(f'pandas 버전: {pd.__version__}')
print('라이브러리 로딩 완료!')

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / '.env')

INPUT_DIR  = BASE_DIR / Path(os.getenv('INPUT_DIR',  'dataset'))
OUTPUT_DIR = BASE_DIR / Path(os.getenv('OUTPUT_DIR', 'output'))

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('경로 설정\n')
print(f'입력 디렉토리: {INPUT_DIR}')
print(f'출력 디렉토리: {OUTPUT_DIR}')

TODAY = date.today()

MIN_AGE,MAX_AGE= 18, 80
MIN_SALARY,MAX_SALARY= 20_000_000, 500_000_000
MIN_OVERTIME,MAX_OVERTIME= 0, 52
MIN_UNUSED_LEAVE,MAX_UNUSED_LEAVE = 0, 30
MIN_GPA,MAX_GPA= 0.0, 4.5
MIN_SCORE,MAX_SCORE= 0, 100
MIN_TOEIC,MAX_TOEIC= 0, 990

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

# ── 1. 데이터 로딩 ─────────────────────────────────────────────────────────────

csv_files = sorted(INPUT_DIR.glob('*.csv'))

if not csv_files:
    raise SystemExit(f'CSV 파일 없음: {INPUT_DIR}')

dfs = {}
for path in csv_files:
    df = pd.read_csv(path, encoding='utf-8-sig', dtype=object)
    dfs[path.stem] = df
    print(f'로딩: {path.name}  ({len(df):,}행 / {len(df.columns)}열)')

print(f'\n로딩 완료! 총 {len(dfs)}개 파일')

# ── 에러 로그 초기화 ───────────────────────────────────────────────────────────

_errors   = []
drop_rows = set()
valid_rrn  = {}
valid_hire = {}


def log(row, emp_id, column, original_value, reason):
    _errors.append({
        '행': row,
        '사원번호': emp_id,
        '컬럼': column,
        '원본값': original_value,
        '사유': reason
    })


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────

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


# ── 검증 함수 ─────────────────────────────────────────────────────────────────

def validate_empid(df):
    if '사원번호' not in df.columns:
        return
    seen_emp = {}
    for row, raw in df['사원번호'].items():
        if raw:
            emp_id = str(raw).strip()
        else:
            emp_id = ' '
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
                if hire_route == '경력' and not cell_value:
                    log(row, df.at[row, '사원번호'], col, cell_value, '채용경로=경력인데 결측')
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
        if retire_type and not retire_date_str and '퇴직일자' in df.columns:
            log(row, emp_id, '퇴직일자', retire_date_str, '퇴직구분 있는데 퇴직일자 없음')
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
                val = int(raw)
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
                continue
            if grade_val not in PERF_GRADES:
                log(row, emp_id, col, grade_val, '고정값 외')
                df.at[row, col] = '미입력'


def validate_qual(df):
    if 'TOEIC점수' in df.columns:
        for row, raw in df['TOEIC점수'].items():
            toeic_str = str(raw).strip() if raw else ''
            if not toeic_str or toeic_str in ('nan', 'NaN'):
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
            df.at[row, col] = ','.join(items)
    if '징계이력' in df.columns and '징계사유' in df.columns:
        for row in df.index:
            discipline_history = str(df.at[row, '징계이력']).strip()
            if discipline_history in ('nan', 'NaN'):
                discipline_history = ''
            discipline_reason = str(df.at[row, '징계사유']).strip() if df.at[row, '징계사유'] else ''
            if discipline_reason in ('nan', 'NaN'):
                discipline_reason = ''
            emp_id = df.at[row, '사원번호']
            if discipline_history and not discipline_reason:
                log(row, emp_id, '징계사유', discipline_reason, '징계이력 있는데 징계사유 없음')
                df.at[row, '징계사유'] = '미입력'
            elif not discipline_history and discipline_reason:
                log(row, emp_id, '징계사유', discipline_reason, '징계이력 없는데 징계사유 존재')
                df.at[row, '징계사유'] = ''


# ── 3. 파일별 처리 및 저장 ───────────────────────────────────────────────────────

saved_files = []
all_errors  = []

for source_name, df in dfs.items():
    print(f'\n처리 중: {source_name}  ({len(df):,}행)')

    _errors   = []
    drop_rows = set()
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

    out_path = OUTPUT_DIR / f'{source_name}_정제.csv'
    df_clean.to_csv(out_path, index=False, encoding='utf-8-sig')
    saved_files.append((out_path, len(df_clean)))
    print(f'  저장: {out_path.name}  ({len(df_clean):,}행 / 제거 {len(drop_rows)}행)')

error_path = OUTPUT_DIR / 'error.log'
pd.DataFrame(all_errors, columns=['파일명', '행', '사원번호', '컬럼', '원본값', '사유']).to_csv(
    error_path, index=False, encoding='utf-8-sig'
)
print(f'\n에러 로그: {error_path.name}  (총 {len(all_errors):,}건)')

print('=' * 60)
print('통합인사정보 데이터 전처리 완료')
print('=' * 60)
print('저장된 파일:')
for path, rows in saved_files:
    print(f'  {path.name}  ({rows:,}행)')
print('=' * 60)
