import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv

print(f'pandas 버전: {pd.__version__}')
print('라이브러리 로딩 완료!')

BASE_DIR = Path(__file__).resolve().parent

load_dotenv()

INPUT_DIR  = Path(os.getenv('INPUT_DIR',  str(BASE_DIR.parent / 'Preprocessing' / 'output')))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', str(BASE_DIR / 'output')))

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'  입력 디렉토리: {INPUT_DIR}')
print(f'  출력 디렉토리: {OUTPUT_DIR}')

# ── 1. 데이터 로딩 ─────────────────────────────────────────────────────────────

csv_files = sorted(INPUT_DIR.glob('*.csv'))

if not csv_files:
    print(f'CSV 파일 없음: {INPUT_DIR}')
    print('전처리 스크립트를 먼저 실행해 주세요.')
    raise SystemExit(1)

dfs = {}
for path in csv_files:
    try:
        df = pd.read_csv(path, encoding='utf-8-sig', dtype=object)
    except Exception as e:
        print(f'CSV 파일 읽기 실패: {path.name} → {e}')
        raise SystemExit(1)
    dfs[path.stem] = df
    print(f'  로딩: {path.name}  ({len(df):,}행 / {len(df.columns)}열)')

basic_df = dfs.get('기본인사정보_정제', pd.DataFrame())

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

print(f'\n로딩 완료! 총 {len(dfs)}개 파일')
if not basic_lookup:
    print('경고: 기본인사정보가 없어 이름/부서/직급이 채워지지 않습니다.')
print(f'기본인사정보 조회 딕셔너리: {len(basic_lookup):,}건')

# ── 2. 변환 함수 정의 ──────────────────────────────────────────────────────────

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


def build_embedding_text(row, info):
    fields = [
        ('이름', info.get('이름', clean(row.get('이름', '')))),
        ('부서', info.get('부서', clean(row.get('부서', '')))),
        ('직급', info.get('직급', clean(row.get('직급', '')))),
    ]
    for field in EMBEDDING_FIELDS:
        val = clean(str(row.get(field, '')))
        if val:
            fields.append((field, val))

    seen_keys = []
    parts = []
    for key, val in fields:
        if val and key not in seen_keys:
            parts.append(f'{key}: {val}')
            seen_keys.append(key)

    return ' '.join(parts)


def to_record(row, source_name):
    emp_id = str(row.get('사원번호', '')).strip()
    info   = basic_lookup.get(emp_id, {})

    return {
        'employee_id':      emp_id,
        'employee_name':    info.get('이름', clean(row.get('이름', ''))),
        'department':       info.get('부서', clean(row.get('부서', ''))),
        'department_level': clean(row.get('부서레벨', '')) or info.get('부서레벨', ''),
        'job_grade':         info.get('직급', clean(row.get('직급', ''))),
        'job_grade_level':   clean(row.get('직급레벨', '')) or info.get('직급레벨', ''),
        'embedding_text':   build_embedding_text(row, info),
        'source':           source_name.replace('_정제', ''),
        'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'changed':          [],
        'embedding_vector': [],
    }


# ── 3. JSONL 변환 및 저장 ──────────────────────────────────────────────────────

saved_files = []

for source_name, df in dfs.items():
    out_path = OUTPUT_DIR / f'{source_name}.jsonl'

    old_records = {}
    if out_path.exists():
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    old_records[rec.get('employee_id', '')] = rec
        except Exception as e:
            print(f'기존 JSONL 파일 읽기 실패: {out_path.name} → {e}')
            raise SystemExit(1)

    rows = df.to_dict('records')

    with open(out_path, 'w', encoding='utf-8') as f:
        for row in rows:
            record = to_record(row, source_name)
            emp_id = record['employee_id']

            if emp_id in old_records:
                old = old_records[emp_id]

                record['changed'] = old.get('changed', [])

                skip = {'embedding_vector', 'timestamp', 'changed'}
                new_fields = []
                for field, new_val in record.items():
                    if field in skip:
                        continue
                    old_val = old.get(field, '')
                    if str(old_val) != str(new_val):
                        new_fields.append({
                            'field': field,
                            'old':   str(old_val),
                            'new':   str(new_val),
                        })

                if new_fields:
                    record['changed'].append({
                        'timestamp': record['timestamp'],
                        'fields':    new_fields,
                    })

            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    saved_files.append((out_path, len(rows)))

# ── 4. 저장 결과 확인 ──────────────────────────────────────────────────────────

for out_path, _ in saved_files:
    with open(out_path, 'r', encoding='utf-8') as f:
        sample = json.loads(f.readline())
    print(f'\n[{out_path.name}] 첫 번째 레코드:')
    print(json.dumps(sample, ensure_ascii=False, indent=2))
    print()

print('=== 저장 완료 ===')
total = 0
for out_path, count in saved_files:
    print(f'  {out_path.name}  ({count:,}건)')
    total += count
print(f'  총 {total:,}건')
