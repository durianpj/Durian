import json
from pathlib import Path
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

print('라이브러리 로딩 완료!')

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR.parent / '.env')

INPUT_DIR  = Path(os.getenv('INPUT_DIR',  str(BASE_DIR.parent / 'JSONL' / 'output')))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', str(BASE_DIR / 'output')))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 임베딩 모델 (embedding_indexer 와 동일)
EMBEDDING_MODEL = os.getenv('EMBED_MODEL_NAME', 'paraphrase-multilingual-MiniLM-L12-v2')

# 청크당 최대 토큰 수 (.env에서 관리)
MAX_TOKENS = int(os.environ['MAX_TOKENS'])

print(f'입력 디렉토리: {INPUT_DIR}')
print(f'출력 디렉토리: {OUTPUT_DIR}')
print(f'최대 토큰 수: {MAX_TOKENS}')

# ── 토크나이저 로딩 ────────────────────────────────────────────────────────────

print(f'\n임베딩 모델 로딩: {EMBEDDING_MODEL}')
try:
    model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer = model.tokenizer
except Exception as e:
    print(f'임베딩 모델 로딩 실패: {e}')
    raise SystemExit(1)


def count_tokens(text):
    # 텍스트의 토큰 수 계산 (special token 포함)
    return len(tokenizer.encode(text))


# ── 1. 데이터 로딩 ─────────────────────────────────────────────────────────────

jsonl_files = sorted(INPUT_DIR.glob('*.jsonl'))

if not jsonl_files:
    print(f'JSONL 파일 없음: {INPUT_DIR}')
    print('JSONL 변환 스크립트를 먼저 실행해 주세요.')
    raise SystemExit(1)

record_sets = {}

for path in jsonl_files:
    records = []
    try:
        f_handle = open(path, 'r', encoding='utf-8')
    except Exception as e:
        print(f'JSONL 파일 열기 실패: {path.name} → {e}')
        raise SystemExit(1)
    with f_handle as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as e:
                print(f'JSONL 파싱 실패: {path.name} → {e}')
                raise SystemExit(1)
            records.append(record)

    record_sets[path.stem] = records
    print(f'  로딩: {path.name}  ({len(records):,}건)')

print(f'\n로딩 완료! 총 {len(record_sets)}개 파일')

# ── 2. 텍스트 정규화 ───────────────────────────────────────────────────────────

def normalize_field(line):
    # 필드 줄 단위로 공백 정규화 (줄바꿈 구조 유지)
    words = line.strip().split()
    return ' '.join(words)


normalize_count = 0

for file_name, records in record_sets.items():
    for rec in records:
        original = rec.get('embedding_text', '')
        fields = [normalize_field(f) for f in original.split('\n') if f.strip()]
        normalized = '\n'.join(fields)
        if original != normalized:
            normalize_count += 1
        rec['embedding_text'] = normalized

print(f'\n정규화 완료! 적용된 레코드 수: {normalize_count:,}건')

# ── 3. 동적 청킹 ──────────────────────────────────────────────────────────────

def chunk_by_tokens(embedding_text, max_tokens):
    # 필드를 하나씩 누적하며 토큰 한계로 청크를 나눈다.
    #
    # [성능] 각 필드의 토큰 수를 미리 한 번씩만 계산해두고 누적 합으로 한계를 판단한다.
    #        예전에는 필드를 더할 때마다 누적된 전체 텍스트를 처음부터 다시 토큰화해서
    #        필드 수의 제곱에 가깝게 토큰화가 일어났는데, 이 방식은 필드당 한 번만 한다.
    fields = [f for f in embedding_text.split('\n') if f.strip()]

    # 각 필드를 special token 없이 토큰화한 길이를 미리 계산
    field_lengths = [len(tokenizer.encode(f, add_special_tokens=False)) for f in fields]
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

    # 마지막 청크 추가
    if current_fields:
        chunks.append('\n'.join(current_fields))

    return chunks


print('\n청크 분할 시작...\n')

chunked_sets = {}

empty_text_count = 0

for file_name, records in record_sets.items():
    chunks = []
    for rec in records:
        embedding_text = rec.get('embedding_text', '')
        # 빈 embedding_text 방어 로직
        if not embedding_text.strip():
            emp_id = rec.get('employee_id', '?')
            print(f'  경고: embedding_text 비어있음 → 사원 {emp_id} 스킵')
            empty_text_count += 1
            continue
        chunk_texts = chunk_by_tokens(embedding_text, MAX_TOKENS)
        for chunk_text in chunk_texts:
            chunk_record = {
                'employee_id':      rec.get('employee_id', ''),
                'employee_name':    rec.get('employee_name', ''),
                'department':       rec.get('department', ''),
                'department_level': rec.get('department_level', ''),
                'job_grade':        rec.get('job_grade', ''),
                'job_grade_level':  rec.get('job_grade_level', ''),
                'embedding_text':   chunk_text,
                'source':           rec.get('source', ''),
                'timestamp':        rec.get('timestamp', ''),
                'changed':          rec.get('changed', []),
            }
            chunks.append(chunk_record)

    chunked_sets[file_name] = chunks
    print(f'청크 분할 결과: [{file_name}]')
    print(f'  원본 레코드 수: {len(records):,}건')
    print(f'  생성된 청크 수: {len(chunks):,}건\n')

print(f'청킹 완료! (빈 embedding_text 스킵: {empty_text_count}건)')

# ── 4. 검증 ────────────────────────────────────────────────────────────────────

print('\n데이터 소실 여부 확인 중...')
print('-' * 50)

for file_name in record_sets:
    original_count = len(record_sets[file_name])
    chunk_count    = len(chunked_sets[file_name])
    status = '정상' if chunk_count >= original_count else '경고: 데이터 소실 발생'
    print(f'  {file_name}')
    print(f'  원본 {original_count:,}건 → 청크 {chunk_count:,}건  [{status}]\n')

print('-' * 50)

required_fields = ['employee_id', 'department', 'job_grade', 'embedding_text']

print('\n필수 필드 유지 확인 중...')
print('-' * 50)

for file_name, chunks in chunked_sets.items():
    missing_count = 0
    for chunk in chunks:
        for field in required_fields:
            if not chunk.get(field):
                missing_count += 1
                break
    status = '정상' if missing_count == 0 else f'누락 {missing_count}건'
    print(f'  {file_name}: [{status}]')

print('-' * 50)

print('\n토큰 한계 초과 여부 확인 중...')
print('-' * 50)

for file_name, chunks in chunked_sets.items():
    # 한계를 넘은 청크를 모은다.
    # 정상 청킹이면 0건이어야 한다. 0건이 아니라면 필드 하나가 MAX_TOKENS보다 길다는 뜻이고,
    # 그런 청크는 임베딩할 때 모델이 뒷부분을 잘라내(truncate) 정보가 일부 손실될 수 있다.
    over_chunks = []
    for chunk in chunks:
        token_count = count_tokens(chunk['embedding_text'])
        if token_count > MAX_TOKENS:
            over_chunks.append((chunk.get('employee_id', '?'), token_count))

    if not over_chunks:
        print(f'  {file_name}: [정상]')
    else:
        print(f'  {file_name}: [초과 {len(over_chunks)}건]')
        # 어느 사원의 청크가 한계를 넘었는지 안내한다 (해당 사원 데이터를 점검할 수 있도록)
        for emp_id, token_count in over_chunks:
            print(f'    경고: 사원 {emp_id} 청크 {token_count}토큰 > 한계 {MAX_TOKENS} → 임베딩 시 잘릴 수 있음')

print('-' * 50)

# ── 5. 결과 저장 ───────────────────────────────────────────────────────────────

print('\n결과 저장 중...\n')

for file_name, chunks in chunked_sets.items():
    out_path = OUTPUT_DIR / f'{file_name}.jsonl'

    with open(out_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    print(f'  저장: {out_path.name}  ({len(chunks):,}건)')

print('\n모든 파일 저장 완료!')
