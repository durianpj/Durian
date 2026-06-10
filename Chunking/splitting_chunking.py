import json
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

print('라이브러리 로딩 완료!')

BASE_DIR = Path(__file__).resolve().parent

load_dotenv()

INPUT_DIR  = Path(os.getenv('INPUT_DIR',  str(BASE_DIR.parent / 'JSONL' / 'output')))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', str(BASE_DIR / 'output')))

os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE    = int(os.getenv('CHUNK_SIZE',    '200'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))

print(f'입력 디렉토리: {INPUT_DIR}')
print(f'출력 디렉토리: {OUTPUT_DIR}')
print(f'\n설정된 하이퍼파라미터:')
print(f'  청크 크기   : {CHUNK_SIZE}')
print(f'  오버랩      : {CHUNK_OVERLAP}')

# ── 1. 데이터 로딩 ─────────────────────────────────────────────────────────────

jsonl_files = sorted(
    f for f in INPUT_DIR.glob('*.jsonl')
    if f.name != 'changes_history.jsonl'
)

if not jsonl_files:
    print(f'JSONL 파일 없음: {INPUT_DIR}')
    print('JSONL 변환 스크립트를 먼저 실행해 주세요.')
    raise SystemExit(1)

doc_sets = {}
first_sample = None

for path in jsonl_files:
    docs = []
    try:
        f_handle = open(path, 'r', encoding='utf-8')
    except Exception as e:
        print(f'JSONL 파일 열기 실패: {path.name} → {e}')
        raise SystemExit(1)
    with f_handle as f:
        for line in f:
            try:
                record = json.loads(line.strip())
            except Exception as e:
                print(f'JSONL 파싱 실패: {path.name} → {e}')
                raise SystemExit(1)

            doc = Document(
                page_content=record['embedding_text'],
                metadata={
                    'employee_id':      record['employee_id'],
                    'employee_name':    record.get('employee_name', ''),
                    'department':       record['department'],
                    'department_level': record.get('department_level', ''),
                    'job_grade':         record['job_grade'],
                    'job_grade_level':   record.get('job_grade_level', ''),
                    'embedding_vector': record['embedding_vector'],
                    'source':           record.get('source', ''),
                    'timestamp':        record.get('timestamp', ''),
                    'changed':          record.get('changed', []),
                }
            )
            docs.append(doc)

    doc_sets[path.stem] = docs
    print(f'  로딩: {path.name}  ({len(docs):,}건)')

    if first_sample is None:
        first_sample = docs[0]

print(f'\n로딩 완료! 총 {len(doc_sets)}개 파일')

# ── 2. 텍스트 정규화 ───────────────────────────────────────────────────────────

def clean(text):
    text = text.strip()
    words = text.split()
    return ' '.join(words)


normalize_count = 0

for file_name, docs in doc_sets.items():
    for doc in docs:
        original   = doc.page_content
        normalized = clean(original)
        if original != normalized:
            normalize_count += 1
        doc.page_content = normalized

print(f'\n정규화 완료! 적용된 레코드 수: {normalize_count:,}건')

# ── 3. 스플릿 및 청킹 ─────────────────────────────────────────────────────────

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=['\n\n', '\n', '.', '!', '?', ',', ' ', '']
)

print('\n텍스트 청크 분할 시작...\n')

chunked_sets = {}

for file_name, docs in doc_sets.items():
    chunks = text_splitter.split_documents(docs)
    chunked_sets[file_name] = chunks

    print(f'청크 분할 결과: [{file_name}]')
    print(f'  원본 레코드 수: {len(docs):,}건')
    print(f'  생성된 청크 수: {len(chunks):,}건\n')

print('청킹 완료!')

# ── 4. 검증 ────────────────────────────────────────────────────────────────────

print('\n데이터 소실 여부 확인 중...')
print('-' * 50)

for file_name in doc_sets:
    original_count = len(doc_sets[file_name])
    chunk_count    = len(chunked_sets[file_name])
    status = '정상' if chunk_count >= original_count else '경고: 데이터 소실 발생'
    print(f'  {file_name}')
    print(f'  원본 {original_count:,}건 → 청크 {chunk_count:,}건  [{status}]\n')

print('-' * 50)

required_meta = ['employee_id', 'department', 'job_grade', 'embedding_vector']

print('\n메타데이터 유지 확인 중...')
print('-' * 50)

for file_name, chunks in chunked_sets.items():
    missing_count = 0
    for chunk in chunks:
        for field in required_meta:
            if field not in chunk.metadata:
                missing_count += 1
                break
    status = '정상' if missing_count == 0 else f'누락 {missing_count}건'
    print(f'  {file_name}: [{status}]')

print('-' * 50)

# ── 5. 결과 저장 ───────────────────────────────────────────────────────────────

print('\n결과 저장 중...\n')

for file_name, chunks in chunked_sets.items():
    out_path = OUTPUT_DIR / f'{file_name}.jsonl'

    with open(out_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            record = {
                'employee_id':      chunk.metadata['employee_id'],
                'employee_name':    chunk.metadata.get('employee_name', ''),
                'department':       chunk.metadata['department'],
                'department_level': chunk.metadata.get('department_level', ''),
                'job_grade':         chunk.metadata['job_grade'],
                'job_grade_level':   chunk.metadata.get('job_grade_level', ''),
                'embedding_text':   chunk.page_content,
                'embedding_vector': chunk.metadata['embedding_vector'],
                'source':           chunk.metadata.get('source', ''),
                'timestamp':        chunk.metadata.get('timestamp', ''),
                'changed':          chunk.metadata.get('changed', []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f'  저장: {out_path.name}  ({len(chunks):,}건)')

print('\n모든 파일 저장 완료!')
