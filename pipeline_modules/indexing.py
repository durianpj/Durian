# ══════════════════════════════════════════════════════════════════════════════
# 3단계 처리: 인덱스 생성 · 인덱스별 청킹/변경감지/적재 (functions.py 헬퍼 사용)
# ══════════════════════════════════════════════════════════════════════════════
from datetime import datetime
from opensearchpy import helpers
import pipeline_modules.functions as fn
from pipeline_modules.config import INDEX_CONFIG, MAX_TOKENS, EMBEDDING_BATCH_SIZE
from pipeline_modules.errors import PipelineError


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

    current_hash = fn.get_dict_hash()       # 현재 사전 파일의 해시값
    last_hash    = fn.read_last_dict_hash() # 지난번 실행 때 저장한 해시값
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
            client.indices.create(index=name, body=fn.build_index_body(config['required_level']))
            print(f'  생성 완료: {name}  (required_level={config["required_level"]})')
        except Exception as error:
            print(f'  인덱스 생성 실패: {name}  → {error}')
            raise PipelineError('파이프라인을 중단합니다. 위 메시지를 확인해주세요.')

    # 사전이 변경됐고 인덱스 재생성도 완료됐다면, 이번 해시값을 파일에 저장한다.
    # 다음 실행 때 사전이 또 바뀌었는지 비교하기 위해서다.
    if apply_change:
        fn.write_last_dict_hash(current_hash)


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
            parsed = fn.parse_embedding_text(rec.get('embedding_text', ''))
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
        old_data = fn.get_existing_docs(client, index_name)

        # (나) 각 직원에 대해 이 인덱스에 필요한 필드만 골라 청킹한다.
        # employees에는 모든 소스의 필드가 합쳐져 있으므로,
        # build_filtered_text로 이 인덱스 필드만 추출한 뒤 청킹한다.
        new_by_emp = {}   # 사원번호 -> {'meta': 레코드, 'chunk_texts': [청크텍스트, ...]}
        for emp_id in employees:
            info = employees[emp_id]
            # 이 인덱스에 필요한 필드만 골라 텍스트를 만든다
            filtered = fn.build_filtered_text(info['parsed'], fields)
            if not filtered:
                # 이 직원에게 이 인덱스의 필드 데이터가 없으면 건너뛴다
                continue
            normalized = fn.normalize_text(filtered)
            if not normalized.strip():
                chunking_errors.append({
                    '사원번호': emp_id,
                    '사유':    '빈 embedding_text로 스킵',
                    '상세':    index_name,
                })
                continue
            # MAX_TOKENS 기준으로 필드를 여러 청크로 나눈다
            chunk_texts = fn.chunk_by_tokens(normalized, MAX_TOKENS, tokenizer)
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
                if fn.doc_signature(old_meta, old_text) == fn.doc_signature(new_meta, new_text):
                    # 아무것도 바뀌지 않았으면 임베딩·적재를 생략한다 (비용 절약)
                    skip_count += 1
                    continue
                # 뭔가 바뀐 직원 → 변경이력을 기존 이력 뒤에 덧붙인다
                old_changed = old_data[emp_id]['changed']
                changed = old_changed + [fn.build_change_entry(old_meta, new_meta, old_text, new_text, indexed_at)]
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
