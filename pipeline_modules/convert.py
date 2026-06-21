# ══════════════════════════════════════════════════════════════════════════════
# 2단계 처리: 정제된 DataFrame -> 직원별 레코드(dict) (functions.py의 to_record 사용)
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd
import pipeline_modules.functions as fn


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
    # 각 CSV(소스)의 DataFrame을 행 단위로 순회하며 fn.to_record()로 레코드를 만든다.
    # DataFrame.to_dict('records') 는 각 행을 {컬럼명: 값} 딕셔너리로 바꿔준다.
    records_by_source = {}
    for source_name, df in dfs_clean.items():
        source_csv = source_filenames.get(source_name, source_name)  # 원본 CSV 파일명
        records = []
        for row in df.to_dict('records'):
            record = fn.to_record(row, source_name, basic_lookup, source_csv)
            records.append(record)
        records_by_source[source_name] = records
        print(f'  변환: {source_name}  ({len(records):,}건)')

    return records_by_source
