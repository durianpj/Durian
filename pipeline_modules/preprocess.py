# ══════════════════════════════════════════════════════════════════════════════
# 1단계 처리: 원본 CSV 검증·교정 (functions.py의 run_validations 호출)
# ══════════════════════════════════════════════════════════════════════════════
# 검증에 쓰는 상태(_errors 등)는 functions.py 안에 캡슐화돼 있고,
# run_validations(df)가 그 df의 (에러목록, 제거할 행번호)를 반환한다.
import pandas as pd
from pipeline_modules.functions import run_validations
from pipeline_modules.config import DATASET_DIR
from pipeline_modules.errors import PipelineError


def run_preprocessing():
    # 원본 CSV들을 읽어 컬럼별 검증·교정 후 정제된 DataFrame들을 메모리로 반환한다.
    print('\n========== 1단계: 전처리 ==========')
    print(f'입력 폴더: {DATASET_DIR}')

    csv_files = sorted(DATASET_DIR.glob('*.csv'))
    if not csv_files:
        raise PipelineError(f'CSV 파일 없음: {DATASET_DIR}')

    dfs = {}
    source_filenames = {}   # path.stem -> 원본 CSV 파일명 (source 필드에 사용)
    for path in csv_files:
        df = pd.read_csv(path, encoding='utf-8-sig', dtype=object)
        dfs[path.stem] = df
        source_filenames[path.stem] = path.name
        print(f'  로딩: {path.name}  ({len(df):,}행 / {len(df.columns)}열)')

    cleaned = {}
    all_errors = []

    for source_name, df in dfs.items():
        print(f'\n처리 중: {source_name}  ({len(df):,}행)')

        # 이 df의 검증 수행 → (에러목록, 제거할 행번호 집합) 반환
        errors, drop_rows = run_validations(df)

        for err in errors:
            err['파일명'] = source_name
        all_errors.extend(errors)
        print(f'  에러: {len(errors):,}건')

        df_clean = df.drop(index=list(drop_rows)).reset_index(drop=True)
        cleaned[source_name] = df_clean
        print(f'  정제 결과: {len(df_clean):,}행 (제거 {len(drop_rows)}행)')

    print(f'\n전처리 에러 누적: {len(all_errors):,}건')

    return cleaned, all_errors, source_filenames
