# ══════════════════════════════════════════════════════════════════════════════
# 데이터 파이프라인 진입점
# ══════════════════════════════════════════════════════════════════════════════
# 이 파일은 '호출 흐름'과 '최상단 예외 처리'만 담당한다.
#   함수 정의(def)        -> pipeline_modules/functions.py
#   단계별 처리 코드        -> pipeline_modules/preprocess, convert, indexing
#   설정·상수 / 예외        -> pipeline_modules/config, errors
#
# 단계 사이의 데이터는 파일이 아니라 메모리(변수)로 직접 넘긴다.
from datetime import datetime

from pipeline_modules.errors import PipelineError, PipelineStop
from pipeline_modules.functions import (
    connect_opensearch, load_embedding_model,
    ensure_user_dictionary, ensure_nori_plugin, write_error_log,
)
from pipeline_modules.preprocess import run_preprocessing
from pipeline_modules.convert import run_jsonl_conversion
from pipeline_modules.indexing import create_indices, run_indexing


def main():
    # 실행 순서:
    #   1) OpenSearch 연결 확인
    #   2) 사용자 사전 복사 -> nori 플러그인 확인(없으면 설치 후 재시작 안내하며 정상 종료)
    #   3) 임베딩 모델 로딩 -> 인덱스 생성
    #   4) 1단계(전처리) -> 2단계(변환) -> 3단계(인덱싱)
    #   5) 에러 로그 저장
    client = connect_opensearch()

    ensure_user_dictionary()
    ensure_nori_plugin(client)

    model = load_embedding_model()
    create_indices(client)

    dfs_clean, preprocessing_errors, source_filenames = run_preprocessing()
    records_by_source = run_jsonl_conversion(dfs_clean, source_filenames)
    indexing_errors, chunking_errors = run_indexing(records_by_source, model, client)

    write_error_log(preprocessing_errors, chunking_errors, indexing_errors)

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n========== 전체 완료 ({now}) ==========')


if __name__ == '__main__':
    try:
        main()
    except PipelineStop as stop:
        print(f'\n{stop}')
    except PipelineError as error:
        print(f'\n파이프라인 중단:\n{error}')
        raise SystemExit(1)
