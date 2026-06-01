---
name: project-key-decisions
description: 전처리 설계에서 내린 주요 기술 결정들
metadata: 
  node_type: memory
  type: project
  originSessionId: 52ee8692-ecea-4863-a326-12882b69efed
---

**파일별 독립 처리 구조**

기본인사정보·역량성과·급여정보 3개 CSV를 합치지 않고 각각 따로 처리해서 저장한다.

**Why:** 처음에 3개 파일을 concat하여 처리했더니 사원번호 중복 감지로 4,000행이 삭제되고, 파일마다 컬럼이 달라 전체 행이 제거되는 문제가 발생했다.

**How to apply:** 새 전처리 로직 추가 시 항상 `if '컬럼명' not in df.columns: return` 으로 컬럼 존재 여부를 먼저 확인한다.

---

**dtype=object로 CSV 읽기**

`pd.read_csv(path, encoding='utf-8-sig', dtype=object)`

**Why:** pandas 3.x에서 float64 컬럼에 문자열('미입력')을 대입하면 TypeError가 발생한다. dtype=object로 읽으면 모든 컬럼이 문자열로 처리되어 이 문제를 피할 수 있다.

---

**전역 상태 변수 파일마다 초기화**

`_errors`, `drop_rows`, `valid_rrn`, `valid_hire`를 루프 안에서 매 파일마다 새로 초기화한다.

**Why:** 파일 간 상태가 섞이면 앞 파일의 오류가 뒷 파일 처리에 영향을 준다.
