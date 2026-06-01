---
name: feedback-coding-style
description: 코드 작성 스타일에 대한 피드백 (초보자 친화적)
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 52ee8692-ecea-4863-a326-12882b69efed
---

모든 코드는 코딩을 막 배운 사람도 이해할 수 있는 수준으로 작성해야 한다.

**Why:** 팀원들이 코드를 직접 읽고 이해할 수 있어야 하기 때문이다.

**How to apply:**
- 정규식(`re` 모듈) 사용 금지 → `startswith`, `isdigit`, `split`, `count` 등 기본 문자열 메서드로 대체
- 타입 힌트 사용 금지
- 리스트 컴프리헨션 중첩, `dict.fromkeys` 등 비직관적 패턴 사용 금지
- `' '.join(seen)` 같은 짧은 표현보다 명시적인 for 루프 선호
- 변수명은 `s` 같은 한 글자보다 `cleaned` 같이 의미 있는 이름 사용
- 모든 셀에 한국어 주석으로 각 명령어 설명 추가
