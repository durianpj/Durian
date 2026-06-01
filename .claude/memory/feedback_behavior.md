---
name: feedback-behavior
description: "작업 방식에 대한 피드백 (파일 생성, 설명 순서 등)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 52ee8692-ecea-4863-a326-12882b69efed
---

**임시 파일 만들지 말 것**

**Why:** 프로젝트 폴더에 .py 같은 임시 파일이 생기면 지저분하고 사용자가 불쾌해한다.

**How to apply:** 스크립트가 필요한 경우 파일을 저장하지 않고 bash 인라인 명령어나 한 줄 python -c로 처리한다.

---

**새 폴더/파일 생성 전에 먼저 설명할 것**

**Why:** 사용자가 원하지 않는 방향으로 작업이 진행되는 걸 막기 위해서다. 이전에 임베딩 폴더를 설명 없이 바로 만들려다 제지당했다.

**How to apply:** 새로운 파일이나 폴더를 만들기 전에 "이렇게 하려고 하는데 괜찮아?" 하고 먼저 확인한다.

---

**git push는 복잡하게 하지 말 것**

**Why:** git push가 이미 잘 작동하고 있는데 API나 스크립트로 우회하려 해서 사용자가 혼란스러워했다.

**How to apply:** .gitignore에 없는 파일은 그냥 git add → commit → push로 처리한다.
