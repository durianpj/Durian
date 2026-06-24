const API_BASE_URL = "/api";

function chatbot() {
  return {

    employeeId: "",
    savedEmployeeId: "",
    question: "",
    messages: [],
    loading: false,
    permissionLevel: null,
    toastMessage: null,

    applyEmployeeId() {
      if (!this.employeeId || !this.employeeId.trim()) {
        this.showToast("사번을 입력해주세요.");
        return;
      }
      const newId = this.employeeId.trim().toUpperCase();
      if (newId === this.savedEmployeeId) {
        this.showToast("이미 적용된 사번입니다.");
        return;
      }
      this.savedEmployeeId = newId;
      this.messages = [];
      this.permissionLevel = null;
      this.showToast("사번 " + this.savedEmployeeId + " 적용 완료");
    },

    async send() {
      if (!this.savedEmployeeId) {
        this.showToast("먼저 사번을 입력하고 적용해주세요.");
        return;
      }
      if (!this.question || !this.question.trim()) {
        this.showToast("질문을 입력해주세요.");
        return;
      }

      const userQuestion = this.question.trim();
      this.messages.push({ role: "user", text: userQuestion });
      this.question = "";
      this.scrollToBottom();

      await this.callBackend(userQuestion);
    },

    async retry(failedQuestion) {
      if (!failedQuestion) return;
      // 재시도 전에 이 질문의 에러 버블을 먼저 제거한다
      this.messages = this.messages.filter(
        msg => !(msg.role === "error" && msg.question === failedQuestion)
      );
      await this.callBackend(failedQuestion);
    },

    async callBackend(userQuestion) {
      this.loading = true;
      this.scrollToBottom();

      const startTime = Date.now();

      try {
        const response = await fetch(API_BASE_URL + "/rag-chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: userQuestion,
            employee_id: this.savedEmployeeId
          })
        });

        if (!response.ok) {
          const errorMessage = await this.parseErrorMessage(response);
          this.messages.push({ role: "error", text: errorMessage, question: userQuestion });
          return;
        }

        const data = await response.json();
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        this.messages.push({ role: "bot", text: data.answer, sources: data.sources || [], elapsed });

        if (data.permission && data.permission.permission_level) {
          this.permissionLevel = data.permission.permission_level;
        }

      } catch (err) {
        this.messages.push({
          role: "error",
          text: "서버에 연결할 수 없습니다. 네트워크 또는 백엔드 상태를 확인해주세요.",
          question: userQuestion
        });
      } finally {
        this.loading = false;
        this.scrollToBottom();
      }
    },

    async parseErrorMessage(response) {
      try {
        const data = await response.json();
        if (data.detail) return "오류: " + data.detail;
      } catch (e) {}

      if (response.status === 404) return "사용자 사번을 찾을 수 없습니다.";
      if (response.status === 400) return "요청이 올바르지 않습니다.";
      if (response.status >= 500) return "서버 내부 오류가 발생했습니다.";
      return "알 수 없는 오류가 발생했습니다.";
    },

    showToast(message) {
      this.toastMessage = message;
      setTimeout(() => { this.toastMessage = null; }, 2000);
    },

    scrollToBottom() {
      setTimeout(() => {
        const area = this.$refs.chatArea;
        if (area) area.scrollTop = area.scrollHeight;
      }, 50);
    },

    uniqueIndices(sources) {
      if (!sources) return [];
      const result = [];
      for (const src of sources) {
        if (src.index && !result.includes(src.index)) result.push(src.index);
      }
      return result;
    }

  };
}
