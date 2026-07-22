"""
记忆管理（ConversationBufferMemory）
对应成品库② · 多轮记忆
"""

from typing import List, Dict


class ConversationMemory:
    """对话记忆管理器"""

    def __init__(self, max_turns: int = 6, max_tokens: int = 4000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.history: List[Dict] = []
        self.current_trace: List[str] = []

    def add_user_message(self, content: str):
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.history.append({"role": "assistant", "content": content})

    def add_trace_step(self, thought: str, action: str, observation: str):
        self.current_trace.append(f"思考: {thought}\n行动: {action}\n观察: {observation[:200]}")

    def get_context_messages(self) -> List[Dict]:
        """只返回历史对话摘要（丢弃中间 trace）"""
        return self.history[-self.max_turns * 2:]

    def finish_turn(self):
        """压缩 trace 并截断超长历史"""
        if self.current_trace:
            summary = "\n".join(self.current_trace[:3])
            if self.history and self.history[-1]["role"] == "assistant":
                self.history[-1]["content"] += f"\n[步骤摘要]\n{summary}"
            self.current_trace = []

        total = sum(len(m.get("content", "")) for m in self.history)
        while total > self.max_tokens and len(self.history) > 4:
            total -= len(self.history.pop(0))

    def clear(self):
        self.history = []
        self.current_trace = []
