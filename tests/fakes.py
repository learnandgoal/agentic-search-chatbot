"""A scripted chat model so the graph can be tested without any API key."""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChat(BaseChatModel):
    """``script(messages, bound_tool_names, tool_choice, call_number) -> AIMessage``."""

    script: Any
    bound_names: list[str] = []
    tool_choice: Any = None
    log: list = []  # shared across bind_tools copies: one entry per model call

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        names = [getattr(t, "name", str(t)) for t in tools]
        return self.model_copy(update={"bound_names": names, "tool_choice": tool_choice})

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.log.append({"tools": list(self.bound_names), "tool_choice": self.tool_choice})
        ai = self.script(messages, self.bound_names, self.tool_choice, len(self.log))
        return ChatResult(generations=[ChatGeneration(message=ai)])


def call(name: str, call_id: str, **args) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def ai(*calls: dict, text: str = "") -> AIMessage:
    return AIMessage(content=text, tool_calls=list(calls))


def last_tool_json(messages: list[BaseMessage]) -> dict:
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            return json.loads(m.content)
    raise AssertionError("no tool message yet")


def make_llm(script: Callable) -> ScriptedChat:
    return ScriptedChat(script=script, log=[])
