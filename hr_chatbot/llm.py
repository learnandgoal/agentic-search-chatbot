"""Model adapter seam.

Section 8 of the design says "the existing GPT model adapter powers the agent node". That adapter was not
part of the document, so this module is the one place to plug it in: return any LangChain chat model that
supports ``bind_tools`` (tool calling). Everything else in the project depends only on that.
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel


def get_chat_model() -> BaseChatModel:
    kwargs: dict = {"timeout": 60, "max_retries": 2}
    if os.getenv("OPENAI_TEMPERATURE"):  # some newer models reject a custom temperature, so it is opt-in
        kwargs["temperature"] = float(os.environ["OPENAI_TEMPERATURE"])

    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            api_version=os.getenv("OPENAI_API_VERSION", "2024-10-21"),
            **kwargs,
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env (or export the variable), "
            "or edit hr_chatbot/llm.py to use your existing GPT adapter."
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1"), **kwargs)
