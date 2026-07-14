import logging

logger = logging.getLogger(__name__)


def simple_llm_call(llm_client, model: str, messages: list, **kwargs) -> str:
    """Simple LLM call: send messages, return response text. Returns empty string on failure.

    ``**kwargs`` are forwarded to ``chat()``/``chat_with_tools()`` so callers can
    pass provider-specific toggles such as ``thinking_mode=False`` (GLM-5.2 thinks
    by default; reasoning is wasted on structured JSON tasks like decomposition).
    """
    try:
        if hasattr(llm_client, "chat"):
            resp = llm_client.chat(messages=messages, model=model, **kwargs)
            return getattr(resp, "content", "") or ""
        if hasattr(llm_client, "chat_with_tools"):
            resp = llm_client.chat_with_tools(messages=messages, tools=[], model=model, **kwargs)
            return getattr(resp, "content", "") or ""
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
    return ""
