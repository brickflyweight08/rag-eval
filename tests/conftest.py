import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Tests marked ``llm`` need a real judge; skip them cleanly when no key is set."""
    if os.environ.get("RAG_EVAL_JUDGE_API_KEY"):
        return
    skip = pytest.mark.skip(reason="RAG_EVAL_JUDGE_API_KEY is not set (real LLM test)")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)
