"""Regression tests for the OpenAI provider model registry (see #601)."""

from src.archi.providers.openai_provider import (DEFAULT_OPENAI_MODELS,
                                                 OpenAIProvider)


def _gpt5_entry():
    matches = [m for m in DEFAULT_OPENAI_MODELS if m.id == "gpt-5"]
    assert len(matches) == 1, "expected exactly one gpt-5 registry entry"
    return matches[0]


class TestGpt5ContextWindow:
    """The gpt-5 registry value must track the documented max input tokens."""

    def test_gpt5_context_window_matches_documented_max_input(self):
        # OpenAI API model docs for gpt-5: 400k total context window,
        # 272,000 max input tokens, 128,000 max output tokens.
        # context_window feeds the trimming budget in base_react.py
        # (max_prompt_tokens = context_window - 15% margin), so a stale
        # 128000 silently shrinks the budget by more than half.
        assert _gpt5_entry().context_window == 272000

    def test_provider_exposes_corrected_gpt5_value(self):
        provider = OpenAIProvider()
        models = {m.id: m for m in provider.list_models()}
        assert models["gpt-5"].context_window == 272000

    def test_all_default_entries_have_positive_int_context_windows(self):
        for model in DEFAULT_OPENAI_MODELS:
            assert isinstance(model.context_window, int)
            assert model.context_window > 0, model.id
