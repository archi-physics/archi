"""Tests for base_react.normalize_answer_text (Responses-API block unwrap)."""

from src.archi.pipelines.agents.base_react import normalize_answer_text


def test_plain_string_passthrough():
    assert normalize_answer_text("A plain answer.") == "A plain answer."


def test_list_of_blocks():
    blocks = [
        {"type": "reasoning", "summary": []},
        {"type": "text", "text": "The tape RSE is T1_RU_JINR_Tape."},
    ]
    assert normalize_answer_text(blocks) == "The tape RSE is T1_RU_JINR_Tape."


def test_stringified_block_list():
    answer = str(
        [
            {"type": "reasoning", "summary": []},
            {"type": "text", "text": "42 datasets."},
        ]
    )
    assert normalize_answer_text(answer) == "42 datasets."


def test_concatenated_dict_reprs():
    answer = "{'type': 'reasoning', 'summary': []} {'type': 'text', 'text': 'Done.'}"
    assert normalize_answer_text(answer) == "Done."


def test_output_text_type():
    assert (
        normalize_answer_text([{"type": "output_text", "text": "hello"}]) == "hello"
    )


def test_unparseable_blob_returned_verbatim():
    blob = "[{not valid python or json"
    assert normalize_answer_text(blob) == blob


def test_no_text_blocks_returns_original():
    blocks = [{"type": "reasoning", "summary": []}]
    assert normalize_answer_text(blocks) == blocks
