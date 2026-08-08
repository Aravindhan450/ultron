"""
Tests for structured output enforcement
(``ultron.core.intelligence.structured_output``).

Covers schema detection (formats, named schemas, ambiguous fields), the
deterministic JSON / XML / Markdown repairers, the conformance guarantee
(repair or explicit non-conformance — never silent deviation, never
fabricated content), the three registered tools, LOW-risk classification,
and enforcement inside both agents' reply pipelines (fake engine, no live
model needed).
"""

import asyncio
import json

import pytest

from ultron.core.agents import security as agent_security
from ultron.core.agents.react import ReActAgent
from ultron.core.agents.simple import handle_llm_fallback
from ultron.core.intelligence import structured_output as so
from ultron.core.tools.registry import get_tool
from ultron.core.types import Role
from ultron.security import SecurityBoundary
from ultron.security.models import Decision, RiskTier


class FakeEngine:
    """Scripted engine stub: returns a canned response, never calls a model."""

    def __init__(self, response: str = ""):
        self.response = response
        self.calls: list[list] = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return self.response


@pytest.fixture()
def interactive_mode(monkeypatch):
    monkeypatch.setattr(
        agent_security, "_boundary", SecurityBoundary(mode="interactive")
    )


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "user_input,fmt",
    [
        ("answer as json with fields name, age", "json"),
        ("reply in json with fields name, age", "json"),
        ("give me json output with fields name, age", "json"),
        ("respond in xml with elements title, body", "xml"),
        ("as a markdown table with columns name, score", "markdown"),
        ("as a table", "markdown"),
        ("table with columns name, score", "markdown"),
    ],
)
def test_detect_format_markers(user_input, fmt):
    spec = so.detect_schema_request(user_input)
    assert spec is not None
    assert spec.format == fmt


def test_detect_fields_with_types():
    spec = so.detect_schema_request("answer as json with fields name: string, age: number")
    assert spec.format == "json"
    names = {f.name: f.type for f in spec.fields}
    assert names == {"name": "string", "age": "number"}


def test_detect_ambiguous_fields_default_to_json():
    # The proposal's ambiguous-input case: fields but no explicit format.
    spec = so.detect_schema_request("answer with fields name, age, summary")
    assert spec is not None
    assert spec.format == "json"
    assert [f.name for f in spec.fields] == ["name", "age", "summary"]


def test_detect_named_schemas():
    assert so.detect_schema_request("write an analysis report").name == "analysis"
    assert so.detect_schema_request("as a comparison in json").name == "comparison"
    assert so.detect_schema_request("use the plan schema").name == "plan"


def test_detect_named_table_with_columns():
    spec = so.detect_schema_request("as a table with columns name, score")
    assert spec.format == "markdown"
    assert spec.columns == ("name", "score")


@pytest.mark.parametrize(
    "user_input",
    [
        "read config.json",
        "what is json",
        "run ls -la",
        "tell me a joke",
        "what time is it",
        "remember that I like coffee",
        "how heavy is pip install",
        "search the web for json tutorials",
    ],
)
def test_detect_negatives(user_input):
    assert so.detect_schema_request(user_input) is None


# ---------------------------------------------------------------------------
# JSON enforcement
# ---------------------------------------------------------------------------


def _json_spec():
    return so.SchemaSpec(
        format="json",
        fields=(so.FieldSpec("name"), so.FieldSpec("age", "number")),
    )


def test_clean_json_preserved():
    final, notes = so.enforce(_json_spec(), '{"name": "Ana", "age": 30}')
    assert notes == []
    assert json.loads(final) == {"name": "Ana", "age": 30}


def test_fenced_json_extracted():
    final, _ = so.enforce(
        _json_spec(), '```json\n{"name": "Ana", "age": 30}\n```'
    )
    assert json.loads(final) == {"name": "Ana", "age": 30}


def test_prose_wrapped_json_extracted():
    final, _ = so.enforce(
        _json_spec(), "Here you go: {'name': 'Ana', 'age': 30}"
    )
    assert json.loads(final) == {"name": "Ana", "age": 30}


def test_trailing_commas_removed():
    final, notes = so.enforce(_json_spec(), '{"name": "Ana", "age": 30,}')
    assert json.loads(final) == {"name": "Ana", "age": 30}
    assert not notes


def test_single_quotes_converted():
    final, _ = so.enforce(_json_spec(), "{'name': 'Ana', 'age': 30}")
    assert json.loads(final) == {"name": "Ana", "age": 30}


def test_unquoted_keys_repaired():
    final, _ = so.enforce(_json_spec(), "{name: 'Ana', age: 30}")
    assert json.loads(final) == {"name": "Ana", "age": 30}


def test_truncated_prefix_recovered():
    final, notes = so.enforce(_json_spec(), '{"name": "Ana", "age": 30}}')
    assert json.loads(final) == {"name": "Ana", "age": 30}
    assert any("truncated" in n for n in notes)


def test_cut_off_json_closed_by_unmatched_braces():
    # The common truncation: the model stopped mid-document, no closing brace.
    final, notes = so.enforce(_json_spec(), '{"name": "Ana", "age": 30')
    assert json.loads(final) == {"name": "Ana", "age": 30}
    assert any("unclosed brace" in n for n in notes)


def test_two_documents_prefix_recovery_drops_trailing():
    final, notes = so.enforce(_json_spec(), '{"name": "Ana"}{"age": 30}')
    # The first document is recovered; the required 'age' is then null-filled.
    assert json.loads(final) == {"name": "Ana", "age": None}
    assert any("trailing content dropped" in n for n in notes)


def test_list_field_type_validated():
    spec = so.SchemaSpec(
        format="json",
        fields=(so.FieldSpec("tags", "list"),),
    )
    ok, _ = so.validate(spec, '{"tags": ["a", "b"]}')
    assert ok
    _, issues2 = so.validate(spec, '{"tags": "not-a-list"}')
    assert any("tags" in i and "should be list" in i for i in issues2)


def test_missing_required_field_null_filled():
    final, notes = so.enforce(_json_spec(), '{"name": "Ana"}')
    data = json.loads(final)
    assert data["name"] == "Ana"
    assert data["age"] is None
    assert any("added null" in n and "age" in n for n in notes)


def test_type_mismatch_noted_kept():
    final, notes = so.enforce(_json_spec(), '{"name": "Ana", "age": "thirty"}')
    data = json.loads(final)
    assert data["age"] == "thirty"  # model's data never rewritten
    assert any("age" in n and "should be number" in n for n in notes)


def test_unexpected_field_noted_kept():
    final, notes = so.enforce(_json_spec(), '{"name": "Ana", "age": 30, "extra": 1}')
    data = json.loads(final)
    assert data["extra"] == 1
    assert any("unexpected field 'extra'" in n for n in notes)


def test_unsalvageable_json_explicit_failure():
    final, notes = so.enforce(_json_spec(), "this is not json at all")
    assert "could not be made to conform" in notes[0]
    assert final == "this is not json at all"


# ---------------------------------------------------------------------------
# XML enforcement
# ---------------------------------------------------------------------------


def _xml_spec():
    return so.SchemaSpec(
        format="xml",
        fields=(so.FieldSpec("title"), so.FieldSpec("body")),
    )


def test_xml_unclosed_tags_closed():
    final, notes = so.enforce(_xml_spec(), "<report><title>Q3</title><body>Up")
    import xml.etree.ElementTree as ET

    root = ET.fromstring(final)
    assert root.tag == "report"
    assert root.find("title").text == "Q3"
    assert root.find("body").text == "Up"
    assert any("closed unclosed tag" in n for n in notes)


def test_xml_stray_closing_dropped():
    final, notes = so.enforce(
        _xml_spec(), "<report><title>Q3</title></wrong><body>Up</body></report>"
    )
    import xml.etree.ElementTree as ET

    ET.fromstring(final)  # must be well-formed
    assert any("stray closing" in n for n in notes)


def test_xml_attribute_with_angle_bracket():
    final, _ = so.enforce(
        _xml_spec(),
        '<report><title lang="en>US">Q3</title><body>Up</body></report>',
    )
    import xml.etree.ElementTree as ET

    root = ET.fromstring(final)
    assert root.find("title").attrib == {"lang": "en>US"}
    assert root.find("title").text == "Q3"


def test_xml_missing_required_element_noted():
    _, notes = so.enforce(_xml_spec(), "<report><title>Q3</title></report>")
    assert any("missing required element 'body'" in n for n in notes)


# ---------------------------------------------------------------------------
# Markdown table enforcement
# ---------------------------------------------------------------------------


def _table_spec():
    return so.SchemaSpec(format="markdown", columns=("name", "score"))


def test_table_missing_separator_inserted():
    final, notes = so.enforce(_table_spec(), "| name | score |\n| Ana | 30 |")
    lines = final.splitlines()
    assert lines[0] == "| name | score |"
    assert "---" in lines[1]
    assert lines[2] == "| Ana | 30 |"
    assert not notes


def test_table_rows_padded_and_truncated():
    final, _ = so.enforce(
        _table_spec(),
        "| name | score |\n|---|---|\n| Ana |\n| Bob | 1 | 2 |",
    )
    lines = final.splitlines()
    assert lines[2] == "| Ana |  |"
    assert lines[3] == "| Bob | 1 |"


def test_table_missing_column_noted():
    _, notes = so.enforce(_table_spec(), "| name |\n|---|\n| Ana |")
    assert any("missing required column" in n and "score" in n for n in notes)


def test_table_missing_table_explicit_failure():
    _, notes = so.enforce(_table_spec(), "Just some prose, no table here.")
    assert any("no markdown table found" in n for n in notes)


# ---------------------------------------------------------------------------
# enforce_reply helper
# ---------------------------------------------------------------------------


def test_enforce_reply_no_schema_unchanged():
    assert so.enforce_reply("tell me a joke", "Why did the chicken…") == (
        "Why did the chicken…"
    )


def test_enforce_reply_repairs_broken_json():
    # The trailing comma + single quotes are repaired silently; the reply is
    # now guaranteed-parseable — something raw model output would not be.
    out = so.enforce_reply(
        "answer as json with fields name, age",
        "{'name': 'Ana', 'age': 30,}",
    )
    assert json.loads(out) == {"name": "Ana", "age": 30}


def test_schema_prompt_block_injected_only_when_requested():
    assert so.schema_prompt_block("tell me a joke") == ""
    block = so.schema_prompt_block("answer as json with fields name, age")
    assert "STRUCTURED OUTPUT" in block
    assert "name" in block


# ---------------------------------------------------------------------------
# Registered tools + security
# ---------------------------------------------------------------------------


def test_tools_registered():
    for name in ("enforce_schema", "schema_validate", "list_schemas"):
        assert callable(get_tool(name))


def test_tools_classified_low_risk():
    boundary = SecurityBoundary(mode="interactive")
    for name in ("enforce_schema", "schema_validate", "list_schemas"):
        assert boundary.classify_action(name, "json") == RiskTier.LOW
        assert boundary.check(name, "", "sample text").decision == Decision.ALLOW


def test_enforce_schema_tool():
    out = get_tool("enforce_schema")(
        '{"name": "Ana"}', format="json", fields="name, age"
    )
    data = json.loads(out.split("[structured]")[0].strip())
    assert data == {"name": "Ana", "age": None}


def test_schema_validate_tool():
    out = get_tool("schema_validate")(
        '{"name": "Ana"}', format="json", fields="name, age"
    )
    assert "does not conform" in out
    assert "age" in out


def test_schema_validate_conforms_branch():
    out = get_tool("schema_validate")(
        '{"name": "Ana", "age": 30}', format="json", fields="name, age"
    )
    assert "✓ conforms" in out


def test_list_schemas_tool():
    out = get_tool("list_schemas")()
    assert "analysis" in out
    assert "table" in out


# ---------------------------------------------------------------------------
# Agent reply-pipeline enforcement
# ---------------------------------------------------------------------------


def test_llm_fallback_enforces_broken_json(interactive_mode):
    engine = FakeEngine('{"name": "Ana", "age": 30,}')
    msg = run(
        handle_llm_fallback(
            "answer as json with fields name, age", [], engine
        )
    )
    assert msg.role == Role.ASSISTANT
    # Without enforcement the raw reply '{"name": "Ana", "age": 30,}'
    # would not parse; the pipeline guarantees it does.
    assert json.loads(msg.content) == {"name": "Ana", "age": 30}


def test_llm_fallback_enforces_xml(interactive_mode):
    engine = FakeEngine("<report><title>Q3</title><body>Up")
    msg = run(
        handle_llm_fallback(
            "answer in xml with elements title, body", [], engine
        )
    )
    assert "Q3" in msg.content
    assert "</body>" in msg.content


def test_llm_fallback_untouched_without_schema(interactive_mode):
    engine = FakeEngine("Just a friendly reply.")
    msg = run(handle_llm_fallback("tell me a joke", [], engine))
    assert msg.content == "Just a friendly reply."
    assert "[structured]" not in msg.content


def test_react_final_answer_enforced():
    engine = FakeEngine("{'name': 'Ana', 'age': 30,}")
    agent = ReActAgent(engine)
    msg = run(agent.run("answer as json with fields name, age"))
    assert json.loads(msg.content) == {"name": "Ana", "age": 30}


def test_react_untouched_without_schema():
    engine = FakeEngine("The answer is 42.")
    agent = ReActAgent(engine)
    msg = run(agent.run("what is 6 times 7"))
    assert msg.content == "The answer is 42."
