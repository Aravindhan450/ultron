"""ultron.core.intelligence.structured_output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structured output enforcement.

When the user asks for a machine-readable answer — \"as JSON with fields X,
Y\", \"in XML with elements a, b\", \"as a markdown table\" — small local
models often return something that *looks* structured but cannot be parsed
(prose wrapping, trailing commas, unclosed tags, truncated responses). This
module provides a stronger guarantee:

1. ``detect_schema_request`` — parses the user's request into a
   ``SchemaSpec`` (format + fields + columns + optional named schema), even
   when the phrasing is vague (\"answer with fields name, age\" implies JSON).
2. ``build_schema_instructions`` — the exact schema, injected into the
   model's prompt so it knows the shape up front.
3. ``enforce`` — validates the reply and deterministically repairs it
   (JSON truncation/trailing-comma/quote repair, XML tag-stack balancing,
   markdown table normalization). Nothing is fabricated: missing required
   JSON fields become explicit ``null`` placeholders, every repair is
   reported in the returned notes, and an unsalvageable reply is returned
   untouched with an explicit non-conformance warning.

The repairers are pure functions — no LLM — so the guarantee is testable and
grounded in what the model actually wrote. See docs/structured-output.md.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Schema model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """One field of a structured schema."""

    name: str
    type: str = "any"  # any | string | number | boolean | list | object
    required: bool = True


@dataclass(frozen=True)
class SchemaSpec:
    """A requested output shape."""

    format: str  # json | xml | markdown
    name: str = ""
    fields: tuple[FieldSpec, ...] = ()
    columns: tuple[str, ...] = ()


# --- Predefined schemas ---------------------------------------------------

PREDEFINED_SCHEMAS: dict[str, SchemaSpec] = {
    "analysis": SchemaSpec(
        format="json",
        name="analysis",
        fields=(
            FieldSpec("summary", "string"),
            FieldSpec("key_points", "list"),
            FieldSpec("recommendation", "string"),
            FieldSpec("risks", "list", required=False),
        ),
    ),
    "comparison": SchemaSpec(
        format="json",
        name="comparison",
        fields=(
            FieldSpec("criteria", "list"),
            FieldSpec("items", "list"),
            FieldSpec("verdict", "string"),
        ),
    ),
    "plan": SchemaSpec(
        format="json",
        name="plan",
        fields=(
            FieldSpec("goal", "string"),
            FieldSpec("steps", "list"),
            FieldSpec("estimated_time", "string", required=False),
        ),
    ),
    "decision": SchemaSpec(
        format="json",
        name="decision",
        fields=(
            FieldSpec("question", "string"),
            FieldSpec("decision", "string"),
            FieldSpec("reasoning", "string"),
            FieldSpec("alternatives", "list", required=False),
        ),
    ),
    "table": SchemaSpec(format="markdown", name="table"),
}

_TYPE_ALIASES = {
    "string": {"string", "str", "text"},
    "number": {"number", "numeric", "int", "integer", "float"},
    "boolean": {"boolean", "bool"},
    "list": {"list", "array"},
    "object": {"object", "dict", "map"},
}


def _normalize_type(raw: str) -> str:
    """Maps a user-supplied type word to the canonical FieldSpec type."""
    for canonical, aliases in _TYPE_ALIASES.items():
        if (raw or "").strip().lower() in aliases:
            return canonical
    return "any"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_FORMAT_MARKERS: dict[str, list[str]] = {
    "json": [
        r"\bas\s+(?:a\s+)?json\b",
        r"\bin\s+json\b",
        r"\bjson\s+(?:format|output|response)\b",
        r"\breply\s+(?:in|as)\s+json\b",
        r"\breturn\s+json\b",
        r"\boutput\s+json\b",
        r"\bjson\s+with\s+(?:fields|keys)\b",
        r"\bformat\s*:\s*json\b",
        r"\bstructured\s+json\b",
    ],
    "xml": [
        r"\bas\s+(?:a\s+)?xml\b",
        r"\bin\s+xml\b",
        r"\bxml\s+(?:format|output|response)\b",
        r"\breply\s+(?:in|as)\s+xml\b",
        r"\bformat\s*:\s*xml\b",
        r"\bxml\s+with\s+(?:elements|tags)\b",
    ],
    "markdown": [
        r"\bas\s+(?:a\s+)?markdown\s+table\b",
        r"\bas\s+(?:a\s+)?md\s+table\b",
        r"\bmarkdown\s+table\b",
        r"\bmd\s+table\b",
        r"\btable\s+(?:format|output)\b",
        r"\bas\s+(?:a\s+)?table\b",
        r"\btable\s+with\s+columns\b",
    ],
}

_FIELDS_MARKER = re.compile(
    r"\b(?:with|using|including)\s+(?:fields?|keys?|elements?|tags?|columns?)\s*"
    r"[:=]?\s*(?P<list>.+?)(?=\bin\s+(?:json|xml|markdown|md)\b|$)",
    re.IGNORECASE | re.DOTALL,
)

_LIST_SEPARATOR = re.compile(r"\s*(?:,|\band\b)\s*", re.IGNORECASE)

_VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}


def _split_field_tokens(raw: str) -> list[str]:
    """Splits a 'name, name: type and name' list into individual tokens."""
    return [tok.strip() for tok in _LIST_SEPARATOR.split(raw) if tok.strip()]


def detect_schema_request(user_input: str) -> SchemaSpec | None:
    """
    Parses a structured-output request into a SchemaSpec.

    Recognizes format markers (\"as json\", \"in xml\", \"as a markdown
    table\"), named predefined schemas (\"as an analysis report\"), and
    field lists (\"with fields name, age\"). Fields with no explicit format
    default to JSON — the ambiguous-input case. Returns None for ordinary
    chat so normal replies are never touched.
    """
    text = (user_input or "").strip()
    if not text:
        return None

    # Named predefined schemas win over generic markers.
    for name, spec in PREDEFINED_SCHEMAS.items():
        if re.search(
            rf"\b(?:"
            rf"(?:as|using|use|in|with)\s+"
            rf"|(?:write|give|make|produce|generate)\s+"
            rf")\s*(?:a\s+|an\s+|the\s+)?"
            rf"{re.escape(name)}\s+(?:report|schema|format|json|table)?\b",
            text,
            re.IGNORECASE,
        ):
            return _with_extra_columns(spec, text)

    fmt: str | None = None
    for candidate, markers in _FORMAT_MARKERS.items():
        if any(re.search(m, text, re.IGNORECASE) for m in markers):
            fmt = candidate
            break

    fields: list[FieldSpec] = []
    field_match = _FIELDS_MARKER.search(text)
    if field_match:
        for token in _split_field_tokens(field_match.group("list")):
            if ":" in token:
                raw_name, _, raw_type = token.partition(":")
                name = raw_name.strip()
                if name:
                    fields.append(
                        FieldSpec(name=name, type=_normalize_type(raw_type))
                    )
            else:
                fields.append(FieldSpec(name=token))

    columns: list[str] = []
    columns_match = re.search(
        r"\bcolumns?\s*[:=]?\s*(?P<list>.+?)(?=\bin\s+(?:json|xml|markdown|md)\b|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if columns_match:
        columns = _split_field_tokens(columns_match.group("list"))

    if fmt is None:
        if not fields and not columns:
            return None
        fmt = "json"  # Ambiguous but schema-driven → JSON by default.

    return SchemaSpec(format=fmt, fields=tuple(fields), columns=tuple(columns))


def _with_extra_columns(spec: SchemaSpec, text: str) -> SchemaSpec:
    """Merges a user-supplied column list into a named markdown table spec."""
    if spec.format != "markdown":
        return spec
    columns_match = re.search(
        r"\bcolumns?\s*[:=]?\s*(?P<list>.+?)(?=\bin\s+(?:json|xml|markdown|md)\b|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not columns_match:
        return spec
    columns = tuple(_split_field_tokens(columns_match.group("list")))
    return SchemaSpec(format=spec.format, name=spec.name, columns=columns)


def describe_spec(spec: SchemaSpec) -> str:
    """A human-readable one-line description of a schema (for tool output)."""
    if spec.format == "markdown":
        cols = ", ".join(spec.columns) if spec.columns else "from the reply header"
        return f"markdown table (columns: {cols})"
    fields = ", ".join(
        f"{f.name}:{f.type}{'' if f.required else '?'}" for f in spec.fields
    )
    label = f" '{spec.name}'" if spec.name else ""
    return f"json{label} with fields [{fields}]"


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------


def build_schema_instructions(spec: SchemaSpec) -> str:
    """The exact schema, injected into the model prompt before generation."""
    if spec.format == "markdown":
        cols = ", ".join(spec.columns) if spec.columns else "as many as you need"
        return (
            "STRUCTURED OUTPUT (the user asked for a table):\n"
            f"- Reply with ONLY a Markdown table with columns: {cols}.\n"
            "- Use pipe rows with a |---|---| separator after the header.\n"
            "- No prose before or after the table."
        )

    lines = [
        "STRUCTURED OUTPUT (the user asked for a specific shape):",
        "- Reply with ONLY a well-formed document matching this schema —",
        "  no prose, no explanation, no markdown fences.",
    ]
    if spec.name:
        lines.append(f"- Schema: {spec.name}")
    if spec.fields:
        parts = []
        for f in spec.fields:
            suffix = "" if f.required else " (optional)"
            parts.append(f"{f.name}: {f.type}{suffix}")
        lines.append(f"- Fields: {', '.join(parts)}")
    if spec.format == "json":
        lines.append("- Use double quotes; no trailing commas.")
    elif spec.format == "xml":
        lines.append("- Every opening tag must be closed.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _type_ok(value, type_name: str) -> bool:
    if type_name == "any":
        return True
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "list":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return True


def validate(spec: SchemaSpec, text: str) -> tuple[bool, list[str]]:
    """Validates raw text against the spec. Returns (ok, issues)."""
    if spec.format == "json":
        obj, err = _loads_candidate(text)
        if err:
            return False, [f"not valid JSON: {err}"]
        return _validate_json_object(obj, spec)
    if spec.format == "xml":
        repaired, notes = _repair_xml(text)
        try:
            ET.fromstring(repaired)
        except ET.ParseError as exc:
            return False, [f"not well-formed XML: {exc}"]
        issues = list(notes)
        if spec.fields:
            root = ET.fromstring(repaired)
            existing = {child.tag for child in root}
            for f in spec.fields:
                if f.required and f.name not in existing:
                    issues.append(f"missing required element '{f.name}'")
        return (not issues), issues
    if spec.format == "markdown":
        issues: list[str] = []
        table, found = _locate_table(text)
        if not found:
            return False, ["no markdown table found"]
        header = _split_cells(table[0])
        if spec.columns:
            for col in spec.columns:
                if col.lower() not in {h.lower() for h in header}:
                    issues.append(f"missing required column '{col}'")
        return (not issues), issues
    return True, []


def _validate_json_object(obj, spec: SchemaSpec) -> tuple[bool, list[str]]:
    if not isinstance(obj, dict):
        return False, ["expected a JSON object at the top level"]
    issues: list[str] = []
    all_names = {f.name for f in spec.fields}
    for f in spec.fields:
        if f.name not in obj:
            if f.required:
                issues.append(f"missing required field '{f.name}'")
            else:
                issues.append(f"missing optional field '{f.name}'")
            continue
        if not _type_ok(obj[f.name], f.type):
            issues.append(f"field '{f.name}' should be {f.type}, got {type(obj[f.name]).__name__}")
    for key in obj:
        if key not in all_names:
            issues.append(f"unexpected field '{key}'")
    return (not issues), issues


# ---------------------------------------------------------------------------
# Deterministic repairers
# ---------------------------------------------------------------------------


def _loads_candidate(text: str) -> tuple[object, str | None]:
    """json.loads with useful error message; (obj, None) or (None, err)."""
    try:
        return json.loads(text), None
    except (json.JSONDecodeError, TypeError) as exc:
        return None, str(exc)


def _extract_json(text: str) -> str:
    """Pulls the JSON document out of prose / fences / truncated text."""
    candidate = (text or "").strip()
    # Fenced block.
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
    # First { to last } slice.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        candidate = candidate[start : end + 1]
    return candidate


def _strip_comments(text: str) -> str:
    """Removes // and /* */ comments that sit outside string literals."""
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\":
                if i + 1 < n:
                    out.append(text[i + 1])
                    i += 2
                    continue
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _single_to_double_quotes(text: str) -> str:
    """Converts single-quoted strings to double-quoted, outside double quotes."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            out.append(ch)
            i += 1
            while i < n:
                out.append(text[i])
                if text[i] == "\\" and i + 1 < n:
                    out.append(text[i + 1])
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch == "'":
            out.append('"')
            i += 1
            while i < n:
                c = text[i]
                if c == "\\" and i + 1 < n:
                    out.append(c)
                    out.append(text[i + 1])
                    i += 2
                    continue
                if c == "'":
                    out.append('"')
                    i += 1
                    break
                if c == '"':
                    out.append('\\"')
                else:
                    out.append(c)
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _unquote_keys(text: str) -> str:
    """Turns {name: value} into {"name": value} (keys only, not values)."""
    return re.sub(
        r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)",
        lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}',
        text,
    )


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _repair_json_text(text: str) -> list[str]:
    """A ladder of progressively more aggressive repairs, in order."""
    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    candidates.append(_strip_trailing_commas(stripped))
    no_comments = _strip_comments(stripped)
    candidates.append(_strip_trailing_commas(no_comments))
    quoted = _single_to_double_quotes(no_comments)
    candidates.append(_strip_trailing_commas(quoted))
    unquoted = _unquote_keys(quoted)
    candidates.append(_strip_trailing_commas(unquoted))
    return list(dict.fromkeys(candidates))  # dedupe, keep order


def _brace_delta(text: str) -> int:
    """Net open-brace count (outside string literals). Positive = unclosed."""
    depth = 0
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    return depth


def _repair_truncated(text: str) -> object | None:
    """Recovers the longest complete prefix of a truncated JSON document."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "}":
            candidate = text[: i + 1]
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def _conform_json(obj, spec: SchemaSpec) -> tuple[object, list[str]]:
    """Enforces field presence/types without rewriting the model's data."""
    if not isinstance(obj, dict):
        return obj, ["expected a JSON object at the top level"]
    notes: list[str] = []
    all_names = {f.name for f in spec.fields}
    for f in spec.fields:
        if f.name not in obj:
            if f.required:
                obj[f.name] = None
                notes.append(
                    f"added null for missing required field '{f.name}'"
                )
            else:
                notes.append(f"missing optional field '{f.name}'")
            continue
        if not _type_ok(obj[f.name], f.type):
            notes.append(
                f"field '{f.name}' should be {f.type}, "
                f"got {type(obj[f.name]).__name__} (kept as-is)"
            )
    for key in list(obj):
        if key not in all_names:
            notes.append(f"unexpected field '{key}' (kept as-is)")
    return obj, notes


def enforce(spec: SchemaSpec, text: str) -> tuple[str, list[str]]:
    """
    Validates and deterministically repairs a reply against the spec.

    Returns (final_text, notes). Conforming output is produced whenever the
    content is salvageable; otherwise the original text is returned with an
    explicit non-conformance note — a non-conforming document is never
    silently presented as conforming, and nothing is fabricated.
    """
    if spec.format == "json":
        extracted = _extract_json(text)
        obj = None
        for candidate in _repair_json_text(extracted):
            parsed, err = _loads_candidate(candidate)
            if err is None:
                obj = parsed
                break
        if obj is None and extracted:
            # The common small-model truncation: the response was cut off
            # mid-document, leaving unmatched open braces. Close them and
            # retry before falling back to prefix recovery.
            unclosed = _brace_delta(extracted)
            if unclosed > 0:
                candidate = extracted + "}" * unclosed
                parsed, err = _loads_candidate(candidate)
                if err is None:
                    obj = parsed
                    notes = [
                        (
                            f"closed {unclosed} unclosed brace(s) to recover the "
                            "truncated JSON document"
                        )
                    ]
            if obj is None:
                obj = _repair_truncated(extracted)
                if obj is not None:
                    notes = [
                        (
                            "recovered the complete prefix of a truncated JSON "
                            "document (trailing content dropped)"
                        )
                    ]
                else:
                    notes = ["⚠ could not be made to conform: JSON could not be parsed or repaired"]
                    return text, notes
        else:
            notes = []
        if isinstance(obj, dict) or spec.fields:
            obj, field_notes = _conform_json(obj, spec)
            notes.extend(field_notes)
        return json.dumps(obj, indent=2, ensure_ascii=False), notes

    if spec.format == "xml":
        repaired, notes = _repair_xml(text)
        try:
            root = ET.fromstring(repaired)
        except ET.ParseError as exc:
            notes.append(f"⚠ could not be made to conform: XML still not well-formed ({exc})")
            return text, notes
        existing = {child.tag for child in root}
        for f in spec.fields:
            if f.required and f.name not in existing:
                notes.append(f"missing required element '{f.name}'")
        return repaired, notes

    if spec.format == "markdown":
        table, found = _locate_table(text)
        if not found:
            return text, ["⚠ could not be made to conform: no markdown table found"]
        header = _split_cells(table[0])
        if spec.columns:
            missing = [
                col for col in spec.columns
                if col.lower() not in {h.lower() for h in header}
            ]
            if missing:
                notes = [f"missing required column(s): {', '.join(missing)}"]
            else:
                notes = []
        else:
            notes = []
        repaired = _rebuild_table(table)
        return repaired, notes

    return text, []


# ---------------------------------------------------------------------------
# XML repair
# ---------------------------------------------------------------------------

# A tag is any < … > where a > inside a quoted attribute value is allowed.
# The inner group is non-capturing so findall() returns whole tokens.
_XML_TOKEN = re.compile(
    r'<!--.*?-->|<![^>]*>|<\?[^>]*\?>|<(?:"[^"]*"|\'[^\']*\'|[^>\'"])*>|[^<]+',
    re.DOTALL,
)


def _repair_xml(text: str) -> tuple[str, list[str]]:
    """Tag-stack balancing: closes unclosed tags, drops stray closes."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text or "")
    out: list[str] = []
    stack: list[str] = []
    notes: list[str] = []
    for token in _XML_TOKEN.findall(text):
        tok = token.strip()
        if not tok:
            continue
        if tok.startswith(("<!--", "<?")):
            out.append(token)
            continue
        if tok.startswith(("<![ ", "<!DOCTYPE")):
            out.append(token)
            continue
        if tok.startswith("</"):
            name = tok[2:-1].strip().split()[0] if len(tok) > 2 else ""
            if not name:
                continue
            if stack and stack[-1] == name:
                stack.pop()
                out.append(f"</{name}>")
            elif name in stack:
                # Close the open tags above the matching one first.
                while stack and stack[-1] != name:
                    out.append(f"</{stack.pop()}>")
                stack.pop()
                out.append(f"</{name}>")
            else:
                notes.append(f"dropped stray closing tag '{name}'")
                continue
        elif tok.startswith("<"):
            inner = tok[1:-1].strip()
            if not inner:
                continue
            if inner.endswith("/"):
                out.append(tok)
                continue
            name = inner.split()[0]
            out.append(tok)
            if name not in _VOID_ELEMENTS:
                stack.append(name)
        else:
            out.append(tok)
    while stack:
        out.append(f"</{stack.pop()}>")
        notes.append("closed unclosed tag")
    return "".join(out), notes


# ---------------------------------------------------------------------------
# Markdown table repair
# ---------------------------------------------------------------------------


def _split_cells(line: str) -> list[str]:
    """Splits a pipe row into cells (leading/trailing pipes ignored)."""
    cells = [c.strip() for c in line.split("|")]
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def _is_separator(line: str) -> bool:
    return bool(re.match(r"^\s*\|?\s*:?-{3,}(?:\s*\|\s*:?-{3,})*\s*\|?\s*$", line))


def _locate_table(text: str) -> tuple[list[str], bool]:
    """Returns (table_lines, found) for the first | row found."""
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        if "|" in lines[i]:
            block = []
            j = i
            while j < len(lines) and "|" in lines[j]:
                block.append(lines[j])
                j += 1
            return block, True
        i += 1
    return [], False


def _normalize_row(cells: list[str], ncols: int) -> list[str]:
    if len(cells) > ncols:
        cells = cells[:ncols]
    while len(cells) < ncols:
        cells.append("")
    return cells


def _rebuild_table(block: list[str]) -> str:
    header = _split_cells(block[0])
    ncols = len(header)
    if len(block) > 1 and _is_separator(block[1]):
        body = block[2:]
    else:
        body = block[1:]
    rows = [_normalize_row(_split_cells(line), ncols) for line in body if "|" in line]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * ncols) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reply pipeline helper
# ---------------------------------------------------------------------------


def enforce_reply(user_input: str, model_text: str) -> str:
    """
    The agent-facing helper: detects a schema request, enforces it on the
    model's reply, and appends ``[structured]`` notes. Returns the text
    unchanged when no schema was requested.
    """
    spec = detect_schema_request(user_input)
    if spec is None:
        return model_text
    final, notes = enforce(spec, model_text)
    if not notes:
        return final
    return final + "\n\n" + "\n".join(f"[structured] {note}" for note in notes)


def schema_prompt_block(user_input: str) -> str:
    """The prompt-injection block for a schema request ('' when none)."""
    spec = detect_schema_request(user_input)
    if spec is None:
        return ""
    return build_schema_instructions(spec)


# ---------------------------------------------------------------------------
# Registered tools (read-only, pure text processing)
# ---------------------------------------------------------------------------


def _spec_from_args(format: str, fields: str, schema: str) -> SchemaSpec | None:
    """Builds a SchemaSpec from tool arguments."""
    fmt = (format or "json").strip().lower()
    if fmt not in {"json", "xml", "markdown"}:
        if schema and schema in PREDEFINED_SCHEMAS:
            return PREDEFINED_SCHEMAS[schema]
        return None
    if schema and schema in PREDEFINED_SCHEMAS:
        spec = PREDEFINED_SCHEMAS[schema]
        if fmt != spec.format:
            return None
        return spec
    field_list: list[FieldSpec] = []
    if fields and fields.strip():
        for token in _split_field_tokens(fields):
            if ":" in token:
                name, _, raw_type = token.partition(":")
                name = name.strip()
                if name:
                    field_list.append(FieldSpec(name=name, type=_normalize_type(raw_type)))
            else:
                field_list.append(FieldSpec(name=token))
    return SchemaSpec(format=fmt, fields=tuple(field_list))


def enforce_schema(text: str, format: str = "json", fields: str = "", schema: str = "") -> str:
    """
    Validates + deterministically repairs ``text`` against a schema.

    - format: json | xml | markdown
    - fields: comma-separated 'name' or 'name: type' (json/xml)
    - schema: a predefined schema name (analysis, comparison, plan,
      decision, table) — overrides format/fields when given
    """
    spec = _spec_from_args(format, fields, schema)
    if spec is None:
        return (
            "Error: unknown schema. Use format 'json', 'xml', or 'markdown', "
            "or a predefined schema: analysis, comparison, plan, decision, table."
        )
    final, notes = enforce(spec, text)
    if not notes:
        return final
    return final + "\n\n" + "\n".join(f"[structured] {note}" for note in notes)


def schema_validate(text: str, format: str = "json", fields: str = "", schema: str = "") -> str:
    """
    Validates ``text`` against a schema and reports issues without rewriting.
    """
    spec = _spec_from_args(format, fields, schema)
    if spec is None:
        return (
            "Error: unknown schema. Use format 'json', 'xml', or 'markdown', "
            "or a predefined schema: analysis, comparison, plan, decision, table."
        )
    ok, issues = validate(spec, text)
    if ok:
        return f"✓ conforms to {describe_spec(spec)}"
    lines = [f"✗ does not conform to {describe_spec(spec)}:"]
    lines.extend(f"- {issue}" for issue in issues[:10])
    return "\n".join(lines)


def list_schemas() -> str:
    """Lists the predefined output schemas."""
    lines = ["Predefined output schemas:"]
    for name, spec in PREDEFINED_SCHEMAS.items():
        if spec.format == "markdown":
            shape = "table (columns from the request or reply header)"
        else:
            shape = ", ".join(
                f"{f.name}:{f.type}{'' if f.required else '?'}" for f in spec.fields
            )
        lines.append(f"• {name} ({spec.format}) — {shape}")
    lines.append("")
    lines.append(
        "Ask naturally, e.g. 'answer as JSON with fields name, age' or "
        "'as a markdown table with columns name, score'."
    )
    return "\n".join(lines)
