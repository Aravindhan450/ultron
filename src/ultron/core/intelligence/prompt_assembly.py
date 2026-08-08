"""
ultron.core.intelligence.prompt_assembly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

System prompt assembly — the single source of truth for the shared parts of
Ultron's prompts.

Currently provides two pieces:

- :func:`build_response_guidance` — the shared "well-mannered + structured"
  guidance block that every agent appends to its system prompt. The model
  follows it naturally; nothing is rewritten.
- :func:`polish_response` — light deterministic cleanup applied to the
  model's *natural-language* replies. Small local models can be terse or
  sloppy, so this strips stray whitespace, collapses blank-line runs, and
  guarantees a polite non-empty fallback — but it never edits the model's
  words.
"""


def build_response_guidance() -> str:
    """
    Returns the shared response-style guidance block.

    Appended to every system prompt (base chat prompt, simple agent tool
    instructions, ReAct agent) so replies are consistently well-mannered and
    well-structured regardless of which engine or model is active.
    """
    return (
        "RESPONSE STYLE (follow always):\n"
        "- Be polite, professional, and respectful — a courteous assistant.\n"
        "- Lead with the direct answer, then explain if that adds value.\n"
        "- Structure substantive answers with Markdown: a short intro line, "
        "clear section headings (##) and bullet lists where they help; use "
        "fenced code blocks for code.\n"
        "- Be concise and concrete; avoid filler, fluff, or restating the "
        "question.\n"
        "- If you are unsure or lack information, say so honestly instead of "
        "guessing.\n"
        "- Never invent tool results, file contents, or facts.\n"
    )


def polish_response(text: str) -> str:
    """
    Light deterministic cleanup for a model's natural-language reply.

    Strips stray leading/trailing whitespace, collapses runs of blank lines
    to at most one, and returns a polite fallback when the reply is empty.
    The model's words are never rewritten.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return (
            "I couldn't generate a response just now — please try rephrasing "
            "your request, or run the command again."
        )

    out: list[str] = []
    blanks = 0
    for line in cleaned.splitlines():
        line = line.rstrip()
        if not line:
            blanks += 1
            if blanks > 1:
                continue  # collapse blank-line runs to a single separator
        else:
            blanks = 0
        out.append(line)
    return "\n".join(out).strip()
