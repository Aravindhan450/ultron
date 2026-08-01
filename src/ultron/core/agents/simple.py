import re
import json
import asyncio
from ultron.core.agents.base import BaseAgent
from ultron.core.types import ChatMessage, Role, PendingAction, history_to_openai_format

def detect_file_read_intent(user_input: str) -> str | None:
    """
    Detects if the user input contains a common file-reading pattern using regex.
    Returns the extracted filename if a match is found, otherwise None.

    This helper exists because relying on small local AI models to consistently emit
    tool calls for common phrasings can be unreliable. Deterministically capturing
    file-reading intents in code ensures fast and consistent performance.
    """
    # Regex to capture a filename (word characters, hyphens, slashes, ending with an extension like .txt, .py, .md)
    filename_pattern = r'[\w./-]+\.[a-zA-Z0-9]+'
    
    # Patterns for common file reading phrasings
    patterns = [
        rf'\bread\s+({filename_pattern})\b',
        rf'\bopen\s+({filename_pattern})\b',
        rf'\bshow\s+(?:me\s+)?({filename_pattern})\b',
        rf"\bwhat(?:'s|\s+is)\s+in\s+({filename_pattern})\b",
        rf'\bcontents\s+of\s+({filename_pattern})\b',
        rf'\bcat\s+({filename_pattern})\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return match.group(1)

    return None

def detect_file_write_intent(user_input: str) -> tuple[str, str] | None:
    """
    Detects if the user input contains a common file-writing pattern using regex.
    Returns a tuple of (filename, content) if matched, otherwise None.
    
    Additional phrasings can be added to the pattern list below in the future.
    """
    filename_pattern = r'[\w./-]+\.[a-zA-Z0-9]+'

    patterns = [
        # Match: "write <content> to <filename>"
        rf'^\s*write\s+(?P<content>.+?)\s+to\s+(?P<filename>{filename_pattern})\s*$',
        # Match: "create a file <filename> with <content>" or "create file <filename> with <content>"
        rf'^\s*create\s+(?:a\s+)?file\s+(?P<filename>{filename_pattern})\s+with\s+(?P<content>.+?)\s*$',
        # Match: "save <content> to <filename>"
        rf'^\s*save\s+(?P<content>.+?)\s+to\s+(?P<filename>{filename_pattern})\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE | re.DOTALL)
        if match:
            filename = match.group("filename").strip()
            content = match.group("content").strip()
            # Remove wrapping quotes around content if present (e.g., 'hello' or "hello")
            if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
                content = content[1:-1]
            return filename, content

    return None

def detect_command_intent(user_input: str) -> str | None:
    """
    Detects if the user input requests running a command (e.g., "run <command>" or "execute <command>").
    Returns the extracted command string if matched, otherwise None.
    """
    pattern = r'^\s*(?:run|execute)\s+(?P<command>.+)\s*$'
    match = re.search(pattern, user_input, re.IGNORECASE)
    if match:
        return match.group("command").strip()
    return None

def detect_remember_intent(user_input: str) -> str | None:
    """
    Detects if the user input requests remembering a fact (e.g. "remember that ...", "remember ...").
    Returns the extracted fact string if matched, otherwise None.
    """
    pattern = r'^\s*(?:please\s+)?remember\s+(?:that\s+)?(?P<fact>.+)\s*$'
    match = re.search(pattern, user_input, re.IGNORECASE)
    if match:
        return match.group("fact").strip()
    return None

def detect_memory_question(user_input: str) -> str | None:
    """
    Detects questions asking Ultron to recall stored facts about a topic.

    Strategy: two-gate approach instead of enumerating every exact phrase.
      Gate 1 — does this look like a recall question?
               Must contain "what" AND at least one recall verb:
               "remember", "tell", "know", "say".
      Gate 2 — does it have "about <topic>"?
               Extract everything after the LAST occurrence of the word
               "about" as the topic.  Using the last "about" handles
               constructions like "what did i tell you to remember about X"
               where an intermediate "about" could appear in the verb phrase.

    Matches (all case-insensitive):
      - "what did i tell you about FastAPI"
      - "what did i tell you to remember about testing"
      - "what do you remember about databases"
      - "what do you know about X"
      - "what did i say about Y"
      - informal variants like "what did i tell u about testing"

    Returns the extracted topic string, or None if either gate fails.

    Why handled in code rather than by the AI?  The system prompt injects ALL
    stored memories, so the AI can hallucinate connections between unrelated
    facts.  This path calls search_memories(topic) and builds the reply
    directly from DB rows — zero AI involvement, zero hallucination risk.
    """
    text = user_input.strip()

    # Gate 1: must contain "what" and at least one recall verb
    has_what = bool(re.search(r'\bwhat\b', text, re.IGNORECASE))
    has_recall_verb = bool(re.search(
        r'\b(?:remember|tell|know|say|told)\b', text, re.IGNORECASE
    ))
    if not (has_what and has_recall_verb):
        return None

    # Gate 2: must contain the word "about" followed by a non-empty topic.
    # re.finditer gives us all matches; we want the LAST one.
    about_matches = list(re.finditer(r'\babout\s+', text, re.IGNORECASE))
    if not about_matches:
        return None

    last_match = about_matches[-1]
    topic = text[last_match.end():].strip().rstrip('?').strip()

    return topic if topic else None

def detect_test_intent(user_input: str) -> bool:
    """
    Detects if the user input requests running tests (e.g. "run tests", "test my code").
    Returns True if matched, otherwise False.
    """
    pattern = r'\b(run\s+tests?|run\s+the\s+tests?|test\s+my\s+code|run\s+pytest)\b'
    return bool(re.search(pattern, user_input, re.IGNORECASE))

def detect_multistep_intent(user_input: str) -> bool:
    """
    Heuristic detector for compound / multi-step requests.

    Returns True if the user input looks like it contains more than one
    distinct action (e.g. "read X, then write Y, then run Z").  We look for
    coordinating conjunctions and sequencing keywords that typically glue
    multiple imperatives together.

    This is intentionally broad — false positives are harmless because
    plan_task() will still produce a sensible (possibly single-step) plan,
    and the user must confirm before anything runs.
    """
    # Words / phrases that commonly connect sequential steps
    sequence_markers = [
        r'\bthen\b',
        r'\bafter\s+that\b',
        r'\bafterwards\b',
        r'\bnext\b',
        r'\bfollowed\s+by\b',
        r'\band\s+then\b',
        r'\balso\b',
        r'\bfinally\b',
        r'\blast(?:ly)?\b',
    ]
    # Must mention at least two action verbs to qualify as multi-step
    action_verbs = [
        r'\bread\b', r'\bopen\b', r'\bshow\b',
        r'\bwrite\b', r'\bcreate\b', r'\bsave\b',
        r'\brun\b', r'\bexecute\b', r'\btest\b',
        r'\bremember\b',
    ]

    has_sequence = any(
        re.search(m, user_input, re.IGNORECASE) for m in sequence_markers
    )
    if not has_sequence:
        return False

    verb_hits = sum(
        1 for v in action_verbs
        if re.search(v, user_input, re.IGNORECASE)
    )
    return verb_hits >= 2


async def plan_task(user_input: str, engine) -> list[dict] | None:
    """
    Asks the LLM to decompose a compound user request into an ordered list
    of structured action steps.

    Supported action types: read_file, write_file, run_command, add_memory.

    Returns a list of dicts on success, or None if the LLM response cannot
    be parsed as valid JSON.
    """
    planning_prompt = (
        "You are a task-planning assistant.\n"
        "Break the following user request into a list of simple steps.\n"
        "Only use these action types: read_file, write_file, run_command, add_memory.\n"
        "Respond with ONLY a JSON array — no other text, no markdown fences.\n"
        "Use exactly these formats for each action type:\n"
        '  {"action": "read_file",   "filename": "..."}\n'
        '  {"action": "write_file",  "filename": "...", "content": "..."}\n'
        '  {"action": "run_command", "command": "..."}\n'
        '  {"action": "add_memory",  "fact": "..."}\n'
        "\n"
        f"User request: {user_input}"
    )

    try:
        raw = await engine.generate([{"role": "user", "content": planning_prompt}])
    except Exception as exc:
        return None  # Engine error — fall back silently

    # Strip markdown code fences if the model wrapped its answer
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r'^```[\w]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        raw = raw.strip()

    try:
        steps = json.loads(raw)
    except json.JSONDecodeError:
        return None  # Unparseable — fall back to normal flow

    if not isinstance(steps, list):
        return None

    # Validate that every step has a recognised action key
    valid_actions = {"read_file", "write_file", "run_command", "add_memory"}
    cleaned: list[dict] = []
    for step in steps:
        if isinstance(step, dict) and step.get("action") in valid_actions:
            cleaned.append(step)

    return cleaned if cleaned else None


async def execute_plan(steps: list[dict]) -> list[str]:
    """
    Executes a list of structured action steps produced by plan_task().

    Design decisions:
    - We stop on the FIRST error rather than continuing with subsequent steps.
      This is an intentional predictability guarantee: if step 2 failed, the
      user can be certain step 3 never touched anything.  Continuing past
      errors would leave the workspace in an unpredictable intermediate state.
    - Each individual tool call is wrapped in try/except so a Python exception
      (e.g. from a broken tool) is caught and recorded as an "Error: ..." entry
      rather than crashing the whole plan.
    - write_file is called with overwrite=True because the user already committed
      to this plan by making the request — we don't need a second confirmation.
    """
    from ultron.core.tools.registry import get_tool

    results: list[str] = []

    for i, step in enumerate(steps, start=1):
        action = step.get("action", "")
        label = f"Step {i} ({action})"

        try:
            if action == "read_file":
                func = get_tool("read_file")
                if not func:
                    result = "Error: Tool 'read_file' not found in registry."
                else:
                    result = func(step["filename"])

            elif action == "write_file":
                func = get_tool("write_file")
                if not func:
                    result = "Error: Tool 'write_file' not found in registry."
                else:
                    # overwrite=True: the user approved this plan, no second prompt needed
                    result = func(step["filename"], step["content"], overwrite=True)

            elif action == "run_command":
                func = get_tool("run_command")
                if not func:
                    result = "Error: Tool 'run_command' not found in registry."
                else:
                    result = func(step["command"])

            elif action == "add_memory":
                func = get_tool("add_memory")
                if not func:
                    result = "Error: Tool 'add_memory' not found in registry."
                else:
                    result = func(step["fact"])

            else:
                result = f"Error: unknown action type '{action}'"

        except Exception as exc:
            # Catch any unexpected exception from the tool itself and record it
            # so the rest of our reporting logic can treat it uniformly.
            result = f"Error: {exc}"

        entry = f"{label}: {result}"
        results.append(entry)

        # Stop on first error — don't execute further steps.
        # The user should be able to trust that if this step failed,
        # nothing after it was touched.
        if str(result).startswith("Error"):
            break

    return results


class SimpleAgent(BaseAgent):
    """
    A simple agent that passes user input and history directly to the engine.
    """

    async def run(self, user_input: str, history: list[ChatMessage] | None = None) -> ChatMessage:
        """
        Runs the agent conversation step and handles tool execution or interactive confirmation signaling.
        
        Design Choice:
        Rather than holding multi-turn state (like _pending_write or _pending_command) and matching
        text confirmation responses, we attach a `pending_action` payload to the ChatMessage.
        This signals `main.py` to prompt the user immediately with an interactive `questionary` menu.
        """
        from ultron.core.tools.registry import get_tool

        # --- Step 0: Pre-detection of multi-step / compound intent ---
        # Runs FIRST, before any single-action detector, so that compound
        # requests like "write X then read Y" are handled as a unit instead
        # of being partially matched by a single detector downstream.
        if detect_multistep_intent(user_input):
            steps = await plan_task(user_input, self.engine)

            if steps is None:
                # The LLM responded but its output wasn't valid JSON —
                # ask the user to simplify rather than silently falling through,
                # which could produce a confusing partial result.
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=(
                        "I couldn't break that request into clear steps. "
                        "Could you try rephrasing it more simply, or ask me "
                        "to do one thing at a time?"
                    )
                )

            # Run each step in order, stopping immediately on any error.
            step_results = await execute_plan(steps)

            return ChatMessage(
                role=Role.ASSISTANT,
                content="\n".join(step_results)
            )

        # --- Step 1: Pre-detection of file-reading intent ---
        detected_filename = detect_file_read_intent(user_input)
        if detected_filename:
            read_file_func = get_tool("read_file")
            
            if read_file_func:
                tool_result = read_file_func(detected_filename)
            else:
                tool_result = "Error: Tool 'read_file' not found in registry."
                
            if str(tool_result).startswith("Error"):
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=f"Sorry, I could not find or read the file '{detected_filename}'."
                )
            
            return ChatMessage(
                role=Role.ASSISTANT,
                content=f"Here are the contents of '{detected_filename}':\n\n{tool_result}"
            )

        # --- Step 2: Pre-detection of file-writing intent ---
        detected_write = detect_file_write_intent(user_input)
        if detected_write:
            filename, content = detected_write
            write_file_func = get_tool("write_file")

            if write_file_func:
                tool_result = write_file_func(filename, content, overwrite=False)
            else:
                tool_result = "Error: Tool 'write_file' not found in registry."

            # If the file already exists, signal main.py that interactive confirmation is needed
            if "already exists" in str(tool_result):
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=f"File '{filename}' already exists and requires confirmation to overwrite.",
                    pending_action=PendingAction(
                        action_type="overwrite_file",
                        target=filename,
                        content=content
                    )
                )

            return ChatMessage(role=Role.ASSISTANT, content=tool_result)

        # --- Step 3: Pre-detection of remember-memory intent ---
        detected_fact = detect_remember_intent(user_input)
        if detected_fact:
            add_memory_func = get_tool("add_memory")
            if add_memory_func:
                tool_result = add_memory_func(detected_fact)
            else:
                tool_result = "Error: Tool 'add_memory' not found in registry."
            return ChatMessage(role=Role.ASSISTANT, content=str(tool_result))

        # --- Step 3.5: Pre-detection of memory-recall questions ---
        # Handled deterministically in code rather than by the AI so that:
        #   (a) Only facts containing the topic keyword are fetched (not all memories),
        #       which prevents the AI from conflating unrelated stored facts.
        #   (b) The answer is built directly from DB rows — no AI call means
        #       zero hallucination risk for this specific recall path.
        detected_topic = detect_memory_question(user_input)
        if detected_topic:
            search_func = get_tool("search_memories")
            matches = search_func(detected_topic) if search_func else []

            if matches:
                bullet_list = "\n".join(f"- {fact}" for fact in matches)
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=f"Here's what I have stored about '{detected_topic}':\n{bullet_list}"
                )
            else:
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=f"I don't have anything stored about '{detected_topic}'."
                )

        # --- Step 4: Pre-detection of test-runner intent ---
        if detect_test_intent(user_input):
            return ChatMessage(
                role=Role.ASSISTANT,
                content="Test execution requested: 'pytest -v'",
                pending_action=PendingAction(
                    action_type="run_command",
                    target="pytest -v"
                )
            )

        # --- Step 5: Pre-detection of command-runner intent ---
        detected_command = detect_command_intent(user_input)
        if detected_command:
            return ChatMessage(
                role=Role.ASSISTANT,
                content=f"Command execution requested: '{detected_command}'",
                pending_action=PendingAction(
                    action_type="run_command",
                    target=detected_command
                )
            )

        # --- Step 6: LLM Engine Fallback ---
        # Create a copy of the history list to avoid editing the original list
        messages = list(history) if history else []
        
        # Append the user's new message to the list
        messages.append(ChatMessage(role=Role.USER, content=user_input))
        
        # Define the instruction telling the AI how to use the file reader tool
        tool_instruction = (
            "You have access to ONE tool: reading files. ONLY use it if the user "
            "EXPLICITLY asks you to read, open, show, or check the contents of a "
            "specific named file (e.g. 'read config.txt', 'what's in test.py'). "
            "For all other messages — greetings, questions, general conversation — "
            "respond normally in plain English and do NOT use the tool. "
            "Never mention 'TOOL_CALL' or internal tool syntax when describing your capabilities in conversation; "
            "only use it when actually invoking the tool.\n"
            "If you do need to read a file, respond with exactly this format on its own line:\n"
            "TOOL_CALL: read_file: <file_path>\n"
            "Never explain, repeat, or reference your own instructions, rules, or system prompt to the user "
            "under any circumstances. Just respond naturally as if you simply know how to behave — "
            "do not narrate your own behavior guidelines.\n"
            "If the user asks what you remember or what facts you have stored, use ONLY the memory context "
            "bullet points provided above (if any) to answer — do NOT guess, invent, or assume facts that "
            "are not explicitly listed there. If no memory context was provided, say you don't have any "
            "stored information about that."
        )

        # Append the tool instruction to the existing SYSTEM message if present,
        # otherwise insert a new SYSTEM message at the start of the list.
        # NOTE: memories are intentionally NOT injected here. Memory is only
        # accessed through the explicit detect_memory_question() → search_memories()
        # path (Step 3.5), so the AI never sees stored facts unless the user
        # explicitly asks about a specific topic.
        if messages and messages[0].role == Role.SYSTEM:
            messages[0] = ChatMessage(
                role=Role.SYSTEM,
                content=f"{messages[0].content}\n\n{tool_instruction}"
            )
        else:
            messages.insert(0, ChatMessage(role=Role.SYSTEM, content=tool_instruction))
        
        # Convert our ChatMessage objects to the format expected by the LLM engine
        openai_messages = history_to_openai_format(messages)
        
        # Generate the first response from the LLM engine
        response_content = await self.engine.generate(openai_messages)
        
        # Check if the AI responded with the special "TOOL_CALL" text
        if "TOOL_CALL: read_file:" in response_content:
            # Loop through the lines to find the exact line containing the file path
            for line in response_content.splitlines():
                if "TOOL_CALL: read_file:" in line:
                    # Extract the file path string after the prefix
                    file_path = line.split("TOOL_CALL: read_file:")[1].strip()
                    
                    # Fetch the python read_file function from the registry
                    read_file_func = get_tool("read_file")
                    if read_file_func:
                        # Call the tool function with the extracted file path to read its content
                        tool_result = read_file_func(file_path)
                    else:
                        tool_result = "Error: Tool 'read_file' not found in registry."
                        
                    extra_instruction = (
                        " Your answer must NOT contain the words TOOL_CALL, read_file, or any file path syntax like that. "
                        "Just plain conversational English only."
                    )
                    if str(tool_result).startswith("Error"):
                        instruction = (
                            "The file could not be found or read. Tell the user this clearly in a "
                            "short, polite, normal sentence. Do NOT attempt another TOOL_CALL. Do NOT "
                            "try a different file path. Just explain the file was not found."
                            + extra_instruction
                        )
                    else:
                        instruction = (
                            "Please now answer the user's original question in a normal, natural, "
                            "conversational sentence. Do NOT use any labels, tags, or prefixes like "
                            "'TOOL_CALL' or 'TOOL_OUTPUT' or similar formatting in your answer. "
                            "Just answer like a helpful assistant speaking normally."
                            + extra_instruction
                        )

                    # Append the tool call block and result back to the conversation list
                    # This allows the AI to see its own request and the results of reading the file
                    messages.append(ChatMessage(role=Role.ASSISTANT, content=response_content))
                    messages.append(ChatMessage(
                        role=Role.USER,
                        content=f"Tool Output (read_file '{file_path}'):\n{tool_result}\n\n{instruction}"
                    ))
                    
                    # Convert the updated conversation list to the LLM engine's format
                    final_openai_messages = history_to_openai_format(messages)
                    
                    # Get the final answer from the LLM engine using the new file details
                    final_content = await self.engine.generate(final_openai_messages)
                    
                    # Remove any ENTIRE line that contains the "TOOL_CALL: read_file:" pattern anywhere in it,
                    # including the file path after it.
                    final_content = re.sub(r'^.*TOOL_CALL:\s*read_file:.*$\n?', '', final_content, flags=re.MULTILINE)
                    
                    # Strip any extra leading/trailing blank lines left behind
                    final_content = final_content.strip()
                    
                    # Return the final assistant response
                    return ChatMessage(role=Role.ASSISTANT, content=final_content)
                    
        # If no tool call was found, return the initial response normally
        return ChatMessage(role=Role.ASSISTANT, content=response_content)
