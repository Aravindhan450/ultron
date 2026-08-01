import re
from ultron.core.agents.base import BaseAgent
from ultron.core.types import ChatMessage, Role, history_to_openai_format

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

class SimpleAgent(BaseAgent):
    """
    A simple agent that passes user input and history directly to the engine.
    """

    def __init__(self, engine):
        super().__init__(engine)
        # Multi-step state tracking: holds (filename, content) when a write requires confirmation
        self._pending_write: tuple[str, str] | None = None

    async def run(self, user_input: str, history: list[ChatMessage] | None = None) -> ChatMessage:
        """
        Runs the agent conversation step and handles tool execution if requested by the AI.
        """
        from ultron.core.tools.registry import get_tool

        # --- Multi-Step Confirmation Handling ---
        # Check if the user is confirming a pending overwrite attempt ("yes overwrite <filename>")
        overwrite_match = re.search(r'^\s*yes\s+overwrite\s+([\w./-]+\.[a-zA-Z0-9]+)\s*$', user_input, re.IGNORECASE)
        if overwrite_match:
            target_filename = overwrite_match.group(1).strip()
            if self._pending_write and self._pending_write[0].lower() == target_filename.lower():
                pending_filename, pending_content = self._pending_write
                self._pending_write = None  # Clear pending state after handling confirmation
                
                write_file_func = get_tool("write_file")
                if write_file_func:
                    result = write_file_func(pending_filename, pending_content, overwrite=True)
                else:
                    result = "Error: Tool 'write_file' not found in registry."

                return ChatMessage(role=Role.ASSISTANT, content=result)
            else:
                self._pending_write = None
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=f"No pending overwrite found for '{target_filename}'."
                )

        # Clear any pending write state if the user sends anything other than an overwrite confirmation
        self._pending_write = None

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

            # Multi-step confirmation flow:
            # If the file already exists, we do NOT overwrite it silently.
            # Instead, store the (filename, content) pair in self._pending_write and ask the user to confirm.
            if "already exists" in str(tool_result):
                self._pending_write = (filename, content)
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=(
                        f"The file '{filename}' already exists. "
                        f"Reply with 'yes overwrite {filename}' if you want me to replace it, "
                        f"or choose a different filename."
                    )
                )

            # Return success or any other error (e.g., access denied outside project folder) directly
            return ChatMessage(role=Role.ASSISTANT, content=tool_result)

        # --- Step 3: LLM Engine Fallback ---
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
            "TOOL_CALL: read_file: <file_path>"
        )
        
        # Append the tool instruction to the existing SYSTEM message if present,
        # otherwise insert a new SYSTEM message at the start of the list.
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
