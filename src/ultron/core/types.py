from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

class Role(str, Enum):
    """
    Represents the sender's role in a conversation.
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class ChatMessage(BaseModel):
    """
    Represents a structured chat message for the Ultron assistant.
    """
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_openai_format(self) -> dict[str, Any]:
        """
        Convert the ChatMessage to standard OpenAI compatible dict structure.
        """
        payload: dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload

def truncate_history(history: list[ChatMessage], max_messages: int = 10) -> list[ChatMessage]:
    """
    Truncates conversation history to stay within message limits.
    
    Preserves all leading SYSTEM messages at the start of history, 
    and keeps up to the last `max_messages` non-system messages.
    """
    leading_system: list[ChatMessage] = []
    index = 0
    # Collect all consecutive system messages at the start of the list
    while index < len(history) and history[index].role == Role.SYSTEM:
        leading_system.append(history[index])
        index += 1

    # Extract all non-system messages that follow
    non_system = [msg for msg in history[index:] if msg.role != Role.SYSTEM]
    truncated_non_system = non_system[-max_messages:]

    return leading_system + truncated_non_system

def history_to_openai_format(history: list[ChatMessage]) -> list[dict[str, Any]]:
    """
    Map history list of ChatMessage instances to OpenAI compatible dictionary list.
    """
    return [msg.to_openai_format() for msg in history]
