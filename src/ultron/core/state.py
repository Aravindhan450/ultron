from dataclasses import dataclass


@dataclass
class CLIState:
    active_model: str
    current_dir: str
    version: str = "v1.0.0"
    status: str = "Ready"  # E.g., "Ready", "Thinking...", "Executing Tool: file_writer"
