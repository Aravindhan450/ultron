import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from rich.logging import RichHandler
from ultron.core.config import settings

# Path to the persistent log file
LOG_FILE = settings.data_dir / "ultron.log"

def setup_logging() -> None:
    """
    Configure the global logging system with Rich console handler and persistent file logging.
    """
    # Create the data directory if it doesn't exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    level_val = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Console Handler using Rich
    console_handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
    console_handler.setLevel(level_val)

    # File Handler with rotation
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(level_val)

    # Root logging configuration
    logging.basicConfig(
        level=level_val,
        handlers=[console_handler, file_handler],
    )

    # Silence verbose HTTP status logs from httpx and httpcore
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the specified name.
    """
    return logging.getLogger(name)

# Initialize logging configuration immediately on module load
setup_logging()
