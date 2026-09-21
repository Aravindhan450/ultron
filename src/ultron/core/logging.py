import logging
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler

from ultron.core.config import settings

# Path to the persistent log file
LOG_FILE = settings.data_dir / "ultron.log"

_console_handler: RichHandler | None = None
_file_handler: RotatingFileHandler | None = None


def is_verbose_logging() -> bool:
    """Returns True if console logging of diagnostics is enabled."""
    if _console_handler is not None:
        return _console_handler.level <= logging.INFO
    import os

    env_verbose = os.environ.get("ULTRON_VERBOSE", "").lower() in ("1", "true", "yes")
    return bool(settings.verbose or env_verbose)


def setup_logging(verbose: bool | None = None) -> None:
    """
    Configure the global logging system with persistent file logging and
    isolated console logging (enabled only in verbose/debug mode).
    """
    global _console_handler, _file_handler

    # Create the data directory if it doesn't exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    level_val = getattr(logging, settings.log_level.upper(), logging.INFO)
    effective_verbose = is_verbose_logging() if verbose is None else bool(verbose)

    # Console Handler using Rich (isolated unless verbose is explicitly enabled)
    if _console_handler is None:
        _console_handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
    _console_handler.setLevel(level_val if effective_verbose else (logging.CRITICAL + 1))

    # File Handler with rotation (always logs all diagnostics)
    if _file_handler is None:
        _file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _file_handler.setFormatter(file_formatter)
    _file_handler.setLevel(level_val)

    # Root logging configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level_val)

    if _console_handler not in root_logger.handlers:
        root_logger.addHandler(_console_handler)
    if _file_handler not in root_logger.handlers:
        root_logger.addHandler(_file_handler)

    # Silence verbose HTTP status logs from httpx, httpcore, and primp
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("primp").setLevel(logging.WARNING)


def set_verbose_logging(enabled: bool) -> None:
    """Dynamically toggle verbose diagnostic console logging."""
    level_val = getattr(logging, settings.log_level.upper(), logging.INFO)
    if _console_handler is not None:
        _console_handler.setLevel(level_val if enabled else (logging.CRITICAL + 1))




def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the specified name.
    """
    return logging.getLogger(name)


# Initialize logging configuration immediately on module load (defaults to clean chat)
setup_logging()
