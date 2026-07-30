import logging
from rich.logging import RichHandler
from ultron.core.config import settings

def setup_logging() -> None:
    """
    Configure the global logging system with Rich handler support.
    """
    # Map string log level to logging constants
    level_val = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level_val,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
    )

def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the specified name.
    """
    return logging.getLogger(name)

# Initialize logging configuration immediately on module load
setup_logging()
