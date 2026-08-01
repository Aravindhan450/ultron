import sqlite3
from ultron.core.tools.paths import ALLOWED_BASE_DIR

# Path to the SQLite database file inside the project root folder
MEMORY_DB_PATH = ALLOWED_BASE_DIR / ".ultron_memory.db"

def init_memory_db() -> None:
    """
    Initializes the SQLite database and creates the 'memories' table if it does not already exist.
    """
    try:
        with sqlite3.connect(MEMORY_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except Exception as e:
        # Print warning to console if DB initialization fails
        print(f"Warning: Failed to initialize memory DB: {e}")

def add_memory(fact: str) -> str:
    """
    Inserts a new memory fact into the database.
    """
    try:
        with sqlite3.connect(MEMORY_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO memories (fact) VALUES (?)", (fact,))
            conn.commit()
        return f"Remembered: {fact}"
    except Exception as e:
        return f"Error storing memory: {str(e)}"

def get_all_memories() -> list[str]:
    """
    Retrieves all stored facts from the database ordered by creation date (oldest first).
    """
    try:
        with sqlite3.connect(MEMORY_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT fact FROM memories ORDER BY created_at ASC")
            rows = cursor.fetchall()
            return [row[0] for row in rows]
    except Exception:
        return []

def clear_all_memories() -> str:
    """
    Deletes all stored facts from the database.
    """
    try:
        with sqlite3.connect(MEMORY_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories")
            conn.commit()
        return "All memories cleared successfully."
    except Exception as e:
        return f"Error clearing memories: {str(e)}"

def search_memories(keyword: str) -> list[str]:
    """
    Returns facts from the database that contain the given keyword (case-insensitive).

    Uses SQL LIKE with wildcards so partial matches work — e.g. searching "FastAPI"
    will match "I use FastAPI for my projects."

    Returning only matching facts (rather than all facts) is the key accuracy
    improvement: the AI never sees unrelated memories, which was causing it to
    mention or conflate irrelevant stored facts in its answers.
    """
    try:
        with sqlite3.connect(MEMORY_DB_PATH) as conn:
            cursor = conn.cursor()
            # The % wildcards on both sides make this a substring match,
            # and LIKE is case-insensitive by default in SQLite for ASCII text.
            cursor.execute(
                "SELECT fact FROM memories WHERE fact LIKE ? ORDER BY created_at ASC",
                (f"%{keyword}%",)
            )
            rows = cursor.fetchall()
            return [row[0] for row in rows]
    except Exception:
        # Return empty list on any DB error — the caller treats this the same
        # as "no results found".
        return []

# Automatically initialize database when module is first imported
init_memory_db()
