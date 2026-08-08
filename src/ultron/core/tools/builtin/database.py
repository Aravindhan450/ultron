import re
import sqlite3

from ultron.core.config import settings


def is_readonly_query(sql: str) -> bool:
    """
    Determines whether an SQL query is a read-only query (i.e. SELECT).

    Why this matters for safety:
    SELECT queries only read data from the database and leave state untouched, so they are
    safe to execute automatically. On the other hand, non-SELECT queries (INSERT, UPDATE,
    DELETE, DROP, ALTER) modify or destroy data on disk/DB, so they represent state-changing
    actions that require explicit user confirmation.
    """
    # Clean leading whitespace and SQL comments (like -- comment or /* comment */)
    cleaned = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    cleaned = re.sub(r'--.*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    return cleaned.lower().startswith("select") or cleaned.lower().startswith("with")


def run_query(sql: str) -> str:
    """
    Executes an SQL query against SQLite or PostgreSQL database according to settings.

    Parameters:
      - sql: The SQL query text to execute

    Returns:
      For SELECT queries: A formatted table-like string with headers and up to 50 rows.
      For non-SELECT queries: A confirmation string showing affected row count.

    Safety Distinction:
      SELECT queries read data without side-effects. Non-SELECT queries modify data,
      commit transactions to disk/DB, and report the count of affected rows.
    """
    if not settings.database_url:
        return "Error: no database configured. Set DATABASE_TYPE and DATABASE_URL in your .env file."

    db_type = settings.database_type.lower()
    db_url = settings.database_url.strip()

    is_select = is_readonly_query(sql)

    try:
        if db_type == "sqlite":
            # Connect to SQLite database file
            conn = sqlite3.connect(db_url)
            cursor = conn.cursor()
            cursor.execute(sql)

            if is_select:
                # Extract column headers from cursor description
                columns = [desc[0] for desc in (cursor.description or [])]
                rows = cursor.fetchmany(50)

                # Check if there are more rows beyond 50
                has_more = cursor.fetchone() is not None
                conn.close()

                if not columns and not rows:
                    return "Query returned no results."

                # Format output table header and rows
                header_line = " | ".join(columns) if columns else "Result"
                separator = "-" * max(len(header_line), 20)
                data_lines = [" | ".join(str(val) for val in row) for row in rows]

                output = [header_line, separator] + data_lines
                if has_more:
                    output.append("\n[Note: showing first 50 rows of results]")

                return "\n".join(output)

            else:
                affected = cursor.rowcount
                conn.commit()
                conn.close()
                return f"Query executed successfully. Rows affected: {affected if affected >= 0 else 'N/A'}"

        elif db_type == "postgres":
            try:
                import psycopg2
            except ImportError:
                return "Error: psycopg2 module not found. Install psycopg2-binary to connect to PostgreSQL."

            try:
                conn = psycopg2.connect(db_url)
                cursor = conn.cursor()
                cursor.execute(sql)

                if is_select:
                    columns = [desc[0] for desc in (cursor.description or [])]
                    rows = cursor.fetchmany(50)
                    has_more = cursor.fetchone() is not None

                    conn.close()

                    if not columns and not rows:
                        return "Query returned no results."

                    header_line = " | ".join(columns) if columns else "Result"
                    separator = "-" * max(len(header_line), 20)
                    data_lines = [" | ".join(str(val) for val in row) for row in rows]

                    output = [header_line, separator] + data_lines
                    if has_more:
                        output.append("\n[Note: showing first 50 rows of results]")

                    return "\n".join(output)

                else:
                    affected = cursor.rowcount
                    conn.commit()
                    conn.close()
                    return f"Query executed successfully. Rows affected: {affected if affected >= 0 else 'N/A'}"
            except (psycopg2.Error, OSError, ValueError) as exc:
                # psycopg2 is an optional extra, so its error type can only be
                # referenced inside the postgres branch.
                return f"Error: {exc}"

        else:
            return f"Error: unsupported DATABASE_TYPE '{db_type}'. Must be 'sqlite' or 'postgres'."

    except (sqlite3.Error, OSError, ValueError) as exc:
        return f"Error: {exc}"
