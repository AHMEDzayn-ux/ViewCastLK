"""Only application connections use these bounds; migrations are unaffected."""
import os

def connection_options():
    statement = int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "5000"))
    lock = int(os.getenv("DB_LOCK_TIMEOUT_MS", "1000"))
    if not 500 <= statement <= 30000 or not 100 <= lock <= 5000:
        raise ValueError("Invalid database timeout configuration")
    return f"-c statement_timeout={statement} -c lock_timeout={lock}"
