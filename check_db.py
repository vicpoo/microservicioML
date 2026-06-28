from app.models.database import init_db, engine
import sqlite3

init_db()
print(engine.url)
conn = sqlite3.connect('app.db')
print(conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
conn.close()
