import sqlite3
import os

# Ensure database folder exists
os.makedirs("database", exist_ok=True)

# Path to the SQLite database
db_path = "database/users.db"

# Connect (this will create the database if it doesn't exist)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create the users table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
''')

conn.commit()
conn.close()

print(f"✅ Database initialized successfully at {db_path}")
