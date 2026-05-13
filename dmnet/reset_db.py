"""
reset_db.py — Run this ONCE to reset the database schema.
Usage: python reset_db.py
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "predictions.db")

print(f"Resetting database at: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)

# Drop old table completely
conn.execute("DROP TABLE IF EXISTS predictions")

# Recreate with full schema
conn.execute("""
    CREATE TABLE predictions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id      TEXT    NOT NULL,
        patient_name    TEXT    DEFAULT '',
        timestamp       TEXT    NOT NULL,
        age             REAL, bmi REAL, glucose REAL, insulin REAL,
        blood_pressure  REAL, pregnancies REAL, hba1c REAL,
        fasting_glucose REAL, physical_activity REAL,
        smoking_history REAL, family_history REAL,
        probability     REAL    NOT NULL,
        prediction      INTEGER NOT NULL,
        label           TEXT    NOT NULL,
        risk_category   TEXT    NOT NULL,
        notes           TEXT    DEFAULT ''
    )
""")
conn.commit()
conn.close()

print("Done! Database reset successfully.")
print("Now restart Streamlit and run a prediction.")
