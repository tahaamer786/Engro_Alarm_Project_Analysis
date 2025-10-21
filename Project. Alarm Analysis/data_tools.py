# data_tools.py (FINAL VERSION with Alarm Logic Integration + SQLite + Data Cleaning)

import pandas as pd
import json
import sqlite3
import io
from typing import Dict, List, Any, Optional
import inspect
from alarm_logic import analyze as alarm_analyze   #  import alarm logic analyzer

# --- Global Database Configuration ---
DB_FILE = 'alerts.db'
TABLE_NAME = 'alerts'

# Sahi column names (used for both CSV loading and database table creation)
COLUMN_NAMES = [
    "Event Time", "Location Tag", "Source", "Condition", "Action",
    "Priority", "Description", "Value", "Units", "Extra"
]
# Columns we will actually use, excluding 'Extra'
USED_COLUMNS = COLUMN_NAMES[:-1]


def load_data(file_path: str, sheet_name: str = 'alert_data'):
    """
    Loads data from the CSV file, cleans it, and writes it to an SQLite database.
    This replaces the global SHEETS dictionary with a permanent database file.
    """
    try:
        # 1. Robust CSV Reading & Cleaning
        with open(file_path, 'rb') as f:
            binary_data = f.read().replace(b'\x00', b'')
        data_string = binary_data.decode('latin1', errors='ignore').replace('\r', '')
        data_io = io.StringIO(data_string)

        df = pd.read_csv(
            data_io,
            sep=',',
            engine='python',
            header=None,
            skiprows=1,
            names=COLUMN_NAMES,
            on_bad_lines='skip'
        )

        if 'Extra' in df.columns:
            df = df.drop(columns=['Extra'])

        df.columns = df.columns.str.strip()
        df = df.dropna(how='all')  # Drop rows where all values are NaN

        # 2. Type Conversion (Crucial for SQLite)
        if 'Event Time' in df.columns:
            df['Event Time'] = pd.to_datetime(df['Event Time'], errors='coerce')
            df = df.dropna(subset=['Event Time'])  # Drop rows with bad 'Event Time'

        # Ensure 'Value' is numeric, coercing errors to NaN before dropping rows
        if 'Value' in df.columns:
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

        # ===  CRITICAL FIX: DATA NORMALIZATION ===
        text_cols_to_clean = ["Location Tag", "Source", "Condition", "Action", "Priority", "Description", "Units"]
        for col in text_cols_to_clean:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        # ===========================================

        if len(df) == 0:
            raise Exception("Data load failed: Zero valid rows found after cleanup.")

        # 3. Write to SQLite Database
        conn = sqlite3.connect(DB_FILE)
        df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        conn.close()

        print(f" Data Loaded: {len(df)} rows written to SQLite database '{DB_FILE}' in table '{TABLE_NAME}'.")
        return True

    except FileNotFoundError:
        print(f" Error: File '{file_path}' not found.")
        return False
    except Exception as e:
        print(f" Error loading data to SQLite: {e}")
        return False


# ==================== SQL EXECUTION TOOL ====================

def execute_sql_query(sql_query: str) -> str:
    """
    Executes a read-only SQL query (SELECT) against the 'alerts' database table.
    """
    try:
        if not sql_query.strip().upper().startswith('SELECT'):
            return json.dumps({"error": "Only SQL SELECT statements are permitted."})

        conn = sqlite3.connect(DB_FILE)
        result_df = pd.read_sql_query(sql_query, conn)
        conn.close()

        if result_df.empty:
            return json.dumps({"message": "Query returned zero results."})

        return result_df.head(10).to_json(orient='records', indent=2)

    except sqlite3.OperationalError as e:
        return json.dumps({"error": f"SQL Query Error (Operational): {str(e)}", "query": sql_query})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error during SQL execution: {str(e)}"})


# ==================== ALARM BEHAVIOR ANALYSIS TOOL ====================

def analyze_alarm_behavior(sql_query: str) -> str:
    """
    Tool: Execute the given SELECT SQL, run alarm logic, return JSON string.
    Expectation: SQL should return rows with at least 'Event Time' and 'Source' columns.
    """
    try:
        if not sql_query.strip().upper().startswith("SELECT"):
            return json.dumps({"error": "Only SELECT queries allowed."})

        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()

        if df.empty:
            return json.dumps({"message": "Query returned zero rows."})

        # Normalize column names like existing cleaning
        cols = {c.upper(): c for c in df.columns}
        if 'SOURCE' in cols:
            df = df.rename(columns={cols['SOURCE']: 'Source'})
        if 'EVENT TIME' in cols:
            df = df.rename(columns={cols['EVENT TIME']: 'Event Time'})

        result = alarm_analyze(df, time_col="Event Time")
        return json.dumps(result, default=str, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ==================== TOOL REGISTRY ====================

AVAILABLE_TOOLS = [
    execute_sql_query,
    analyze_alarm_behavior
]
