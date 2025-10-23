import os
import shutil
import sqlite3
import pandas as pd
import requests
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import re
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Database setup (unchanged)
db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"
backup_file = "travel2.backup.sqlite"
overwrite = False

if overwrite or not os.path.exists(local_file):
    response = requests.get(db_url)
    response.raise_for_status()
    with open(local_file, "wb") as f:
        f.write(response.content)
    shutil.copy(local_file, backup_file)

def update_dates(file):
    shutil.copy(backup_file, file)
    conn = sqlite3.connect(file)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * FROM {t}", conn)
    
    example_time = pd.to_datetime(tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time
    
    tdf["bookings"]["book_date"] = pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True) + time_diff
    
    datetime_columns = ["scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival"]
    for col in datetime_columns:
        tdf["flights"][col] = pd.to_datetime(tdf["flights"][col].replace("\\N", pd.NaT)) + time_diff
    
    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.commit()
    return file

db_path = update_dates(local_file)

# Database updates
conn = sqlite3.connect(local_file)
cursor = conn.cursor()

cursor.execute("UPDATE hotels SET booked = 1 WHERE id in(1,2,3,4,5,6)")
cursor.execute("UPDATE trip_recommendations SET booked = 1 WHERE id in(1,2,3,4,5,6)")
cursor.execute("UPDATE car_rentals SET booked = 1 WHERE id in(1,2,3,4,5,6)")

conn.commit()

# Load Model
def load_model():
    MODEL_PATH = 'gaussalgo/T5-LM-Large-text2sql-spider'
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return model, tokenizer

model, tokenizer = load_model()

def extract_user_schema(user_prompt, conn_path=local_file):
    """Extract mentioned tables and columns, and get real schema from DB if needed"""
    prompt_lower = user_prompt.lower()

    # Available tables in DB
    conn = sqlite3.connect(conn_path)
    available_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)["name"].tolist()

    mentioned_tables = [t for t in available_tables if t.lower() in prompt_lower]
    user_columns = []

    # Extract potential columns mentioned explicitly
    column_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
    tokens = re.findall(column_pattern, user_prompt)
    sql_keywords = {'select', 'from', 'where', 'group', 'order', 'join', 'on', 'and', 'or', 'by', 'as', 'limit'}
    tokens = [t for t in tokens if t.lower() not in sql_keywords]

    # Build schema dictionary (actual DB columns)
    schema = {}
    for table in mentioned_tables:
        try:
            cols_df = pd.read_sql(f"PRAGMA table_info({table});", conn)
            schema[table] = cols_df["name"].tolist()
        except Exception:
            schema[table] = []

    conn.close()
    return mentioned_tables, tokens, schema


def generate_user_constrained_sql(question: str, schema_text: str):
    """
    Generate SQL using the user-provided schema and question.
    The model is strictly limited to the schema content.
    """
    if not question.strip():
        return "Error: Question input is empty. Please enter your business logic."
    if not schema_text.strip():
        return "Error: Schema input is empty. Please enter table and column definitions."

    # Construct model input
    input_text = (
        f"### INSTRUCTION:\n"
        f"Generate a valid SQL query using ONLY the provided schema. "
        f"Do not invent or use any tables, columns, or joins not defined in the schema.\n\n"
        f"### SCHEMA:\n{schema_text}\n\n"
        f"### QUESTION:\n{question.strip()}"
    )

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
        )

    sql = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    sql = sql.replace("\n", " ")

    # Clean if model outputs extra prefixes
    if sql.lower().startswith("sql:"):
        sql = sql.split(":", 1)[-1].strip()

    return sql

def run_query(sql_query: str):
    """Execute the generated SQL on the SQLite database"""
    conn = sqlite3.connect(local_file)
    try:
        df = pd.read_sql_query(sql_query, conn)
        return df
    except Exception as e:
        return f"Error executing SQL: {str(e)}"
    finally:
        conn.close()

def get_available_tables():
    """Get list of available tables"""
    conn = sqlite3.connect(local_file)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).name.tolist()
    conn.close()
    return tables
