# langchaintravel2sqlite.py
import os
import pandas as pd
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
from dotenv import load_dotenv
import shutil
import sqlite3
import requests

db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"
# The backup lets us restart for each tutorial section
backup_file = "travel2.backup.sqlite"
overwrite = False
if overwrite or not os.path.exists(local_file):
    response = requests.get(db_url)
    response.raise_for_status()  # Ensure the request was successful
    with open(local_file, "wb") as f:
        f.write(response.content)
    # Backup - we will use this to "reset" our DB in each section
    shutil.copy(local_file, backup_file)


# Load environment variables from a .env file
load_dotenv()
# Load OpenAI key securely:
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

sql_prompt = PromptTemplate(
    input_variables=["schema_description", "question"],
    template="""
You are an expert Snowflake SQL generator.
Given the following Snowflake schema:

{schema_description}

Convert the business question below into a valid Snowflake SQL query.

Business Question: {question}

Only output the SQL query, nothing else.
"""
)

chain = sql_prompt | llm

def generate_sql(schema_description: str, question: str) -> str:
    """Convert NL to SQL using OpenAI."""
    result = chain.invoke({
        "schema_description": schema_description,
        "question": question
    })
    sql = result.content.strip()
    # remove any markdown fences ```sql ... ```
    if sql.startswith("```"):
        sql = sql.split("```")[1]  # take inside of first fence
    sql = sql.replace("sql", "", 1).strip("`\n ")
    return sql

def run_query(file):
    """Run SQL against Snowflake and return (columns, results)."""

    shutil.copy(backup_file, file)
    conn = sqlite3.connect(file)
    cursor = conn.cursor()

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
        + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    del df
    del tdf
    conn.commit()

    return file

db = run_query(local_file)

conn = sqlite3.connect(local_file)
cursor = conn.cursor()

tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()

# run_query now executes SQL string
def execute_sql(sql_query: str, db_file="travel2.sqlite"):
    """Run a SQL query against the SQLite DB and return DataFrame."""
    conn = sqlite3.connect(db_file)
    df = pd.read_sql(sql_query, conn)
    # conn.close()
    return df

