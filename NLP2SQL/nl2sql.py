# nl2sql.py
import os
import pandas as pd
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
from dotenv import load_dotenv

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

def run_query(sql_query: str):
    """Run SQL against Snowflake and return (columns, results)."""

    engine = create_engine(URL(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE")
    ), pool_size=10, max_overflow=20, connect_args={'insecure_mode': True})

    with engine.connect() as conn:
        df = pd.read_sql(sql_query, conn)
    return list(df.columns), df.head(10)