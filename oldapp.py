# app.py
import streamlit as st
from nl2sql import generate_sql, run_query

st.title("Natural Language to Snowflake SQL")

schema_desc = st.text_area("Snowflake Schema Description", value=""" 
Table DD_SALES_METRICS (
  COMPANYID NUMBER,
  RUNNAME VARCHAR,
  CAMPAIGNACTUALSENTDATE DATE,
  DEALS_30 NUMBER
)
Table DD_SERVICE_METRICS (
  COMPANYID NUMBER,
  RUNNAME VARCHAR,
  CAMPAIGNACTUALSENTDATE DATE,
  NUMREPAIRORDERS NUMBER
)
""")

question = st.text_input("Ask a question:", "Show me sum of DEALS_30 by RUNNAME and sum of NUMREPAIRORDERS for last month join using COMPANYID columns from both tables")

if st.button("Generate SQL"):
    sql_query = generate_sql(schema_desc, question)
    st.code(sql_query, language="sql")

    columns, df = run_query(sql_query)
    st.write("Results:")
    st.dataframe(df.head(10))
