# app.py
import streamlit as st
# from nl2sql import generate_sql, run_query
from langchaintravel2sqlite import generate_sql, run_query, execute_sql

st.title("Natural Language Text to SQL")

schema_desc = st.text_area("Schema Description", placeholder="""Example: 
Table flights(flight_id, flight_no, scheduled_departure, scheduled_arrival, departure_airport, arrival_airport, status, aircraft_code, actual_departure, actual_arrival)
""", height=150)

question = st.text_area("Ask a question:", placeholder="Show the top 10 most frequent flight routes using columns departure airport, arrival airport, count of total number of scheduled flights where status Scheduled, grouped by departure airport, arrival airport, ordered by the total number of scheduled flights in descending order", height=150)

if st.button("Generate SQL"):
    sql_query = generate_sql(schema_desc, question)
    st.code(sql_query, language="sql")

    df = execute_sql(sql_query)
    st.write("Results:")
    st.dataframe(df.head(10))
