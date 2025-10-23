from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Load Hugging Face model
model_name = "gaussalgo/T5-LM-Large-text2sql-spider"
pipe = pipeline(
    "text2text-generation",
    model=model_name,
    tokenizer=model_name,
    max_length=512,
    do_sample=False  # deterministic generation
)

llm = HuggingFacePipeline(pipeline=pipe)

# Enhanced prompt with stronger instructions
template = """
You are an expert SQL data analyst.

Convert the user's question into a valid SQL query using the given schema.
Rules:
- Include WHERE, GROUP BY, ORDER BY, and LIMIT when they are implied.
- Use COUNT(), SUM(), or AVG() for totals or averages.
- Use only the columns and tables defined in the schema.
- Do not add self-joins or duplicate table aliases unless explicitly asked.
- Output only the SQL query.

Schema:
{schema}

Question:
{question}

SQL:
"""

prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=template
)

# New RunnableSequence style instead of deprecated LLMChain
chain = prompt | llm

schema = """
Table flights(
  flight_id,
  airline_id,
  departure_airport,
  arrival_airport,
  status,
  scheduled_departure,
  actual_departure,
  scheduled_arrival,
  actual_arrival
)
"""

question = "List all details of flights using columns departure airport, arrival airport, count of total number of scheduled flights where status Scheduled, grouped by departure airport, arrival airport, ordered by the total number of scheduled flights in descending order"

# Use invoke() instead of run()
result = chain.invoke({"schema": schema, "question": question})
print("\nGenerated SQL:\n", result)
