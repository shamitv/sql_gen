from pathlib import Path

from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_core.runnables import Runnable

template= """
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{user_question}`
{instructions}

DDL statements:
{create_table_statements}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The following SQL query best answers the question `{user_question}`:
```sql
"""

file_name = '/home/shamit/proj/models/llama-3-sqlcoder-8b-Q4_K_M.gguf'
llm = LlamaCpp(
    model_path=file_name,
    temperature=0,
    n_ctx=2048,
    top_p=1,
    verbose=True,  
)


user_question= """
Question: Which customer generated maximum revenue ? 
"""

db_path = Path(__file__).parent.parent / "data" / "Chinook_Sqlite.sqlite"
print(db_path)
db_string = f"sqlite:///{db_path}"
db = SQLDatabase.from_uri(db_string, sample_rows_in_table_info=0)

def get_schema():
    table_info = db.get_table_info()
    #print(table_info)
    return table_info


instructions=""

create_table_statements=get_schema()

prompt=PromptTemplate(input_variables=['user_question', 'create_table_statements','instructions'],
                      template=template, validate_template=False )


print()

prompt_value = prompt.format(user_question=user_question,create_table_statements=create_table_statements,
                    instructions=instructions)
#print(llm.invoke(prompt_value))
chain: Runnable = prompt | llm
msg = chain.invoke({"user_question":user_question,
              "create_table_statements":create_table_statements,"instructions":instructions})
print(msg)