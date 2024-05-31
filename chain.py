# Get LLM
import os
import re
from pathlib import Path
from typing import Optional, List, Any, Iterator, Dict

import requests
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.utilities import SQLDatabase
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import GenerationChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# File name and URL
""""
file_name = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
url = (
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/"
    "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
)
"""

file_name = '/home/shamit/proj/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf'
#file_name = '/home/shamit/proj/models/Meta-Llama-3-8B-Instruct-v2.Q8_0.gguf'
#file_name = '/home/shamit/proj/models/llama-3-sqlcoder-8b-Q4_K_M.gguf'

url = ''

model_path = file_name

class MyLlamaCpp(LlamaCpp):
    def __init__(self, /, **data: Any):
        super().__init__(**data)



    def validate_environment(cls, values: Dict) -> Dict:
        return super().validate_environment(values)

    def build_model_kwargs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return super().build_model_kwargs(values)

    @property
    def _default_params(self) -> Dict[str, Any]:
        return super()._default_params

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return super()._identifying_params

    @property
    def _llm_type(self) -> str:
        return super()._llm_type

    def _get_parameters(self, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        return super()._get_parameters(stop)

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        return super()._call(prompt, stop, run_manager, **kwargs)

    def _stream(self, prompt: str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Iterator[GenerationChunk]:
        return super()._stream(prompt, stop, run_manager, **kwargs)

    def get_num_tokens(self, text: str) -> int:
        return super().get_num_tokens(text)


llm = MyLlamaCpp(
    model_path=model_path,
    seed=450,
    temperature=0.95,
    n_ctx=2048,
    # f16_kv MUST set to True
    # otherwise you will run into problem after a couple of calls
    verbose=True,
)

db_path = Path(__file__).parent.parent / "data" / "Chinook_Sqlite.sqlite"
#db_path = Path(__file__).parent.parent / "data" / "nba_roster.db"
print(db_path)
#rel = db_path.relative_to(Path.cwd())
db_string = f"sqlite:///{db_path}"
db = SQLDatabase.from_uri(db_string, sample_rows_in_table_info=0)


def get_schema(_):
    table_info = db.get_table_info()
    #print(table_info)
    return table_info


def run_query(query):
    return db.run(query)


# Prompt

template = """Based on the table schema below, 
write a SQL query that would answer the user's question. Write SQL query in SQLite dialect:
{schema}

Question: {question}
SQL Query:"""  # noqa: E501
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Given an input question, convert it to a SQL query. No pre-amble."),
        MessagesPlaceholder(variable_name="history"),
        ("human", template),
    ]
)

memory = ConversationBufferMemory(return_messages=True)

# Chain to query with memory

sql_chain = (
        RunnablePassthrough.assign(
            schema=get_schema,
            history=RunnableLambda(lambda x: memory.load_memory_variables(x)["history"]),
        )
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
)


def save(input_output):
    output = {"output": input_output.pop("output")}
    memory.save_context(input_output, output)
    return output["output"]


sql_response_memory = RunnablePassthrough.assign(output=sql_chain) | save

# Chain to answer
template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""  # noqa: E501
prompt_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and SQL response, convert it to a natural "
            "language answer. No pre-amble.",
        ),
        ("human", template),
    ]
)


# Supply the input types to the prompt
class InputType(BaseModel):
    question: str


def exec_qry(db, qry):
    pre_qry = qry
    #new_qry = re.sub(r".*```", " " ,pre_qry)
    print(pre_qry)
    pre_qry = pre_qry.strip()
    new_qry = re.sub(r"\n\n.*", " ", pre_qry, flags=re.DOTALL).strip()
    new_qry_1 = re.sub(r"```sql\s", " ", new_qry, flags=re.DOTALL).strip()
    new_qry_2 = re.sub(r"```", " ", new_qry_1, flags=re.DOTALL).strip()
    new_qry_3 = re.sub(r";.*", " ", new_qry_2, flags=re.DOTALL).strip()
    print(new_qry_3)
    retval = db.run(new_qry_2)
    return retval


chain = (
        RunnablePassthrough.assign(query=sql_response_memory).with_types(
            input_type=InputType
        )
        | RunnablePassthrough.assign(
    schema=get_schema,
    response=lambda x: exec_qry(db, x["query"]),
)
        | prompt_response
        | llm
)

if __name__ == "__main__":
    chain.invoke({
        "question": "Which customer generated maximum revenue ? "
    })
