from langchain.chains import LLMCheckerChain
from langchain.llms import OpenAI
import json
from termcolor import colored

import langchain
from dotenv import load_dotenv, find_dotenv
from langchain import ConversationChain
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.cache import InMemoryCache
from langchain.chains.openai_functions import (
    create_structured_output_chain,
)
from langchain import (
    LLMMathChain,
    OpenAI,
    SerpAPIWrapper,
    SQLDatabase,
)
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from langchain.memory import ConversationSummaryMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel, Field

from adventure.utils import get_model, get_default_kwargs

load_dotenv(find_dotenv(raise_error_if_not_found=True))

llm = OpenAI(temperature=0.3)

prompt = PromptTemplate(
    input_variables=["answer", "correct_answer"],
    template=""""
    Your task is to compare two statements. 
    The first statement is "{answer}.
    The second statement is "{correct_answer}".
    Compare the main ideas of the statements.
    Wryte the result of comparison. and Add Yes or No to the end of the result.
    """
)
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run({
    'answer': "Python",
    'correct_answer': "Is it a Python?"
    }))

print(chain.run({
    'answer': "Python",
    'correct_answer': "Sounds like Python"
    }))