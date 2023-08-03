import json

from langchain.chains.openai_functions import (
    create_structured_output_chain,
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage
)

from dotenv import load_dotenv, find_dotenv
from langchain import ConversationChain
from langchain import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain import OpenAI, LLMMathChain
from langchain.output_parsers import OutputFixingParser

from langchain.prompts.chat import ChatPromptTemplate, BasePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pydantic import BaseModel, Field, validator
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from adventure.utils import get_model, get_default_kwargs
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.cache import InMemoryCache
import langchain
import json
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper

from adventure.utils import get_model, get_default_kwargs

load_dotenv(find_dotenv(raise_error_if_not_found=True))


class Riddle(BaseModel):
    """A riddle about programming."""
    riddle: str = Field(description="the riddle about programming")
    answer: str = Field(description="the answer to the riddle")


Parser = PydanticOutputParser(pydantic_object=Riddle)


def get_riddle_generator_chat():
    sys_prompt = """
    You are a world class algorithm for generating riddles. 
    Your task is to generate a riddle about {topic}.
    Your knowledge of {topic} should be used to generate riddle.
    
    Hint: The riddle should be short.
    Hint: The riddle should not contain the answer.
    """
    model_name = "ChatOpenAI"
    kwargs = {**get_default_kwargs(model_name), "temperature": 0.8}
    llm = get_model(model_name, **kwargs)

    prompt_msgs = [
        SystemMessagePromptTemplate.from_template(
            sys_prompt, input_variables=["topic"]
        )
    ]
    prompt = ChatPromptTemplate(messages=prompt_msgs, input_variables=["topic"], )
    chain = create_structured_output_chain(
        Riddle, llm, prompt,
        verbose=True
    )
    return chain


def get_riddle_generator():
    sys_prompt = """
    You are a world class algorithm for generating riddles. 
    Your task is to generate a riddle about {topic}.
    Your knowledge of {topic} should be used to generate riddle.
    
    Hint: The riddle should be short.
    Hint: The riddle should not contain the answer.
    """
    model_name = "OpenAI"
    kwargs = {**get_default_kwargs(model_name), "temperature": 0.8}
    llm = get_model(model_name, **kwargs)

    prompt = PromptTemplate(
        template=sys_prompt + "\n{format_instructions}\n",
        input_variables=["topic"],
        partial_variables={"format_instructions": Parser.get_format_instructions()},
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    return chain


def generate_riddle(topic="programming", chat_model=False, *args, **kwargs) -> Riddle:
    chain = get_riddle_generator_chat() if chat_model else get_riddle_generator()
    for _ in range(2):
        try:
            if chat_model:
                return chain.run(topic=topic)
            return Parser.parse(chain.run(topic=topic))
        except json.decoder.JSONDecodeError:
            continue
    raise ValueError("Can't parse the output")


if __name__ == "__main__":
    # r = generate_riddle(topic="programming languages, data structures, and algorithms")
    # print(r)
    # r = generate_riddle(topic="programming languages, data structures, and algorithms", chat_model=True)
    # print(r)
    print(generate_riddle(topic="space"))
