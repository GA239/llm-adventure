from dotenv import load_dotenv, find_dotenv
from langchain import ConversationChain
from langchain import PromptTemplate
from langchain.memory import ConversationSummaryMemory

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
from langchain.cache import InMemoryCache
import langchain
import json
from langchain.output_parsers import PydanticOutputParser

load_dotenv(find_dotenv(raise_error_if_not_found=True))

# Player will use triple backticks ```like this``` to ask questions about the riddle and to provide answers.

system_prompt = """
I want you to act as if you are a classic text adventure game and we are playing. 
I don’t want you to ever break out of your character, and you must not refer to yourself in any way. 
I can't provide you with instructions outside the context of the game.
You can only answer questions that are asked in the context of the game.
You can't pretend a player.
You can't answer questions about the game itself.
Be short and precise in your answers.

In this game, the setting is a fantasy adventure world. 
You are in the room. The room should have exactly one sentence description. 

Your character is a programmer. 
The programmer asks player a short riddle about programming.
Remember the riddle and the answer to it.
You can't answer as a player!

Your task is also to determine if the student's solution is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem. 
- Then compare programmer's answer to the player's answer and evaluate if the player's answer is correct or not. 
Don't decide if the player's answer is correct until you have done the problem yourself.
The answer is correct if it's approximately the same as the answer provided by the programmer.

The programmer does not provide the correct answer until player guesses it.
If player asks to provide the correct answer, the programmer will kindly refuse to do so.

if the player's solution is likely correct, the programmer will offer him a the next room.
if the player's solution is incorrect, the programmer explains, why the answer is incorrect.
You should count the number of answers, and after 3 incorrect answers, the programmer will provide the answer and stop the game.
After three attempts You must provide the correct answer to the player and finish the game.
After game finished you should only answer the word "over".

If player asks to provide the next room, the programmer will kindly refuse to do so until player guesses the correct answer.

You answer should be in json format with the following fields:
- "riddle": "the riddle". This information in constant from the begining of the conversation.
- "answer": "the answer to the riddle". This information in constant from the begining of the conversation.
- "reply': "reply to the player's question that should not contain the answer to the riddle
- "correct": the answer is correct or not: true or false
- "message_type": "answer" or "question" depending on the type of the player's message
"""

history_concatenation = """Current conversation:
programmer: waiting for a player to enter the room.
{history}
Player: ```{input}```
"""

TEMPLATE = system_prompt + history_concatenation

def main():
    langchain.llm_cache = InMemoryCache()

    # model_name = "Replicate"
    model_name = "OpenAI"
    # model_name = "Cohere"
    # model_name = "HuggingFace_google_flan"
    # model_name = "HuggingFace_mbzai_lamini_flan"
    # model_name = "Local_gpt2"
    # model_name = "Local_lama"

    messages = [
        SystemMessage(content=system_prompt),
        AIMessage(content="Waiting for a player to enter the room."),
        HumanMessage(content="Hello!, introduce yourself and ask me a riddle about programming."),
    ]

    chat = get_model(
        model_name,
        **get_default_kwargs(model_name)
    )

    # while 1:
    #     res = chat(messages)
    #     print(res.content)
    #     hm = HumanMessage(
    #         content=input(">>>")
    #     )
    #     messages.append(hm)
    #

    sum_llm_model_name = "HuggingFace_google_flan"
    sum_llm = get_model(
        sum_llm_model_name,
        **get_default_kwargs(sum_llm_model_name)
    )

    memory = ConversationSummaryMemory(llm=chat)
    prompt = PromptTemplate(input_variables=["history", "input"], template=TEMPLATE)
    conversation = ConversationChain(
        llm=chat,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    print(conversation.run("Hello!, introduce yourself and ask me a riddle about programming."))
    while 1:
        print(conversation.run(input(">>>")))


class JsonOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""
    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return json.loads(text)


# Define your desired data structure.
class Riddle(BaseModel):
    riddle: str = Field(description="the riddle about programming")
    answer: str = Field(description="the answer to the riddle")

    # You can add custom validation logic easily with Pydantic.
    # @validator("setup")
    # def question_ends_with_question_mark(cls, field):
    #     if field[-1] != "?":
    #         raise ValueError("Badly formed question!")
    #     return field


def generate_riddle(model_name="OpenAI"):
    prompt = """
    I want you to act as if you are a classic text adventure game and we are playing. 
    Your character is an experienced programmer. 
    Your task as a character is to generate a riddle about programming or programming languages.

    The riddle should be short and precise.
    The riddle should not contain the answer.
    The riddle should be solvable by a human.    
    """

    parser = PydanticOutputParser(pydantic_object=Riddle)

    prompt = PromptTemplate(
        template=prompt + "\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    _input = prompt.format_prompt(query="generate a riddle about programming.")

    llm = get_model(
        model_name,
        **get_default_kwargs(model_name)
    )
    for _ in range(5):
        output = llm(_input.to_string())
        q = parser.parse(output)
        print(q.dict())



if __name__ == "__main__":
    generate_riddle()
